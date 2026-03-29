// bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "fitness.hpp"
#include "neat/config.hpp"
#include "neat/genome.hpp"
#include "neat/network.hpp"

namespace py = pybind11;

// Global data cache (loaded once)
static std::vector<RecordingFrames> g_data;
static NeatConfig g_cfg;
static bool g_loaded = false;

bool load_data(const std::string& data_dir) {
    if (g_loaded) return true;

    g_data = load_or_compute_frames(data_dir, g_cfg);
    if (g_data.empty()) return false;

    // Sort alphabetically (same as C++ train)
    std::sort(g_data.begin(), g_data.end(),
              [](const RecordingFrames& a, const RecordingFrames& b) {
                  return a.name < b.name;
              });

    g_loaded = true;
    return true;
}

int get_n_inputs() { return g_cfg.n_inputs(); }
int get_n_outputs() { return g_cfg.n_outputs(); }
int get_num_recordings() { return (int)g_data.size(); }
float get_theoretical_max() { return racing_theoretical_max(g_data); }

std::vector<std::string> get_recording_names() {
    std::vector<std::string> names;
    for (const auto& rf : g_data) names.push_back(rf.name);
    return names;
}

// Convert neat-python genome representation to our Genome struct
// neat-python gives us: nodes dict, connections dict
Genome genome_from_neat_python(
    const std::vector<std::tuple<int, float, float, float>>& nodes,  // (key, bias, response, activation)
    const std::vector<std::tuple<int, int, float, bool, int>>& conns  // (in, out, weight, enabled, innov)
) {
    Genome g;
    g.id = 0;

    int n_in = g_cfg.n_inputs();
    int n_out = g_cfg.n_outputs();

    // Add input nodes
    for (int i = 0; i < n_in; ++i) {
        g.nodes.push_back({i, NodeType::INPUT, 0.0f});
    }

    // Add output nodes
    for (int i = 0; i < n_out; ++i) {
        g.nodes.push_back({n_in + i, NodeType::OUTPUT, 0.0f});
    }

    // Add hidden nodes from neat-python (keys >= n_in + n_out are hidden)
    for (const auto& [key, bias, response, activation] : nodes) {
        if (key >= 0 && key < n_in) continue;  // input node
        if (key >= n_in && key < n_in + n_out) {
            // output node - update bias
            for (auto& n : g.nodes) {
                if (n.id == key) {
                    n.bias = bias * response;
                    break;
                }
            }
        } else {
            // hidden node
            g.nodes.push_back({key, NodeType::HIDDEN, bias * response});
        }
    }

    // Add connections
    for (const auto& [in_node, out_node, weight, enabled, innov] : conns) {
        g.conns.push_back({in_node, out_node, weight, enabled, (uint32_t)innov});
    }

    return g;
}

// Evaluate a genome and return detailed results
py::dict evaluate_genome_py(
    const std::vector<std::tuple<int, float, float, float>>& nodes,
    const std::vector<std::tuple<int, int, float, bool, int>>& conns,
    int start_file_idx = 0,
    float threshold = 0.5f,
    float kill_threshold = 0.2f,
    float window_secs = 5.0f
) {
    if (!g_loaded) {
        throw std::runtime_error("Data not loaded. Call load_data() first.");
    }

    Genome g = genome_from_neat_python(nodes, conns);
    auto result = evaluate_genome_racing_detailed(g, g_data, g_cfg, start_file_idx,
                                                   threshold, kill_threshold, window_secs);

    py::dict d;
    d["fitness"] = result.fitness;
    d["frames_processed"] = result.frames_processed;
    d["total_frames"] = result.total_frames;
    d["files_completed"] = result.files_completed;
    d["total_files"] = result.total_files;
    d["avg_accuracy"] = result.avg_accuracy;
    d["n_nodes"] = g.nodes.size();
    d["n_conns"] = g.conns.size();
    return d;
}

// Batch evaluate multiple genomes (more efficient)
std::vector<py::dict> evaluate_genomes_batch(
    const std::vector<std::pair<
        std::vector<std::tuple<int, float, float, float>>,
        std::vector<std::tuple<int, int, float, bool, int>>
    >>& genomes,
    int start_file_idx = 0
) {
    std::vector<py::dict> results;
    results.reserve(genomes.size());

    for (const auto& [nodes, conns] : genomes) {
        results.push_back(evaluate_genome_py(nodes, conns, start_file_idx));
    }

    return results;
}

PYBIND11_MODULE(neat_fitness, m) {
    m.doc() = "NEAT fitness evaluation using C++ backend";

    m.def("load_data", &load_data, "Load and cache recording data",
          py::arg("data_dir"));

    m.def("get_n_inputs", &get_n_inputs);
    m.def("get_n_outputs", &get_n_outputs);
    m.def("get_num_recordings", &get_num_recordings);
    m.def("get_theoretical_max", &get_theoretical_max);
    m.def("get_recording_names", &get_recording_names);

    m.def("evaluate_genome", &evaluate_genome_py,
          "Evaluate a single genome",
          py::arg("nodes"),
          py::arg("conns"),
          py::arg("start_file_idx") = 0,
          py::arg("threshold") = 0.5f,
          py::arg("kill_threshold") = 0.2f,
          py::arg("window_secs") = 5.0f);

    m.def("evaluate_genomes_batch", &evaluate_genomes_batch,
          "Evaluate multiple genomes",
          py::arg("genomes"),
          py::arg("start_file_idx") = 0);
}
