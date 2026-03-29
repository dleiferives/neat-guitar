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

// Convert neat-python node ID to C++ node ID
// neat-python: inputs=-1..-n_inputs, outputs=0..n_outputs-1, hidden>=n_outputs
// C++:         inputs=0..n_inputs-1, outputs=n_inputs..n_inputs+n_outputs-1, hidden>=n_inputs+n_outputs
static int convert_node_id(int neat_id, int n_inputs, int n_outputs) {
    if (neat_id < 0) {
        // Input node: -1 -> 0, -2 -> 1, etc.
        return -neat_id - 1;
    } else if (neat_id < n_outputs) {
        // Output node: 0 -> n_inputs, 1 -> n_inputs+1, etc.
        return n_inputs + neat_id;
    } else {
        // Hidden node: shift by n_inputs
        return n_inputs + neat_id;
    }
}

// Convert neat-python genome representation to our Genome struct
Genome genome_from_neat_python(
    const std::vector<std::tuple<int, float, float>>& nodes,  // (key, bias, response)
    const std::vector<std::tuple<int, int, float, bool>>& conns  // (in, out, weight, enabled)
) {
    Genome g;
    g.id = 0;

    int n_in = g_cfg.n_inputs();
    int n_out = g_cfg.n_outputs();

    // Add input nodes (IDs 0 to n_in-1)
    for (int i = 0; i < n_in; ++i) {
        g.nodes.push_back({i, NodeType::INPUT, 0.0f});
    }

    // Add output nodes (IDs n_in to n_in+n_out-1)
    for (int i = 0; i < n_out; ++i) {
        g.nodes.push_back({n_in + i, NodeType::OUTPUT, 0.0f});
    }

    // Process neat-python nodes to get biases and find hidden nodes
    for (const auto& [key, bias, response] : nodes) {
        int cpp_id = convert_node_id(key, n_in, n_out);
        float effective_bias = bias * response;

        if (key < 0) {
            // Input node - no bias needed
            continue;
        } else if (key < n_out) {
            // Output node - update bias
            for (auto& n : g.nodes) {
                if (n.id == cpp_id) {
                    n.bias = effective_bias;
                    break;
                }
            }
        } else {
            // Hidden node - add it
            g.nodes.push_back({cpp_id, NodeType::HIDDEN, effective_bias});
        }
    }

    // Add connections with converted node IDs
    uint32_t innov = 1;
    for (const auto& [in_node, out_node, weight, enabled] : conns) {
        int cpp_in = convert_node_id(in_node, n_in, n_out);
        int cpp_out = convert_node_id(out_node, n_in, n_out);
        g.conns.push_back({cpp_in, cpp_out, weight, enabled, innov++});
    }

    return g;
}

// Evaluate a genome and return detailed results
py::dict evaluate_genome_py(
    const std::vector<std::tuple<int, float, float>>& nodes,
    const std::vector<std::tuple<int, int, float, bool>>& conns,
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
}
