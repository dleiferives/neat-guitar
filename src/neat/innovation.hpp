#pragma once
#include <cstdint>
#include <tuple>
#include <unordered_map>

// Tracks global innovation numbers and node IDs for NEAT.
// Connection innovations are globally unique per (in_node, out_node) pair.
// Within a single generation, duplicate structural mutations get the same ID.
struct InnovationTracker {
    uint32_t next_innov   = 1;
    int      next_node_id = 0;  // caller sets this after building initial nodes

    struct PairHash {
        size_t operator()(std::pair<int,int> p) const noexcept {
            return std::hash<uint64_t>{}(((uint64_t)(uint32_t)p.first << 32)
                                        | (uint32_t)p.second);
        }
    };

    // Within-generation dedup: same (in,out) → same innov this generation
    std::unordered_map<std::pair<int,int>, uint32_t, PairHash> gen_conn_map;
    // Within-generation dedup: same split_innov → same new node id
    std::unordered_map<uint32_t, int> gen_split_map;

    // Get or assign an innovation number for a new connection.
    uint32_t get_conn(int in_node, int out_node) {
        auto key = std::make_pair(in_node, out_node);
        auto it  = gen_conn_map.find(key);
        if (it != gen_conn_map.end()) return it->second;
        uint32_t inv      = next_innov++;
        gen_conn_map[key] = inv;
        return inv;
    }

    // Get or assign ids/innovations for splitting a connection.
    // Returns {new_node_id, innov_in_to_new, innov_new_to_out}.
    std::tuple<int, uint32_t, uint32_t> get_split(uint32_t split_innov,
                                                    int in_node, int out_node) {
        auto it = gen_split_map.find(split_innov);
        int  nid;
        if (it != gen_split_map.end()) {
            nid = it->second;
        } else {
            nid                        = next_node_id++;
            gen_split_map[split_innov] = nid;
        }
        return { nid, get_conn(in_node, nid), get_conn(nid, out_node) };
    }

    // Call at the start of each generation.
    void reset_generation() {
        gen_conn_map.clear();
        gen_split_map.clear();
    }
};
