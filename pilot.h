#ifndef PILOT_H_
#define PILOT_H_

#include "common.h"
#include "diffusion_simulator.h"
#include "graph.h"
#include "tracked_graph.h"

#include <cstdint>
#include <random>
#include <vector>

namespace dcgs {

// PilotResult contains the canonical tracked graph E_hat_tau and the pilot
// sample size that produced it.
struct PilotResult {
  TrackedGraph tracked_graph;
  SampleCount t0 = 0;
  Real elapsed_sec = 0.0;
};

// PilotWorkspace owns the reusable buffers for pilot-stage sampling and exact
// pair deduplication under a memory budget.
struct PilotWorkspace {
  DiffusionWorkspace diffusion_workspace;

  std::vector<std::uint8_t> is_seed;
  std::vector<NodeId> filtered_active_nodes;

  std::vector<std::uint64_t> pair_buffer;
  std::vector<std::uint64_t> chunk_buffer;
  std::vector<std::uint64_t> global_unique_keys;
  std::vector<std::uint64_t> merge_buffer;

  std::vector<std::uint8_t> tracked_node_seen;
  SampleCount tracked_nodes_count = 0;
};

// PilotStage discovers the tracked pair set E_hat_tau using the pilot sample
// bound derived from tau and delta0.
PilotResult PilotStage(const Graph& graph,
                       const std::vector<NodeId>& seeds,
                       Real delta0,
                       const RuntimeOptions& opts,
                       PilotWorkspace* workspace,
                       std::mt19937_64* rng);

// Convenience overload that uses a temporary PilotWorkspace.
PilotResult PilotStage(const Graph& graph,
                       const std::vector<NodeId>& seeds,
                       Real delta0,
                       const RuntimeOptions& opts,
                       std::mt19937_64* rng);

}  // namespace dcgs

#endif  // PILOT_H_
