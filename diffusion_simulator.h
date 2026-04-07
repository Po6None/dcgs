#ifndef DIFFUSION_SIMULATOR_H_
#define DIFFUSION_SIMULATOR_H_

#include "common.h"
#include "graph.h"

#include <random>
#include <vector>

namespace dcgs {

// DiffusionWorkspace holds reusable per-realization buffers for both IC and LT
// simulation paths. Reuse avoids repeated O(n) allocations across samples.
struct DiffusionWorkspace {
  std::vector<std::uint8_t> active;
  std::vector<NodeId> frontier;
  std::vector<NodeId> activated_nodes;

  std::vector<std::uint8_t> lt_seen;
  std::vector<NodeId> lt_touched_nodes;
  std::vector<Real> lt_sum_in;
  std::vector<Real> lt_theta;
};

// SimulateDiffusion dispatches to the IC or LT simulator selected by opts and
// returns the activated nodes for one realization.
const std::vector<NodeId>& SimulateDiffusion(const Graph& graph,
                                             const std::vector<NodeId>& seeds,
                                             const RuntimeOptions& opts,
                                             DiffusionWorkspace* workspace,
                                             std::mt19937_64* rng);

// SimulateIC runs one independent-cascade realization.
const std::vector<NodeId>& SimulateIC(const Graph& graph,
                                      const std::vector<NodeId>& seeds,
                                      DiffusionWorkspace* workspace,
                                      std::mt19937_64* rng);

// SimulateLT runs one linear-threshold realization.
const std::vector<NodeId>& SimulateLT(const Graph& graph,
                                      const std::vector<NodeId>& seeds,
                                      DiffusionWorkspace* workspace,
                                      std::mt19937_64* rng);

// The overloads below create a temporary workspace for convenience callers.
std::vector<NodeId> SimulateDiffusion(const Graph& graph,
                                      const std::vector<NodeId>& seeds,
                                      const RuntimeOptions& opts,
                                      std::mt19937_64* rng);
std::vector<NodeId> SimulateIC(const Graph& graph,
                               const std::vector<NodeId>& seeds,
                               std::mt19937_64* rng);
std::vector<NodeId> SimulateLT(const Graph& graph,
                               const std::vector<NodeId>& seeds,
                               std::mt19937_64* rng);

}  // namespace dcgs

#endif  // DIFFUSION_SIMULATOR_H_
