#ifndef CERTIFIED_H_
#define CERTIFIED_H_

#include "diffusion_simulator.h"
#include "exact_solver.h"
#include "graph.h"
#include "peeling.h"
#include "pilot.h"
#include "tracked_graph.h"

#include <cstdint>
#include <random>
#include <vector>

namespace dcgs {

// CertifiedCheckpointTrace records the summary statistics of one certified
// checkpoint, including optional exact-solver costs.
struct CertifiedCheckpointTrace {
  SampleCount t = 0;
  Real radius = 0.0;
  Real rho_tilde = 0.0;
  Real rho_lower = 0.0;
  Real rho_upper = 0.0;
  NodeCount u_hat_size = 0;
  NodeCount u_plus_size = 0;
  Real opt_sec = 0.0;
  std::uint64_t exact_solve_calls = 0;
  std::uint64_t exact_flow_calls = 0;
  Real exact_avg_flow_calls = 0.0;
  std::uint64_t exact_binary_search_iters = 0;
  Real exact_avg_binary_search_iters = 0.0;
  std::uint64_t exact_topology_rebuilds = 0;
  std::int64_t exact_capacity_scale_last = 0;
  std::int64_t exact_capacity_scale_max = 0;
  Real exact_solver_sec = 0.0;
  Real exact_avg_solver_time_sec = 0.0;
  bool early_stop = false;
};

// CertifiedFinalExactTrace records the exact-solver work performed by the final
// solve when the pipeline does not stop at an earlier checkpoint.
struct CertifiedFinalExactTrace {
  bool used = false;
  SampleCount t = 0;
  NodeCount u_hat_size = 0;
  Real rho = 0.0;
  Real opt_sec = 0.0;
  std::uint64_t exact_solve_calls = 0;
  std::uint64_t exact_flow_calls = 0;
  Real exact_avg_flow_calls = 0.0;
  std::uint64_t exact_binary_search_iters = 0;
  Real exact_avg_binary_search_iters = 0.0;
  std::uint64_t exact_topology_rebuilds = 0;
  std::int64_t exact_capacity_scale_last = 0;
  std::int64_t exact_capacity_scale_max = 0;
  Real exact_solver_sec = 0.0;
  Real exact_avg_solver_time_sec = 0.0;
};

// CertifiedResult is the full return object of the certified stage.
struct CertifiedResult {
  std::vector<NodeId> u_hat;
  std::vector<CertifiedCheckpointTrace> checkpoint_trace;
  CertifiedFinalExactTrace final_exact_trace;
  bool early_stop = false;
  SampleCount t_stop = 0;
  std::uint32_t checkpoints = 0;
  std::uint32_t max_checkpoints = 0;
  SampleCount first_checkpoint_t = 0;
  SampleCount t_max = 0;
  Real rho_final = 0.0;
  Real sampling_sec = 0.0;
  Real checkpoint_opt_sec = 0.0;
  std::uint64_t exact_solve_calls = 0;
  std::uint64_t exact_flow_calls = 0;
  std::uint64_t exact_binary_search_iters = 0;
  std::uint64_t exact_topology_rebuilds = 0;
  std::int64_t exact_capacity_scale_last = 0;
  std::int64_t exact_capacity_scale_max = 0;
  Real exact_solver_sec = 0.0;
};

// CertifiedWorkspace holds reusable sampling, solver, and density-evaluation
// buffers for repeated certified-stage invocations.
struct CertifiedWorkspace {
  const TrackedGraph* tracked_graph = nullptr;
  DiffusionWorkspace diffusion_workspace;
  PeelingWorkspace peeling_workspace;
  ExactSolverWorkspace exact_workspace;

  std::vector<std::uint8_t> tracked_node_mask;
  std::vector<std::uint8_t> active_mark;

  std::vector<SampleCount> counters;
  std::vector<Real> weight_buffer;

  std::vector<std::uint8_t> local_membership;
  std::vector<NodeCount> touched_local_nodes;
};

// CertifiedStage runs the standard early-stopping certified procedure on a
// canonical tracked graph.
CertifiedResult CertifiedStage(const Graph& graph,
                               const std::vector<NodeId>& seeds,
                               const TrackedGraph& tracked_graph,
                               Real delta1,
                               const RuntimeOptions& opts,
                               CertifiedWorkspace* workspace,
                               std::mt19937_64* rng);

// Convenience overload that uses a temporary CertifiedWorkspace.
CertifiedResult CertifiedStage(const Graph& graph,
                               const std::vector<NodeId>& seeds,
                               const TrackedGraph& tracked_graph,
                               Real delta1,
                               const RuntimeOptions& opts,
                               std::mt19937_64* rng);

// CertifiedStageCappedNoEarlyStop runs the same certified estimator but forces
// sampling to continue until sample_cap instead of stopping early.
CertifiedResult CertifiedStageCappedNoEarlyStop(
    const Graph& graph,
    const std::vector<NodeId>& seeds,
    const TrackedGraph& tracked_graph,
    Real delta1,
    SampleCount sample_cap,
    const RuntimeOptions& opts,
    CertifiedWorkspace* workspace,
    std::mt19937_64* rng);

CertifiedResult CertifiedStageCappedNoEarlyStop(
    const Graph& graph,
    const std::vector<NodeId>& seeds,
    const TrackedGraph& tracked_graph,
    Real delta1,
    SampleCount sample_cap,
    const RuntimeOptions& opts,
    std::mt19937_64* rng);

}  // namespace dcgs

#endif  // CERTIFIED_H_
