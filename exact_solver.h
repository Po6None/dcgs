#ifndef EXACT_SOLVER_H_
#define EXACT_SOLVER_H_

#include "peeling.h"

#include <cstdint>
#include <vector>

namespace dcgs {

// ExactSolverOptions controls the integer-capacity exact min-cut solver.
struct ExactSolverOptions {
  std::uint32_t max_binary_search_iters = 0;
  std::int64_t target_capacity_scale = 1000000000ll;
};

// ExactSolverCallStats captures the cost of one exact solve call.
struct ExactSolverCallStats {
  std::uint64_t solve_calls = 0;
  std::uint64_t flow_calls = 0;
  std::uint32_t binary_search_iters = 0;
  std::int64_t capacity_scale_used = 0;
  bool topology_rebuilt = false;
  Real solver_sec = 0.0;
};

// DinicEdge is one directed residual edge in the internal max-flow graph.
struct DinicEdge {
  std::int32_t to = 0;
  std::int32_t rev = 0;
  std::int64_t base_cap = 0;
  std::int64_t cap = 0;
};

// DinicEdgeHandle identifies a mutable edge so capacities can be updated
// without rebuilding the entire flow graph topology.
struct DinicEdgeHandle {
  std::int32_t from = -1;
  std::int32_t index = -1;
};

// DinicMaxflow is the internal max-flow engine used by the exact solver.
class DinicMaxflow {
 public:
  // Reset rebuilds the residual graph for a new node count.
  void Reset(std::int32_t num_nodes);
  // AddEdge inserts a directed edge and returns a handle to its forward arc.
  DinicEdgeHandle AddEdge(std::int32_t from, std::int32_t to, std::int64_t cap);
  // SetBaseCapacity changes the stored base capacity of a previously added arc.
  void SetBaseCapacity(const DinicEdgeHandle& handle, std::int64_t cap);
  // ResetCapacities restores every residual capacity to its stored base value.
  void ResetCapacities();
  // MaxFlow computes the s-t max flow for the current residual graph.
  std::int64_t MaxFlow(std::int32_t source, std::int32_t sink);
  // MinCutSourceSide returns the nodes reachable from source in the residual graph.
  const std::vector<std::uint8_t>& MinCutSourceSide(std::int32_t source);

 private:
  std::int64_t Dfs(std::int32_t node,
                   std::int32_t sink,
                   std::int64_t pushed);

  std::int32_t num_nodes_ = 0;
  std::vector<std::vector<DinicEdge>> graph_;
  std::vector<std::int32_t> level_;
  std::vector<std::int32_t> next_edge_;
  std::vector<std::int32_t> queue_;
  std::vector<std::uint8_t> min_cut_source_side_;
};

// ExactSolverWorkspace stores reusable flow topology and scratch buffers across
// repeated exact solves on the same tracked graph.
struct ExactSolverWorkspace {
  DinicMaxflow flow;
  std::vector<std::int64_t> scaled_edge_weights;
  std::vector<std::int64_t> scaled_degree;
  std::vector<std::uint32_t> selected_local_epoch;
  std::uint32_t current_selected_epoch = 1;
  std::vector<NodeCount> candidate_local_nodes;
  std::vector<NodeCount> best_local_nodes;

  std::uint64_t topology_signature = 0;
  NodeCount topology_local_nodes = 0;
  EdgeIndex topology_edge_count = 0;
  std::vector<DinicEdgeHandle> source_handles;
  std::vector<DinicEdgeHandle> sink_handles;
  std::vector<DinicEdgeHandle> edge_uv_handles;
  std::vector<DinicEdgeHandle> edge_vu_handles;
};

// ExactMinCutSolver solves the weighted densest-subgraph subproblem exactly on
// the tracked graph using a min-cut reduction.
PeelingResult ExactMinCutSolver(const TrackedGraph& tracked_graph,
                                const std::vector<Real>& weights,
                                const ExactSolverOptions& opts,
                                ExactSolverWorkspace* workspace,
                                ExactSolverCallStats* stats);

// Convenience overloads with default options and/or temporary workspaces.
PeelingResult ExactMinCutSolver(const TrackedGraph& tracked_graph,
                                const std::vector<Real>& weights,
                                const ExactSolverOptions& opts,
                                ExactSolverWorkspace* workspace);

PeelingResult ExactMinCutSolver(const TrackedGraph& tracked_graph,
                                const std::vector<Real>& weights,
                                ExactSolverWorkspace* workspace,
                                ExactSolverCallStats* stats);

PeelingResult ExactMinCutSolver(const TrackedGraph& tracked_graph,
                                const std::vector<Real>& weights,
                                ExactSolverWorkspace* workspace);

PeelingResult ExactMinCutSolver(const TrackedGraph& tracked_graph,
                                const std::vector<Real>& weights);

}  // namespace dcgs

#endif  // EXACT_SOLVER_H_
