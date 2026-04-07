#include "exact_solver.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

namespace dcgs {

namespace {

constexpr Real kDensityTolerance = 1e-12;

// NowSeconds provides monotonic wall-clock timing for exact solver profiling.
Real NowSeconds() {
  using clock = std::chrono::steady_clock;
  const auto now = clock::now().time_since_epoch();
  return std::chrono::duration<Real>(now).count();
}

// MixHash is a small helper for building a topology signature.
std::uint64_t MixHash(std::uint64_t seed, std::uint64_t value) {
  seed ^= value + 0x9e3779b97f4a7c15ull + (seed << 6) + (seed >> 2);
  return seed;
}

std::uint64_t ComputeTopologySignature(const TrackedGraph& tracked_graph) {
  // The exact solver is often invoked multiple times on the same tracked graph
  // with different checkpoint weights. A cheap topology signature lets the
  // workspace reuse the flow graph whenever only capacities change.
  std::uint64_t signature = 0xcbf29ce484222325ull;
  signature = MixHash(signature,
                      static_cast<std::uint64_t>(tracked_graph.local_to_global.size()));
  signature = MixHash(signature,
                      static_cast<std::uint64_t>(tracked_graph.edge_u.size()));
  for (NodeId node : tracked_graph.local_to_global) {
    signature = MixHash(signature, static_cast<std::uint64_t>(node));
  }
  for (std::size_t edge_index = 0; edge_index < tracked_graph.edge_u.size(); ++edge_index) {
    signature = MixHash(signature,
                        static_cast<std::uint64_t>(tracked_graph.edge_u[edge_index]));
    signature = MixHash(signature,
                        static_cast<std::uint64_t>(tracked_graph.edge_v[edge_index]));
  }
  return signature;
}

std::int64_t ChooseCapacityScale(Real total_weight,
                                 NodeCount num_local_nodes,
                                 const ExactSolverOptions& opts) {
  // The implementation solves an integer-capacity min-cut instance. This scale
  // attempts to preserve weight precision while keeping every derived capacity
  // comfortably inside int64_t.
  if (!(total_weight > 0.0)) return 1;

  const std::int64_t target_scale =
      std::max<std::int64_t>(1, opts.target_capacity_scale);
  const long double weight_ld = static_cast<long double>(total_weight);
  const long double node_factor =
      static_cast<long double>(std::max<NodeCount>(1, num_local_nodes));
  const long double denominator = 4.0L * weight_ld * node_factor;
  if (!(denominator > 0.0L)) return target_scale;

  const long double safe_scale_ld =
      static_cast<long double>(std::numeric_limits<std::int64_t>::max()) /
      denominator;
  if (!(safe_scale_ld >= 1.0L)) return 1;

  const std::int64_t safe_scale = static_cast<std::int64_t>(safe_scale_ld);
  return std::max<std::int64_t>(1, std::min(target_scale, safe_scale));
}

std::int64_t ScaleWeight(Real weight, std::int64_t scale) {
  if (!(weight > 0.0) || scale <= 0) return 0;
  const long double scaled_ld =
      static_cast<long double>(weight) * static_cast<long double>(scale);
  if (scaled_ld >=
      static_cast<long double>(std::numeric_limits<std::int64_t>::max())) {
    return std::numeric_limits<std::int64_t>::max();
  }
  const std::int64_t scaled = static_cast<std::int64_t>(std::llround(scaled_ld));
  return std::max<std::int64_t>(0, scaled);
}

bool BetterCandidateDensity(Real lhs_density,
                            const std::vector<NodeCount>& lhs_nodes,
                            Real rhs_density,
                            const std::vector<NodeCount>& rhs_nodes) {
  // Ties are broken deterministically so repeated runs with the same seed stay
  // reproducible even when multiple candidate cuts have the same density.
  if (lhs_density > rhs_density + kDensityTolerance) return true;
  if (rhs_density > lhs_density + kDensityTolerance) return false;
  if (lhs_nodes.size() != rhs_nodes.size()) return lhs_nodes.size() < rhs_nodes.size();
  return lhs_nodes < rhs_nodes;
}

std::uint32_t ComputeBinarySearchLimit(std::int64_t low,
                                       std::int64_t high,
                                       const ExactSolverOptions& opts) {
  // The solver now stops according to the actual integer search interval rather
  // than a hard-coded global iteration count.
  if (high <= low + 1) return 0;

  std::uint64_t remaining =
      static_cast<std::uint64_t>(high - low - 1);
  std::uint32_t dynamic_iters = 0;
  while (remaining > 0) {
    ++dynamic_iters;
    remaining >>= 1;
  }

  if (opts.max_binary_search_iters == 0) return dynamic_iters;
  return std::min(dynamic_iters, opts.max_binary_search_iters);
}

std::uint32_t BeginSelectionEpoch(std::size_t num_local_nodes,
                                  ExactSolverWorkspace* workspace) {
  // Epoch-based marking avoids clearing the selection array for each candidate cut.
  if (workspace->selected_local_epoch.size() != num_local_nodes) {
    workspace->selected_local_epoch.assign(num_local_nodes, 0);
    workspace->current_selected_epoch = 1;
  }
  if (workspace->current_selected_epoch == std::numeric_limits<std::uint32_t>::max()) {
    std::fill(workspace->selected_local_epoch.begin(),
              workspace->selected_local_epoch.end(), 0u);
    workspace->current_selected_epoch = 1;
  }
  return workspace->current_selected_epoch++;
}

Real ComputeDensity(const TrackedGraph& tracked_graph,
                    const std::vector<Real>& weights,
                    const std::vector<NodeCount>& local_nodes,
                    ExactSolverWorkspace* workspace) {
  if (!workspace || local_nodes.empty()) return 0.0;

  // Candidate sets are checked repeatedly during binary search. Epoch-based
  // marking avoids clearing a full membership array for every candidate cut.
  const std::size_t num_local_nodes = tracked_graph.local_to_global.size();
  const std::uint32_t epoch = BeginSelectionEpoch(num_local_nodes, workspace);
  for (NodeCount local_node : local_nodes) {
    if (local_node < num_local_nodes) {
      workspace->selected_local_epoch[local_node] = epoch;
    }
  }

  Real total_weight = 0.0;
  for (EdgeIndex edge_index = 0;
       edge_index < static_cast<EdgeIndex>(tracked_graph.edge_u.size());
       ++edge_index) {
    const NodeCount local_u = tracked_graph.edge_u[static_cast<std::size_t>(edge_index)];
    const NodeCount local_v = tracked_graph.edge_v[static_cast<std::size_t>(edge_index)];
    if (workspace->selected_local_epoch[local_u] != epoch ||
        workspace->selected_local_epoch[local_v] != epoch) {
      continue;
    }
    if (edge_index >= static_cast<EdgeIndex>(weights.size())) continue;
    total_weight +=
        std::max(static_cast<Real>(0.0),
                 weights[static_cast<std::size_t>(edge_index)]);
  }

  return total_weight / static_cast<Real>(local_nodes.size());
}

std::vector<NodeId> ConvertToGlobalNodes(const TrackedGraph& tracked_graph,
                                         const std::vector<NodeCount>& local_nodes) {
  // Solver internals work in local tracked-graph ids, but all public outputs
  // are reported in original graph ids.
  std::vector<NodeId> nodes;
  nodes.reserve(local_nodes.size());
  for (NodeCount local_node : local_nodes) {
    if (local_node < tracked_graph.local_to_global.size()) {
      nodes.push_back(tracked_graph.local_to_global[static_cast<std::size_t>(local_node)]);
    }
  }
  return nodes;
}

bool EnsureFlowTopology(const TrackedGraph& tracked_graph,
                        ExactSolverWorkspace* workspace) {
  if (!workspace) return false;

  // Only the terminal capacities depend on lambda. The source/sink structure
  // and the symmetric pair edges depend solely on the tracked graph topology,
  // so they are rebuilt only when the topology signature changes.
  const NodeCount num_local_nodes =
      static_cast<NodeCount>(tracked_graph.local_to_global.size());
  const EdgeIndex num_edges =
      static_cast<EdgeIndex>(tracked_graph.edge_u.size());
  const std::uint64_t topology_signature =
      ComputeTopologySignature(tracked_graph);

  if (workspace->topology_signature == topology_signature &&
      workspace->topology_local_nodes == num_local_nodes &&
      workspace->topology_edge_count == num_edges &&
      workspace->source_handles.size() == static_cast<std::size_t>(num_local_nodes) &&
      workspace->sink_handles.size() == static_cast<std::size_t>(num_local_nodes) &&
      workspace->edge_uv_handles.size() == static_cast<std::size_t>(num_edges) &&
      workspace->edge_vu_handles.size() == static_cast<std::size_t>(num_edges)) {
    return false;
  }

  const std::int32_t source = 0;
  const std::int32_t sink = static_cast<std::int32_t>(num_local_nodes) + 1;
  workspace->flow.Reset(sink + 1);

  workspace->source_handles.resize(static_cast<std::size_t>(num_local_nodes));
  workspace->sink_handles.resize(static_cast<std::size_t>(num_local_nodes));
  workspace->edge_uv_handles.resize(static_cast<std::size_t>(num_edges));
  workspace->edge_vu_handles.resize(static_cast<std::size_t>(num_edges));

  for (NodeCount local_node = 0; local_node < num_local_nodes; ++local_node) {
    const std::int32_t flow_node = static_cast<std::int32_t>(local_node) + 1;
    workspace->source_handles[static_cast<std::size_t>(local_node)] =
        workspace->flow.AddEdge(source, flow_node, 0);
    workspace->sink_handles[static_cast<std::size_t>(local_node)] =
        workspace->flow.AddEdge(flow_node, sink, 0);
  }

  for (EdgeIndex edge_index = 0; edge_index < num_edges; ++edge_index) {
    const std::int32_t local_u =
        static_cast<std::int32_t>(tracked_graph.edge_u[static_cast<std::size_t>(edge_index)]) + 1;
    const std::int32_t local_v =
        static_cast<std::int32_t>(tracked_graph.edge_v[static_cast<std::size_t>(edge_index)]) + 1;
    workspace->edge_uv_handles[static_cast<std::size_t>(edge_index)] =
        workspace->flow.AddEdge(local_u, local_v, 0);
    workspace->edge_vu_handles[static_cast<std::size_t>(edge_index)] =
        workspace->flow.AddEdge(local_v, local_u, 0);
  }

  workspace->topology_signature = topology_signature;
  workspace->topology_local_nodes = num_local_nodes;
  workspace->topology_edge_count = num_edges;
  return true;
}

void SetStaticBaseCapacities(const TrackedGraph& tracked_graph,
                             std::int64_t total_weight_scaled,
                             const std::vector<std::int64_t>& scaled_edge_weights,
                             ExactSolverWorkspace* workspace) {
  if (!workspace) return;

  // The source capacities and the pair-edge capacities stay fixed throughout
  // the binary search for one solve. Only node-to-sink capacities depend on
  // the current density guess lambda.
  const NodeCount num_local_nodes =
      static_cast<NodeCount>(tracked_graph.local_to_global.size());
  for (NodeCount local_node = 0; local_node < num_local_nodes; ++local_node) {
    workspace->flow.SetBaseCapacity(
        workspace->source_handles[static_cast<std::size_t>(local_node)],
        total_weight_scaled);
  }

  for (EdgeIndex edge_index = 0;
       edge_index < static_cast<EdgeIndex>(tracked_graph.edge_u.size());
       ++edge_index) {
    const std::int64_t scaled_weight =
        scaled_edge_weights[static_cast<std::size_t>(edge_index)];
    workspace->flow.SetBaseCapacity(
        workspace->edge_uv_handles[static_cast<std::size_t>(edge_index)],
        scaled_weight);
    workspace->flow.SetBaseCapacity(
        workspace->edge_vu_handles[static_cast<std::size_t>(edge_index)],
        scaled_weight);
  }
}

bool SolveDecision(const TrackedGraph& tracked_graph,
                   std::int64_t lambda_scaled,
                   std::int64_t total_weight_scaled,
                   const std::vector<std::int64_t>& scaled_degree,
                   ExactSolverWorkspace* workspace,
                   ExactSolverCallStats* stats) {
  if (!workspace) return false;

  // This is the weighted densest-subgraph decision reduction: a non-empty
  // source-side cut cheaper than the base cut witnesses density > lambda.
  const std::int32_t num_local_nodes =
      static_cast<std::int32_t>(tracked_graph.local_to_global.size());
  const std::int32_t source = 0;
  const std::int32_t sink = num_local_nodes + 1;

  for (std::int32_t local_node = 0; local_node < num_local_nodes; ++local_node) {
    const std::int64_t sink_cap =
        total_weight_scaled + 2 * lambda_scaled -
        scaled_degree[static_cast<std::size_t>(local_node)];
    workspace->flow.SetBaseCapacity(
        workspace->sink_handles[static_cast<std::size_t>(local_node)],
        std::max<std::int64_t>(0, sink_cap));
  }

  workspace->flow.ResetCapacities();
  if (stats) ++stats->flow_calls;
  const std::int64_t min_cut_value = workspace->flow.MaxFlow(source, sink);
  const std::int64_t base_cut =
      total_weight_scaled * static_cast<std::int64_t>(num_local_nodes);
  if (!(min_cut_value < base_cut)) return false;

  workspace->candidate_local_nodes.clear();
  const std::vector<std::uint8_t>& source_side =
      workspace->flow.MinCutSourceSide(source);
  for (std::int32_t local_node = 0; local_node < num_local_nodes; ++local_node) {
    if (source_side[static_cast<std::size_t>(local_node + 1)]) {
      workspace->candidate_local_nodes.push_back(static_cast<NodeCount>(local_node));
    }
  }
  return !workspace->candidate_local_nodes.empty();
}

}  // namespace

void DinicMaxflow::Reset(std::int32_t num_nodes) {
  // Reset discards the current residual graph and prepares storage for a new topology.
  num_nodes_ = num_nodes;
  graph_.assign(static_cast<std::size_t>(num_nodes_), {});
  level_.assign(static_cast<std::size_t>(num_nodes_), -1);
  next_edge_.assign(static_cast<std::size_t>(num_nodes_), 0);
  queue_.clear();
  queue_.reserve(static_cast<std::size_t>(num_nodes_));
  min_cut_source_side_.assign(static_cast<std::size_t>(num_nodes_), 0);
}

DinicEdgeHandle DinicMaxflow::AddEdge(std::int32_t from,
                                      std::int32_t to,
                                      std::int64_t cap) {
  // Each forward edge is created together with its reverse residual edge.
  if (cap < 0) cap = 0;
  const std::int32_t from_rev =
      static_cast<std::int32_t>(graph_[static_cast<std::size_t>(to)].size());
  const std::int32_t to_index =
      static_cast<std::int32_t>(graph_[static_cast<std::size_t>(from)].size());
  graph_[static_cast<std::size_t>(from)].push_back({to, from_rev, cap, cap});
  graph_[static_cast<std::size_t>(to)].push_back({from, to_index, 0, 0});
  return {from, to_index};
}

void DinicMaxflow::SetBaseCapacity(const DinicEdgeHandle& handle, std::int64_t cap) {
  // Base capacities are the values restored before every max-flow call.
  if (handle.from < 0 || handle.index < 0) return;
  if (cap < 0) cap = 0;
  DinicEdge& edge = graph_[static_cast<std::size_t>(handle.from)]
                          [static_cast<std::size_t>(handle.index)];
  edge.base_cap = cap;
  DinicEdge& reverse =
      graph_[static_cast<std::size_t>(edge.to)][static_cast<std::size_t>(edge.rev)];
  reverse.base_cap = 0;
}

void DinicMaxflow::ResetCapacities() {
  // ResetCapacities reuses the same topology for a new lambda-dependent solve.
  for (std::vector<DinicEdge>& edges : graph_) {
    for (DinicEdge& edge : edges) {
      edge.cap = edge.base_cap;
    }
  }
}

std::int64_t DinicMaxflow::Dfs(std::int32_t node,
                               std::int32_t sink,
                               std::int64_t pushed) {
  // Dfs augments one blocking flow path in the current level graph.
  if (node == sink || pushed == 0) return pushed;
  for (std::int32_t& edge_index = next_edge_[static_cast<std::size_t>(node)];
       edge_index <
       static_cast<std::int32_t>(graph_[static_cast<std::size_t>(node)].size());
       ++edge_index) {
    DinicEdge& edge =
        graph_[static_cast<std::size_t>(node)][static_cast<std::size_t>(edge_index)];
    if (edge.cap <= 0) continue;
    if (level_[static_cast<std::size_t>(edge.to)] !=
        level_[static_cast<std::size_t>(node)] + 1) {
      continue;
    }
    const std::int64_t flow = Dfs(edge.to, sink, std::min(pushed, edge.cap));
    if (flow <= 0) continue;
    edge.cap -= flow;
    DinicEdge& reverse =
        graph_[static_cast<std::size_t>(edge.to)][static_cast<std::size_t>(edge.rev)];
    reverse.cap += flow;
    return flow;
  }
  return 0;
}

std::int64_t DinicMaxflow::MaxFlow(std::int32_t source, std::int32_t sink) {
  // A compact Dinic implementation is sufficient here because the exact solver
  // is intended as a small/medium-instance subproblem solver, not as a generic
  // high-performance max-flow library.
  std::int64_t total_flow = 0;
  while (true) {
    std::fill(level_.begin(), level_.end(), -1);
    queue_.clear();
    level_[static_cast<std::size_t>(source)] = 0;
    queue_.push_back(source);

    for (std::size_t head = 0; head < queue_.size(); ++head) {
      const std::int32_t node = queue_[head];
      for (const DinicEdge& edge : graph_[static_cast<std::size_t>(node)]) {
        if (edge.cap <= 0) continue;
        if (level_[static_cast<std::size_t>(edge.to)] >= 0) continue;
        level_[static_cast<std::size_t>(edge.to)] =
            level_[static_cast<std::size_t>(node)] + 1;
        queue_.push_back(edge.to);
      }
    }

    if (level_[static_cast<std::size_t>(sink)] < 0) break;
    std::fill(next_edge_.begin(), next_edge_.end(), 0);
    while (true) {
      const std::int64_t pushed =
          Dfs(source, sink, std::numeric_limits<std::int64_t>::max());
      if (pushed <= 0) break;
      total_flow += pushed;
    }
  }
  return total_flow;
}

const std::vector<std::uint8_t>& DinicMaxflow::MinCutSourceSide(std::int32_t source) {
  // After MaxFlow finishes, residual reachability from source gives the source side.
  std::fill(min_cut_source_side_.begin(), min_cut_source_side_.end(), 0);
  queue_.clear();
  queue_.push_back(source);
  min_cut_source_side_[static_cast<std::size_t>(source)] = 1;

  for (std::size_t head = 0; head < queue_.size(); ++head) {
    const std::int32_t node = queue_[head];
    for (const DinicEdge& edge : graph_[static_cast<std::size_t>(node)]) {
      if (edge.cap <= 0) continue;
      if (min_cut_source_side_[static_cast<std::size_t>(edge.to)]) continue;
      min_cut_source_side_[static_cast<std::size_t>(edge.to)] = 1;
      queue_.push_back(edge.to);
    }
  }
  return min_cut_source_side_;
}

PeelingResult ExactMinCutSolver(const TrackedGraph& tracked_graph,
                                const std::vector<Real>& weights,
                                const ExactSolverOptions& opts,
                                ExactSolverWorkspace* workspace,
                                ExactSolverCallStats* stats) {
  PeelingResult result;
  ExactSolverCallStats local_stats;
  if (!stats) stats = &local_stats;
  *stats = ExactSolverCallStats{};
  stats->solve_calls = 1;

  const Real solver_start = NowSeconds();
  if (!workspace) {
    static thread_local ExactSolverWorkspace fallback_workspace;
    workspace = &fallback_workspace;
  }

  if (tracked_graph.local_to_global.empty()) {
    if (tracked_graph.fallback_node != kInvalidNode) {
      result.u_max.push_back(tracked_graph.fallback_node);
    }
    result.rho_max = 0.0;
    stats->solver_sec = NowSeconds() - solver_start;
    return result;
  }

  const NodeCount num_local_nodes =
      static_cast<NodeCount>(tracked_graph.local_to_global.size());
  Real total_weight = 0.0;
  for (EdgeIndex edge_index = 0;
       edge_index < static_cast<EdgeIndex>(tracked_graph.edge_u.size());
       ++edge_index) {
    if (edge_index >= static_cast<EdgeIndex>(weights.size())) continue;
    total_weight +=
        std::max(static_cast<Real>(0.0),
                 weights[static_cast<std::size_t>(edge_index)]);
  }

  const std::int64_t capacity_scale =
      ChooseCapacityScale(total_weight, num_local_nodes, opts);
  stats->capacity_scale_used = capacity_scale;
  workspace->scaled_edge_weights.assign(tracked_graph.edge_u.size(), 0);
  workspace->scaled_degree.assign(static_cast<std::size_t>(num_local_nodes), 0);

  std::int64_t total_weight_scaled = 0;
  for (EdgeIndex edge_index = 0;
       edge_index < static_cast<EdgeIndex>(tracked_graph.edge_u.size());
       ++edge_index) {
    if (edge_index >= static_cast<EdgeIndex>(weights.size())) continue;

    const Real weight =
        std::max(static_cast<Real>(0.0),
                 weights[static_cast<std::size_t>(edge_index)]);
    const std::int64_t scaled_weight = ScaleWeight(weight, capacity_scale);
    workspace->scaled_edge_weights[static_cast<std::size_t>(edge_index)] = scaled_weight;
    if (scaled_weight <= 0) continue;

    const NodeCount local_u = tracked_graph.edge_u[static_cast<std::size_t>(edge_index)];
    const NodeCount local_v = tracked_graph.edge_v[static_cast<std::size_t>(edge_index)];
    workspace->scaled_degree[static_cast<std::size_t>(local_u)] += scaled_weight;
    workspace->scaled_degree[static_cast<std::size_t>(local_v)] += scaled_weight;
    total_weight_scaled += scaled_weight;
  }

  if (total_weight_scaled <= 0) {
    result.u_max = tracked_graph.local_to_global;
    if (result.u_max.empty() && tracked_graph.fallback_node != kInvalidNode) {
      result.u_max.push_back(tracked_graph.fallback_node);
    }
    result.rho_max = 0.0;
    stats->solver_sec = NowSeconds() - solver_start;
    return result;
  }

  stats->topology_rebuilt = EnsureFlowTopology(tracked_graph, workspace);
  SetStaticBaseCapacities(tracked_graph, total_weight_scaled,
                          workspace->scaled_edge_weights, workspace);

  workspace->best_local_nodes.clear();
  Real best_density = 0.0;

  auto consider_candidate = [&]() {
    if (workspace->candidate_local_nodes.empty()) return;
    const Real density = ComputeDensity(tracked_graph,
                                        weights,
                                        workspace->candidate_local_nodes,
                                        workspace);
    if (workspace->best_local_nodes.empty() ||
        BetterCandidateDensity(density,
                               workspace->candidate_local_nodes,
                               best_density,
                               workspace->best_local_nodes)) {
      best_density = density;
      workspace->best_local_nodes = workspace->candidate_local_nodes;
    }
  };

  std::int64_t low = 0;
  std::int64_t high = total_weight_scaled + 1;
  const std::uint32_t iter_limit =
      ComputeBinarySearchLimit(low, high, opts);
  std::uint32_t iterations = 0;
  // This is an engineering exact solver based on repeated min-cut decisions and
  // binary search over lambda. It intentionally avoids the added complexity of
  // a full parametric min-cut implementation.
  while (low + 1 < high && iterations < iter_limit) {
    const std::int64_t mid = low + (high - low) / 2;
    if (SolveDecision(tracked_graph, mid, total_weight_scaled,
                      workspace->scaled_degree, workspace, stats)) {
      low = mid;
      consider_candidate();
    } else {
      high = mid;
    }
    ++iterations;
  }
  stats->binary_search_iters = iterations;

  if (SolveDecision(tracked_graph, low, total_weight_scaled,
                    workspace->scaled_degree, workspace, stats)) {
    consider_candidate();
  }

  if (workspace->best_local_nodes.empty()) {
    result.u_max = tracked_graph.local_to_global;
    result.rho_max = 0.0;
    if (result.u_max.empty() && tracked_graph.fallback_node != kInvalidNode) {
      result.u_max.push_back(tracked_graph.fallback_node);
    }
    stats->solver_sec = NowSeconds() - solver_start;
    return result;
  }

  result.u_max = ConvertToGlobalNodes(tracked_graph, workspace->best_local_nodes);
  result.rho_max = best_density;
  if (result.u_max.empty() && tracked_graph.fallback_node != kInvalidNode) {
    result.u_max.push_back(tracked_graph.fallback_node);
  }
  stats->solver_sec = NowSeconds() - solver_start;
  return result;
}

PeelingResult ExactMinCutSolver(const TrackedGraph& tracked_graph,
                                const std::vector<Real>& weights,
                                const ExactSolverOptions& opts,
                                ExactSolverWorkspace* workspace) {
  return ExactMinCutSolver(tracked_graph, weights, opts, workspace, nullptr);
}

PeelingResult ExactMinCutSolver(const TrackedGraph& tracked_graph,
                                const std::vector<Real>& weights,
                                ExactSolverWorkspace* workspace,
                                ExactSolverCallStats* stats) {
  ExactSolverOptions opts;
  return ExactMinCutSolver(tracked_graph, weights, opts, workspace, stats);
}

PeelingResult ExactMinCutSolver(const TrackedGraph& tracked_graph,
                                const std::vector<Real>& weights,
                                ExactSolverWorkspace* workspace) {
  return ExactMinCutSolver(tracked_graph, weights, workspace, nullptr);
}

PeelingResult ExactMinCutSolver(const TrackedGraph& tracked_graph,
                                const std::vector<Real>& weights) {
  ExactSolverWorkspace workspace;
  return ExactMinCutSolver(tracked_graph, weights, &workspace, nullptr);
}

}  // namespace dcgs
