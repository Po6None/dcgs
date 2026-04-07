#include "peeling.h"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

namespace dcgs {

namespace {

// GetWorkspaceOrFallback lets callers skip explicit workspace management.
PeelingWorkspace* GetWorkspaceOrFallback(PeelingWorkspace* workspace) {
  static thread_local PeelingWorkspace fallback_workspace;
  return workspace ? workspace : &fallback_workspace;
}

// HeapEntryLess turns std::make_heap into a min-heap keyed by current degree.
struct HeapEntryLess {
  bool operator()(const std::pair<Real, NodeCount>& lhs,
                  const std::pair<Real, NodeCount>& rhs) const {
    if (lhs.first != rhs.first) return lhs.first > rhs.first;
    return lhs.second > rhs.second;
  }
};

constexpr Real kDegreeTolerance = 1e-12;

// EnsureWorkspaceCapacity resets the mutable peeling state for one solve.
void EnsureWorkspaceCapacity(NodeCount num_local_nodes, PeelingWorkspace* workspace) {
  workspace->alive.assign(static_cast<std::size_t>(num_local_nodes), 1);
  workspace->degree.assign(static_cast<std::size_t>(num_local_nodes), 0.0);
  workspace->removed_order.clear();
  workspace->removed_order.reserve(static_cast<std::size_t>(num_local_nodes));
  workspace->heap_entries.clear();
  workspace->heap_entries.reserve(static_cast<std::size_t>(num_local_nodes));
}

}  // namespace

PeelingResult WeightedPeeling(const TrackedGraph& prepared_graph,
                              const std::vector<Real>& weights,
                              PeelingWorkspace* workspace) {
  // The implementation is the standard weighted min-degree peeling heuristic
  // on the tracked graph induced by certified or pilot output.
  PeelingResult result;
  if (prepared_graph.local_to_global.empty()) {
    if (prepared_graph.fallback_node != kInvalidNode) {
      result.u_max.push_back(prepared_graph.fallback_node);
    }
    result.rho_max = 0.0;
    return result;
  }

  const NodeCount num_local_nodes =
      static_cast<NodeCount>(prepared_graph.local_to_global.size());
  workspace = GetWorkspaceOrFallback(workspace);
  EnsureWorkspaceCapacity(num_local_nodes, workspace);

  Real current_weight_sum = 0.0;
  for (EdgeIndex edge_index = 0;
       edge_index < static_cast<EdgeIndex>(prepared_graph.edge_u.size());
       ++edge_index) {
    if (edge_index >= static_cast<EdgeIndex>(weights.size())) continue;
    const Real weight =
        std::max(static_cast<Real>(0.0), weights[static_cast<std::size_t>(edge_index)]);
    if (weight <= 0.0) continue;

    const NodeCount local_u = prepared_graph.edge_u[static_cast<std::size_t>(edge_index)];
    const NodeCount local_v = prepared_graph.edge_v[static_cast<std::size_t>(edge_index)];
    workspace->degree[local_u] += weight;
    workspace->degree[local_v] += weight;
    current_weight_sum += weight;
  }

  HeapEntryLess heap_less;
  for (NodeCount local_node = 0; local_node < num_local_nodes; ++local_node) {
    workspace->heap_entries.push_back({workspace->degree[local_node], local_node});
  }
  std::make_heap(workspace->heap_entries.begin(), workspace->heap_entries.end(), heap_less);

  NodeCount alive_count = num_local_nodes;
  Real best_density = current_weight_sum / static_cast<Real>(alive_count);
  EdgeIndex best_removed_count = 0;

  while (alive_count > 1) {
    std::pair<Real, NodeCount> top_entry = {0.0, 0};
    bool found = false;
    while (!workspace->heap_entries.empty()) {
      std::pop_heap(workspace->heap_entries.begin(), workspace->heap_entries.end(), heap_less);
      top_entry = workspace->heap_entries.back();
      workspace->heap_entries.pop_back();

      const NodeCount local_node = top_entry.second;
      if (!workspace->alive[local_node]) continue;
      if (std::fabs(top_entry.first - workspace->degree[local_node]) > kDegreeTolerance) continue;
      found = true;
      break;
    }
    if (!found) break;

    const NodeCount local_u = top_entry.second;
    workspace->alive[local_u] = 0;
    --alive_count;
    workspace->removed_order.push_back(local_u);

    Real removed_weight = 0.0;
    for (EdgeOffset pos = prepared_graph.offsets[local_u];
         pos < prepared_graph.offsets[static_cast<std::size_t>(local_u) + 1];
         ++pos) {
      const NodeCount local_v = prepared_graph.neighbors[static_cast<std::size_t>(pos)];
      if (!workspace->alive[local_v]) continue;

      const EdgeIndex edge_index =
          prepared_graph.incident_positions[static_cast<std::size_t>(pos)];
      if (edge_index >= static_cast<EdgeIndex>(weights.size())) continue;

      const Real weight =
          std::max(static_cast<Real>(0.0), weights[static_cast<std::size_t>(edge_index)]);
      if (weight <= 0.0) continue;

      workspace->degree[local_v] -= weight;
      removed_weight += weight;
      workspace->heap_entries.push_back({workspace->degree[local_v], local_v});
      std::push_heap(workspace->heap_entries.begin(), workspace->heap_entries.end(), heap_less);
    }

    current_weight_sum -= removed_weight;
    if (alive_count > 0) {
      const Real density = current_weight_sum / static_cast<Real>(alive_count);
      if (density > best_density) {
        best_density = density;
        best_removed_count =
            static_cast<EdgeIndex>(workspace->removed_order.size());
      }
    }
  }

  std::vector<std::uint8_t> removed_before_best(static_cast<std::size_t>(num_local_nodes), 0);
  for (EdgeIndex idx = 0; idx < best_removed_count; ++idx) {
    const NodeCount local_node = workspace->removed_order[static_cast<std::size_t>(idx)];
    removed_before_best[local_node] = 1;
  }

  for (NodeCount local_node = 0; local_node < num_local_nodes; ++local_node) {
    if (!removed_before_best[local_node]) {
      result.u_max.push_back(
          prepared_graph.local_to_global[static_cast<std::size_t>(local_node)]);
    }
  }
  if (result.u_max.empty() && prepared_graph.fallback_node != kInvalidNode) {
    result.u_max.push_back(prepared_graph.fallback_node);
  }
  result.rho_max = best_density;
  return result;
}

}  // namespace dcgs
