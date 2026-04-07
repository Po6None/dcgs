#include "tracked_graph.h"

#include <algorithm>
#include <utility>
#include <vector>

namespace dcgs {

namespace {

EdgePair UnpackPairKey(std::uint64_t key) {
  return {static_cast<NodeId>(key >> 32),
          static_cast<NodeId>(key & 0xffffffffu)};
}

// This builder performs all one-time cleanup needed by downstream solvers:
// it removes invalid/self/seed incidents, compresses surviving endpoints to a
// dense local id space, and constructs CSR adjacency aligned with canonical
// tracked edge indices.
TrackedGraph BuildTrackedGraphFromEdges(const Graph& graph,
                                        const std::vector<NodeId>& seeds,
                                        const std::vector<EdgePair>& edges) {
  TrackedGraph tracked_graph;

  std::vector<std::uint8_t> is_seed(static_cast<std::size_t>(graph.num_nodes), 0);
  for (NodeId seed : seeds) {
    if (seed < graph.num_nodes) is_seed[seed] = 1;
  }

  for (NodeId node = 0; node < graph.num_nodes; ++node) {
    if (!is_seed[node]) {
      tracked_graph.fallback_node = node;
      break;
    }
  }

  // Collect every endpoint first so that local ids are stable and compact.
  // The later edge pass can then use lower_bound over this sorted array without
  // allocating a separate hash map.
  std::vector<NodeId> local_to_global;
  local_to_global.reserve(edges.size() * 2);
  for (const EdgePair& edge : edges) {
    if (edge.u >= graph.num_nodes || edge.v >= graph.num_nodes) continue;
    if (edge.u == edge.v) continue;
    if (is_seed[edge.u] || is_seed[edge.v]) continue;
    local_to_global.push_back(edge.u);
    local_to_global.push_back(edge.v);
  }

  if (local_to_global.empty()) return tracked_graph;

  std::sort(local_to_global.begin(), local_to_global.end());
  local_to_global.erase(std::unique(local_to_global.begin(), local_to_global.end()),
                        local_to_global.end());
  tracked_graph.local_to_global = std::move(local_to_global);

  const NodeCount num_local_nodes =
      static_cast<NodeCount>(tracked_graph.local_to_global.size());
  std::vector<EdgeOffset> local_degree(static_cast<std::size_t>(num_local_nodes), 0);

  tracked_graph.edge_u.reserve(edges.size());
  tracked_graph.edge_v.reserve(edges.size());

  auto get_local = [&](NodeId global_node) -> NodeCount {
    const auto it = std::lower_bound(tracked_graph.local_to_global.begin(),
                                     tracked_graph.local_to_global.end(),
                                     global_node);
    return static_cast<NodeCount>(it - tracked_graph.local_to_global.begin());
  };

  // edge_u / edge_v define the single authoritative edge indexing used by
  // counters, checkpoint weights, peeling, and exact min-cut.
  for (const EdgePair& edge : edges) {
    if (edge.u >= graph.num_nodes || edge.v >= graph.num_nodes) continue;
    if (edge.u == edge.v) continue;
    if (is_seed[edge.u] || is_seed[edge.v]) continue;

    const NodeCount local_u = get_local(edge.u);
    const NodeCount local_v = get_local(edge.v);
    if (local_u == local_v) continue;

    tracked_graph.edge_u.push_back(local_u);
    tracked_graph.edge_v.push_back(local_v);
    ++local_degree[local_u];
    ++local_degree[local_v];
  }

  tracked_graph.offsets.assign(static_cast<std::size_t>(num_local_nodes) + 1, 0);
  for (NodeCount local_node = 0; local_node < num_local_nodes; ++local_node) {
    tracked_graph.offsets[static_cast<std::size_t>(local_node) + 1] =
        tracked_graph.offsets[static_cast<std::size_t>(local_node)] +
        local_degree[local_node];
  }

  const std::size_t adjacency_size =
      static_cast<std::size_t>(tracked_graph.offsets.back());
  tracked_graph.neighbors.assign(adjacency_size, 0);
  tracked_graph.incident_positions.assign(adjacency_size, 0);

  // Each undirected tracked edge contributes two adjacency entries. The cursor
  // array is a standard CSR construction trick that avoids per-node push_back.
  std::vector<EdgeOffset> cursor = tracked_graph.offsets;
  for (EdgeIndex edge_index = 0;
       edge_index < static_cast<EdgeIndex>(tracked_graph.edge_u.size());
       ++edge_index) {
    const NodeCount local_u =
        tracked_graph.edge_u[static_cast<std::size_t>(edge_index)];
    const NodeCount local_v =
        tracked_graph.edge_v[static_cast<std::size_t>(edge_index)];

    const EdgeOffset pos_u = cursor[local_u]++;
    tracked_graph.neighbors[static_cast<std::size_t>(pos_u)] = local_v;
    tracked_graph.incident_positions[static_cast<std::size_t>(pos_u)] = edge_index;

    const EdgeOffset pos_v = cursor[local_v]++;
    tracked_graph.neighbors[static_cast<std::size_t>(pos_v)] = local_u;
    tracked_graph.incident_positions[static_cast<std::size_t>(pos_v)] = edge_index;
  }

  return tracked_graph;
}

}  // namespace

TrackedGraph BuildTrackedGraph(const Graph& graph,
                               const std::vector<NodeId>& seeds,
                               const std::vector<EdgePair>& edges) {
  return BuildTrackedGraphFromEdges(graph, seeds, edges);
}

TrackedGraph BuildTrackedGraphFromSortedPairKeys(
    const Graph& graph,
    const std::vector<NodeId>& seeds,
    const std::vector<std::uint64_t>& sorted_unique_pair_keys) {
  // The pilot stage already paid for global deduplication, so this conversion
  // is intentionally simple: decode once and immediately build the canonical
  // tracked graph.
  std::vector<EdgePair> edges;
  edges.reserve(sorted_unique_pair_keys.size());
  for (std::uint64_t key : sorted_unique_pair_keys) {
    edges.push_back(UnpackPairKey(key));
  }
  return BuildTrackedGraphFromEdges(graph, seeds, edges);
}

}  // namespace dcgs
