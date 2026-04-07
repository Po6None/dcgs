#ifndef TRACKED_GRAPH_H_
#define TRACKED_GRAPH_H_

#include "common.h"
#include "graph.h"

#include <cstdint>
#include <vector>

namespace dcgs {

struct EdgePair {
  NodeId u = kInvalidNode;
  NodeId v = kInvalidNode;
};

// TrackedGraph is the canonical in-memory representation of E_hat_tau.
// The structure stores a compact local reindexing of the non-seed endpoints,
// plus CSR-style adjacency over the tracked undirected pair set. All downstream
// modules share this single representation instead of rebuilding their own
// copies from a raw EdgePair list.
struct TrackedGraph {
  // fallback_node is returned by exact/heuristic solvers when the tracked graph
  // is empty and the caller still expects a valid non-seed node candidate.
  NodeId fallback_node = kInvalidNode;

  // local_to_global is sorted and contains exactly the non-seed vertices that
  // appear as endpoints of tracked pairs.
  std::vector<NodeId> local_to_global;

  // offsets / neighbors / incident_positions form a CSR adjacency over local
  // node ids. incident_positions maps each adjacency entry back to the unique
  // tracked edge index in edge_u / edge_v.
  std::vector<EdgeOffset> offsets;
  std::vector<NodeCount> neighbors;
  std::vector<EdgeIndex> incident_positions;

  // edge_u / edge_v store each tracked undirected pair exactly once using local
  // node ids. All per-edge arrays in certified / peeling / exact align to this
  // canonical edge indexing.
  std::vector<NodeCount> edge_u;
  std::vector<NodeCount> edge_v;
};

TrackedGraph BuildTrackedGraph(const Graph& graph,
                               const std::vector<NodeId>& seeds,
                               const std::vector<EdgePair>& edges);

// sorted_unique_pair_keys must already be canonicalized with u <= v and globally
// deduplicated. This builder is used by the pilot stage to materialize the final
// tracked graph without creating an extra long-lived EdgePair vector.
TrackedGraph BuildTrackedGraphFromSortedPairKeys(
    const Graph& graph,
    const std::vector<NodeId>& seeds,
    const std::vector<std::uint64_t>& sorted_unique_pair_keys);

inline EdgeIndex TrackedEdgeCount(const TrackedGraph& graph) {
  return static_cast<EdgeIndex>(graph.edge_u.size());
}

inline NodeCount TrackedNodeCount(const TrackedGraph& graph) {
  return static_cast<NodeCount>(graph.local_to_global.size());
}

}  // namespace dcgs

#endif  // TRACKED_GRAPH_H_
