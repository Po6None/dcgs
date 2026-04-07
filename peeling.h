#ifndef PEELING_H_
#define PEELING_H_

#include "graph.h"
#include "tracked_graph.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace dcgs {

// PeelingResult stores the selected vertex subset and its estimated density.
struct PeelingResult {
  std::vector<NodeId> u_max;
  Real rho_max = 0.0;
};

// PeelingWorkspace holds the mutable state for one weighted-peeling run on a
// tracked graph.
struct PeelingWorkspace {
  std::vector<std::uint8_t> alive;
  std::vector<Real> degree;
  std::vector<NodeCount> removed_order;
  std::vector<std::pair<Real, NodeCount>> heap_entries;
};

// WeightedPeeling runs the weighted min-degree peeling heuristic on the
// canonical tracked graph.
PeelingResult WeightedPeeling(const TrackedGraph& tracked_graph,
                              const std::vector<Real>& weights,
                              PeelingWorkspace* workspace);

}  // namespace dcgs

#endif  // PEELING_H_
