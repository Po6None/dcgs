#ifndef GRAPH_H_
#define GRAPH_H_

#include "common.h"

#include <vector>

namespace dcgs {

// Graph stores the runtime CSR representation used by all simulators and
// solvers. out_values contains the active IC probability or LT weight per edge.
struct Graph {
  NodeCount num_nodes = 0;
  EdgeOffset num_edges = 0;
  std::vector<EdgeOffset> out_offsets;
  std::vector<NodeId> out_neighbors;
  std::vector<Real> out_values;
  std::vector<Degree> indeg;
  std::vector<Degree> outdeg;
};

// LoadGraph reads either a text edge list or a preprocessed binary CSR graph
// and materializes the runtime edge values specified by RuntimeOptions.
bool LoadGraph(const RuntimeOptions& opts, Graph* graph, std::string* err);

}  // namespace dcgs

#endif  // GRAPH_H_
