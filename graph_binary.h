#ifndef GRAPH_BINARY_H_
#define GRAPH_BINARY_H_

#include "common.h"

#include <string>
#include <vector>

namespace dcgs {

// Binary graph files carry a versioned header so runtime loading can reject
// incompatible or incomplete preprocessed inputs.
constexpr std::uint32_t kGraphBinaryVersion = 1;
constexpr std::uint32_t kGraphBinaryFlagHasRawValues = 1u << 0;
constexpr std::uint32_t kGraphBinaryFlagUndirectedMaterialized = 1u << 1;

// BinaryGraphData stores the preprocessed CSR topology and optional raw third
// column exactly as written to and read from .dcgsbin files.
struct BinaryGraphData {
  NodeCount num_nodes = 0;
  EdgeOffset num_edges = 0;
  bool has_raw_values = false;
  bool undirected_materialized = false;

  std::vector<EdgeOffset> out_offsets;
  std::vector<NodeId> out_neighbors;
  std::vector<Degree> indeg;
  std::vector<Degree> outdeg;
  std::vector<Real> raw_values;
};

// GraphDegreeStats is a lightweight topology summary used by preprocessing
// tools that only need node counts and degree sequences.
struct GraphDegreeStats {
  NodeCount num_nodes = 0;
  EdgeOffset num_edges = 0;
  bool undirected_materialized = false;

  std::vector<Degree> indeg;
  std::vector<Degree> outdeg;
};

// BuildBinaryGraphFromTextEdgeList converts a SNAP-style text edge list into
// the binary CSR payload used by WriteBinaryGraphFile.
bool BuildBinaryGraphFromTextEdgeList(const std::string& input_path,
                                      bool undirected,
                                      BinaryGraphData* graph,
                                      std::string* err);

// BuildGraphDegreeStatsFromTextEdgeList scans a text edge list without
// materializing adjacency arrays and keeps only the degree sequences.
bool BuildGraphDegreeStatsFromTextEdgeList(const std::string& input_path,
                                           bool undirected,
                                           GraphDegreeStats* stats,
                                           std::string* err);

// WriteBinaryGraphFile serializes BinaryGraphData into the project-specific
// .dcgsbin format.
bool WriteBinaryGraphFile(const std::string& output_path,
                          const BinaryGraphData& graph,
                          std::string* err);

// ReadBinaryGraphFile deserializes a .dcgsbin file back into BinaryGraphData.
bool ReadBinaryGraphFile(const std::string& input_path,
                         BinaryGraphData* graph,
                         std::string* err);

// ReadBinaryGraphDegreeStats reads only node/edge counts and degree arrays from
// a .dcgsbin file so tools can avoid loading the full CSR payload.
bool ReadBinaryGraphDegreeStats(const std::string& input_path,
                                GraphDegreeStats* stats,
                                std::string* err);

}  // namespace dcgs

#endif  // GRAPH_BINARY_H_
