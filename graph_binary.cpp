#include "graph_binary.h"

#include <algorithm>
#include <array>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace dcgs {

namespace {

constexpr std::array<char, 8> kGraphBinaryMagic = {
    'D', 'C', 'G', 'S', 'B', 'I', 'N', '1'};
constexpr std::size_t kHeaderReservedCount = 8;
constexpr std::uint64_t kHeaderSize =
    8 + 4 + 4 + 7 * 8 + kHeaderReservedCount * 8;

struct ParsedEdge {
  NodeId from = kInvalidNode;
  NodeId to = kInvalidNode;
  bool has_input_value = false;
  Real input_value = 0.0;
};

// Tokenize splits one text edge-list line into whitespace-separated fields.
std::vector<std::string> Tokenize(const std::string& line) {
  std::istringstream iss(line);
  std::vector<std::string> tokens;
  std::string token;
  while (iss >> token) {
    tokens.push_back(token);
  }
  return tokens;
}

// ParseUInt64 rejects negative tokens and trailing characters.
bool ParseUInt64(const std::string& s, std::uint64_t* out) {
  if (!out || s.empty() || s[0] == '-') return false;
  try {
    std::size_t idx = 0;
    const std::uint64_t value = std::stoull(s, &idx);
    if (idx != s.size()) return false;
    *out = value;
    return true;
  } catch (...) {
    return false;
  }
}

// ParseNodeIdToken narrows a validated uint64 token into the configured NodeId type.
bool ParseNodeIdToken(const std::string& s, NodeId* out) {
  std::uint64_t value = 0;
  if (!ParseUInt64(s, &value)) return false;
  if (value > static_cast<std::uint64_t>(std::numeric_limits<NodeId>::max())) {
    return false;
  }
  *out = static_cast<NodeId>(value);
  return true;
}

// ParseReal validates one floating-point token without accepting suffix garbage.
bool ParseReal(const std::string& s, Real* out) {
  if (!out) return false;
  try {
    std::size_t idx = 0;
    const Real value = std::stod(s, &idx);
    if (idx != s.size()) return false;
    *out = value;
    return true;
  } catch (...) {
    return false;
  }
}

// IsLittleEndianHost guards the current on-disk format, which is written in the
// native little-endian layout of the development environment.
bool IsLittleEndianHost() {
  const std::uint32_t value = 1;
  return *reinterpret_cast<const std::uint8_t*>(&value) == 1;
}

// ScanEdgeList is the preprocessing counterpart of the runtime text scanner.
// It validates each edge once and optionally materializes reverse edges.
template <typename Fn>
bool ScanEdgeList(const std::string& input_path, bool undirected, Fn&& fn, std::string* err) {
  std::ifstream fin(input_path);
  if (!fin) {
    if (err) *err = "Failed to open input graph file: " + input_path;
    return false;
  }

  std::string line;
  bool third_column_mode_initialized = false;
  bool expect_input_value = false;

  while (std::getline(fin, line)) {
    if (line.empty() || line[0] == '#') continue;
    const std::vector<std::string> tokens = Tokenize(line);
    if (tokens.size() < 2) continue;

    ParsedEdge edge;
    if (!ParseNodeIdToken(tokens[0], &edge.from) ||
        !ParseNodeIdToken(tokens[1], &edge.to)) {
      if (err) *err = "Invalid edge line: " + line;
      return false;
    }

    edge.has_input_value = tokens.size() >= 3;
    if (edge.has_input_value && !ParseReal(tokens[2], &edge.input_value)) {
      if (err) *err = "Invalid third column in edge line: " + line;
      return false;
    }

    if (!third_column_mode_initialized) {
      expect_input_value = edge.has_input_value;
      third_column_mode_initialized = true;
    } else if (expect_input_value != edge.has_input_value) {
      if (err) {
        *err = "Mixed edge formats: all edges must either have the third column "
               "or all omit it.";
      }
      return false;
    }

    if (!fn(edge)) return false;
    if (undirected && edge.from != edge.to) {
      ParsedEdge reverse = edge;
      std::swap(reverse.from, reverse.to);
      if (!fn(reverse)) return false;
    }
  }

  return true;
}

// WriteBytes writes one typed array section into the binary file.
template <typename T>
bool WriteBytes(std::ofstream* fout, const T* data, std::size_t count, std::string* err) {
  if (!fout) {
    if (err) *err = "Internal error: output stream is null.";
    return false;
  }
  if (count == 0) return true;
  fout->write(reinterpret_cast<const char*>(data),
              static_cast<std::streamsize>(sizeof(T) * count));
  if (!*fout) {
    if (err) *err = "Failed to write binary graph file.";
    return false;
  }
  return true;
}

// ReadBytes reads one typed array section from the binary file.
template <typename T>
bool ReadBytes(std::ifstream* fin, T* data, std::size_t count, std::string* err) {
  if (!fin) {
    if (err) *err = "Internal error: input stream is null.";
    return false;
  }
  if (count == 0) return true;
  fin->read(reinterpret_cast<char*>(data),
            static_cast<std::streamsize>(sizeof(T) * count));
  if (!*fin) {
    if (err) *err = "Failed to read binary graph file.";
    return false;
  }
  return true;
}

// ValidateBinaryGraphData checks that the in-memory CSR payload is internally
// consistent before serialization or after deserialization.
bool ValidateBinaryGraphData(const BinaryGraphData& graph, std::string* err) {
  if (graph.out_offsets.size() != static_cast<std::size_t>(graph.num_nodes) + 1) {
    if (err) *err = "Invalid binary graph data: out_offsets size mismatch.";
    return false;
  }
  if (graph.out_neighbors.size() != static_cast<std::size_t>(graph.num_edges)) {
    if (err) *err = "Invalid binary graph data: out_neighbors size mismatch.";
    return false;
  }
  if (graph.indeg.size() != static_cast<std::size_t>(graph.num_nodes)) {
    if (err) *err = "Invalid binary graph data: indeg size mismatch.";
    return false;
  }
  if (graph.outdeg.size() != static_cast<std::size_t>(graph.num_nodes)) {
    if (err) *err = "Invalid binary graph data: outdeg size mismatch.";
    return false;
  }
  if (graph.out_offsets.empty()) {
    if (graph.num_edges != 0) {
      if (err) *err = "Invalid binary graph data: empty offsets with non-zero edges.";
      return false;
    }
  } else if (graph.out_offsets.back() != graph.num_edges) {
    if (err) *err = "Invalid binary graph data: CSR edge count mismatch.";
    return false;
  }
  if (graph.has_raw_values &&
      graph.raw_values.size() != static_cast<std::size_t>(graph.num_edges)) {
    if (err) *err = "Invalid binary graph data: raw_values size mismatch.";
    return false;
  }
  if (!graph.has_raw_values && !graph.raw_values.empty()) {
    if (err) *err = "Invalid binary graph data: raw_values present without flag.";
    return false;
  }
  return true;
}

// ValidateGraphDegreeStats checks that degree-only summaries are internally
// consistent before they are consumed by seed-selection tools.
bool ValidateGraphDegreeStats(const GraphDegreeStats& stats, std::string* err) {
  if (stats.indeg.size() != static_cast<std::size_t>(stats.num_nodes)) {
    if (err) *err = "Invalid degree stats: indeg size mismatch.";
    return false;
  }
  if (stats.outdeg.size() != static_cast<std::size_t>(stats.num_nodes)) {
    if (err) *err = "Invalid degree stats: outdeg size mismatch.";
    return false;
  }
  return true;
}

// SeekTo positions the binary input stream at one section offset from the header.
bool SeekTo(std::ifstream* fin, std::uint64_t offset, std::string* err) {
  if (!fin) {
    if (err) *err = "Internal error: input stream is null.";
    return false;
  }
  fin->clear();
  fin->seekg(static_cast<std::streamoff>(offset), std::ios::beg);
  if (!*fin) {
    if (err) *err = "Failed to seek in binary graph file.";
    return false;
  }
  return true;
}

}  // namespace

bool BuildBinaryGraphFromTextEdgeList(const std::string& input_path,
                                      bool undirected,
                                      BinaryGraphData* graph,
                                      std::string* err) {
  // The converter mirrors the runtime text loader, but stops after producing a
  // reusable topology payload with optional raw third-column values.
  if (!graph) {
    if (err) *err = "Internal error: graph is null.";
    return false;
  }

  graph->num_nodes = 0;
  graph->num_edges = 0;
  graph->has_raw_values = false;
  graph->undirected_materialized = undirected;
  graph->out_offsets.clear();
  graph->out_neighbors.clear();
  graph->indeg.clear();
  graph->outdeg.clear();
  graph->raw_values.clear();

  NodeId max_node = 0;
  bool saw_any_edge = false;
  EdgeOffset edge_count = 0;
  bool has_raw_values = false;
  if (!ScanEdgeList(
          input_path, undirected,
          [&](const ParsedEdge& edge) {
            saw_any_edge = true;
            max_node = std::max(max_node, std::max(edge.from, edge.to));
            ++edge_count;
            has_raw_values = edge.has_input_value;
            return true;
          },
          err)) {
    return false;
  }

  graph->num_nodes = saw_any_edge ? static_cast<NodeCount>(max_node + 1) : 0;
  graph->num_edges = edge_count;
  graph->has_raw_values = has_raw_values;
  graph->indeg.assign(static_cast<std::size_t>(graph->num_nodes), 0);
  graph->outdeg.assign(static_cast<std::size_t>(graph->num_nodes), 0);

  if (!ScanEdgeList(
          input_path, undirected,
          [&](const ParsedEdge& edge) {
            ++graph->outdeg[edge.from];
            ++graph->indeg[edge.to];
            return true;
          },
          err)) {
    return false;
  }

  graph->out_offsets.assign(static_cast<std::size_t>(graph->num_nodes) + 1, 0);
  for (NodeId node = 0; node < graph->num_nodes; ++node) {
    graph->out_offsets[static_cast<std::size_t>(node) + 1] =
        graph->out_offsets[static_cast<std::size_t>(node)] +
        static_cast<EdgeOffset>(graph->outdeg[node]);
  }
  graph->num_edges = graph->out_offsets.back();
  graph->out_neighbors.assign(static_cast<std::size_t>(graph->num_edges), 0);
  if (graph->has_raw_values) {
    graph->raw_values.assign(static_cast<std::size_t>(graph->num_edges), 0.0);
  } else {
    graph->raw_values.clear();
  }

  std::vector<EdgeOffset> cursor = graph->out_offsets;
  if (!ScanEdgeList(
          input_path, undirected,
          [&](const ParsedEdge& edge) {
            const EdgeOffset position = cursor[edge.from]++;
            graph->out_neighbors[static_cast<std::size_t>(position)] = edge.to;
            if (graph->has_raw_values) {
              graph->raw_values[static_cast<std::size_t>(position)] = edge.input_value;
            }
            return true;
          },
          err)) {
    return false;
  }

  return ValidateBinaryGraphData(*graph, err);
}

bool BuildGraphDegreeStatsFromTextEdgeList(const std::string& input_path,
                                           bool undirected,
                                           GraphDegreeStats* stats,
                                           std::string* err) {
  // Degree-only preprocessing avoids allocating adjacency arrays when a tool
  // only needs ranking statistics such as in-degree or out-degree.
  if (!stats) {
    if (err) *err = "Internal error: stats is null.";
    return false;
  }

  stats->num_nodes = 0;
  stats->num_edges = 0;
  stats->undirected_materialized = undirected;
  stats->indeg.clear();
  stats->outdeg.clear();

  if (!ScanEdgeList(
          input_path, undirected,
          [&](const ParsedEdge& edge) {
            const NodeId max_endpoint = std::max(edge.from, edge.to);
            const std::size_t required_size =
                static_cast<std::size_t>(max_endpoint) + 1;
            if (stats->indeg.size() < required_size) {
              stats->indeg.resize(required_size, 0);
              stats->outdeg.resize(required_size, 0);
            }
            ++stats->outdeg[edge.from];
            ++stats->indeg[edge.to];
            ++stats->num_edges;
            return true;
          },
          err)) {
    return false;
  }

  stats->num_nodes = static_cast<NodeCount>(stats->indeg.size());
  return ValidateGraphDegreeStats(*stats, err);
}

bool WriteBinaryGraphFile(const std::string& output_path,
                          const BinaryGraphData& graph,
                          std::string* err) {
  // The binary layout is intentionally simple: one fixed-size header followed
  // by contiguous typed array sections for CSR topology and raw values.
  if (!IsLittleEndianHost()) {
    if (err) *err = "Binary graph writing currently requires a little-endian host.";
    return false;
  }
  if (!ValidateBinaryGraphData(graph, err)) return false;

  std::ofstream fout(output_path, std::ios::binary);
  if (!fout) {
    if (err) *err = "Failed to open output binary graph file: " + output_path;
    return false;
  }

  const std::uint32_t flags =
      (graph.has_raw_values ? kGraphBinaryFlagHasRawValues : 0u) |
      (graph.undirected_materialized ? kGraphBinaryFlagUndirectedMaterialized : 0u);

  const std::uint64_t node_count = graph.num_nodes;
  const std::uint64_t edge_count = graph.num_edges;
  const std::uint64_t out_offsets_offset = kHeaderSize;
  const std::uint64_t out_neighbors_offset =
      out_offsets_offset +
      sizeof(EdgeOffset) * static_cast<std::uint64_t>(graph.out_offsets.size());
  const std::uint64_t indeg_offset =
      out_neighbors_offset +
      sizeof(NodeId) * static_cast<std::uint64_t>(graph.out_neighbors.size());
  const std::uint64_t outdeg_offset =
      indeg_offset + sizeof(Degree) * static_cast<std::uint64_t>(graph.indeg.size());
  const std::uint64_t raw_values_offset =
      graph.has_raw_values
          ? (outdeg_offset + sizeof(Degree) *
                               static_cast<std::uint64_t>(graph.outdeg.size()))
          : 0ull;
  const std::uint64_t reserved[kHeaderReservedCount] = {};

  if (!WriteBytes(&fout, kGraphBinaryMagic.data(), kGraphBinaryMagic.size(), err) ||
      !WriteBytes(&fout, &kGraphBinaryVersion, 1, err) ||
      !WriteBytes(&fout, &flags, 1, err) ||
      !WriteBytes(&fout, &node_count, 1, err) ||
      !WriteBytes(&fout, &edge_count, 1, err) ||
      !WriteBytes(&fout, &out_offsets_offset, 1, err) ||
      !WriteBytes(&fout, &out_neighbors_offset, 1, err) ||
      !WriteBytes(&fout, &indeg_offset, 1, err) ||
      !WriteBytes(&fout, &outdeg_offset, 1, err) ||
      !WriteBytes(&fout, &raw_values_offset, 1, err) ||
      !WriteBytes(&fout, reserved, kHeaderReservedCount, err) ||
      !WriteBytes(&fout, graph.out_offsets.data(), graph.out_offsets.size(), err) ||
      !WriteBytes(&fout, graph.out_neighbors.data(), graph.out_neighbors.size(), err) ||
      !WriteBytes(&fout, graph.indeg.data(), graph.indeg.size(), err) ||
      !WriteBytes(&fout, graph.outdeg.data(), graph.outdeg.size(), err)) {
    return false;
  }

  if (graph.has_raw_values &&
      !WriteBytes(&fout, graph.raw_values.data(), graph.raw_values.size(), err)) {
    return false;
  }

  return true;
}

bool ReadBinaryGraphFile(const std::string& input_path,
                         BinaryGraphData* graph,
                         std::string* err) {
  // Reading reconstructs BinaryGraphData exactly as preprocessing wrote it, so
  // the runtime graph loader can apply the current IC/LT rule afterwards.
  if (!graph) {
    if (err) *err = "Internal error: graph is null.";
    return false;
  }
  if (!IsLittleEndianHost()) {
    if (err) *err = "Binary graph reading currently requires a little-endian host.";
    return false;
  }

  std::ifstream fin(input_path, std::ios::binary);
  if (!fin) {
    if (err) *err = "Failed to open input binary graph file: " + input_path;
    return false;
  }

  fin.seekg(0, std::ios::end);
  const std::uint64_t file_size =
      static_cast<std::uint64_t>(fin.tellg());
  fin.seekg(0, std::ios::beg);
  if (file_size < kHeaderSize) {
    if (err) *err = "Binary graph file is too small to contain a valid header.";
    return false;
  }

  std::array<char, 8> magic{};
  std::uint32_t version = 0;
  std::uint32_t flags = 0;
  std::uint64_t node_count_u64 = 0;
  std::uint64_t edge_count = 0;
  std::uint64_t out_offsets_offset = 0;
  std::uint64_t out_neighbors_offset = 0;
  std::uint64_t indeg_offset = 0;
  std::uint64_t outdeg_offset = 0;
  std::uint64_t raw_values_offset = 0;
  std::uint64_t reserved[kHeaderReservedCount] = {};

  if (!ReadBytes(&fin, magic.data(), magic.size(), err) ||
      !ReadBytes(&fin, &version, 1, err) ||
      !ReadBytes(&fin, &flags, 1, err) ||
      !ReadBytes(&fin, &node_count_u64, 1, err) ||
      !ReadBytes(&fin, &edge_count, 1, err) ||
      !ReadBytes(&fin, &out_offsets_offset, 1, err) ||
      !ReadBytes(&fin, &out_neighbors_offset, 1, err) ||
      !ReadBytes(&fin, &indeg_offset, 1, err) ||
      !ReadBytes(&fin, &outdeg_offset, 1, err) ||
      !ReadBytes(&fin, &raw_values_offset, 1, err) ||
      !ReadBytes(&fin, reserved, kHeaderReservedCount, err)) {
    return false;
  }

  if (magic != kGraphBinaryMagic) {
    if (err) *err = "Invalid binary graph magic.";
    return false;
  }
  if (version != kGraphBinaryVersion) {
    if (err) *err = "Unsupported binary graph version: " + std::to_string(version);
    return false;
  }
  if (node_count_u64 >
      static_cast<std::uint64_t>(std::numeric_limits<NodeCount>::max())) {
    if (err) *err = "Binary graph node_count exceeds supported NodeCount range.";
    return false;
  }

  graph->num_nodes = static_cast<NodeCount>(node_count_u64);
  graph->num_edges = edge_count;
  graph->has_raw_values = (flags & kGraphBinaryFlagHasRawValues) != 0;
  graph->undirected_materialized =
      (flags & kGraphBinaryFlagUndirectedMaterialized) != 0;

  const std::uint64_t expected_out_offsets_size =
      sizeof(EdgeOffset) * (static_cast<std::uint64_t>(graph->num_nodes) + 1);
  const std::uint64_t expected_out_neighbors_size =
      sizeof(NodeId) * static_cast<std::uint64_t>(graph->num_edges);
  const std::uint64_t expected_indeg_size =
      sizeof(Degree) * static_cast<std::uint64_t>(graph->num_nodes);
  const std::uint64_t expected_outdeg_size =
      sizeof(Degree) * static_cast<std::uint64_t>(graph->num_nodes);
  const std::uint64_t expected_raw_values_size =
      sizeof(Real) * static_cast<std::uint64_t>(graph->num_edges);

  if (out_offsets_offset < kHeaderSize ||
      out_neighbors_offset < out_offsets_offset + expected_out_offsets_size ||
      indeg_offset < out_neighbors_offset + expected_out_neighbors_size ||
      outdeg_offset < indeg_offset + expected_indeg_size) {
    if (err) *err = "Invalid section offsets in binary graph file.";
    return false;
  }

  if (graph->has_raw_values) {
    if (raw_values_offset < outdeg_offset + expected_outdeg_size ||
        raw_values_offset + expected_raw_values_size > file_size) {
      if (err) *err = "Invalid raw_values section in binary graph file.";
      return false;
    }
  } else if (raw_values_offset != 0) {
    if (err) *err = "raw_values_offset must be zero when raw values are absent.";
    return false;
  }

  graph->out_offsets.assign(static_cast<std::size_t>(graph->num_nodes) + 1, 0);
  graph->out_neighbors.assign(static_cast<std::size_t>(graph->num_edges), 0);
  graph->indeg.assign(static_cast<std::size_t>(graph->num_nodes), 0);
  graph->outdeg.assign(static_cast<std::size_t>(graph->num_nodes), 0);
  graph->raw_values.assign(graph->has_raw_values
                               ? static_cast<std::size_t>(graph->num_edges)
                               : 0,
                           0.0);

  if (!SeekTo(&fin, out_offsets_offset, err) ||
      !ReadBytes(&fin, graph->out_offsets.data(), graph->out_offsets.size(), err) ||
      !SeekTo(&fin, out_neighbors_offset, err) ||
      !ReadBytes(&fin, graph->out_neighbors.data(), graph->out_neighbors.size(), err) ||
      !SeekTo(&fin, indeg_offset, err) ||
      !ReadBytes(&fin, graph->indeg.data(), graph->indeg.size(), err) ||
      !SeekTo(&fin, outdeg_offset, err) ||
      !ReadBytes(&fin, graph->outdeg.data(), graph->outdeg.size(), err)) {
    return false;
  }

  if (graph->has_raw_values &&
      (!SeekTo(&fin, raw_values_offset, err) ||
       !ReadBytes(&fin, graph->raw_values.data(), graph->raw_values.size(), err))) {
    return false;
  }

  return ValidateBinaryGraphData(*graph, err);
}

bool ReadBinaryGraphDegreeStats(const std::string& input_path,
                                GraphDegreeStats* stats,
                                std::string* err) {
  // Degree-only binary loading reads just the sections needed for seed ranking
  // and skips the much larger CSR neighbor arrays.
  if (!stats) {
    if (err) *err = "Internal error: stats is null.";
    return false;
  }
  if (!IsLittleEndianHost()) {
    if (err) *err = "Binary graph reading currently requires a little-endian host.";
    return false;
  }

  std::ifstream fin(input_path, std::ios::binary);
  if (!fin) {
    if (err) *err = "Failed to open input binary graph file: " + input_path;
    return false;
  }

  fin.seekg(0, std::ios::end);
  const std::uint64_t file_size = static_cast<std::uint64_t>(fin.tellg());
  fin.seekg(0, std::ios::beg);
  if (file_size < kHeaderSize) {
    if (err) *err = "Binary graph file is too small to contain a valid header.";
    return false;
  }

  std::array<char, 8> magic{};
  std::uint32_t version = 0;
  std::uint32_t flags = 0;
  std::uint64_t node_count_u64 = 0;
  std::uint64_t edge_count = 0;
  std::uint64_t out_offsets_offset = 0;
  std::uint64_t out_neighbors_offset = 0;
  std::uint64_t indeg_offset = 0;
  std::uint64_t outdeg_offset = 0;
  std::uint64_t raw_values_offset = 0;
  std::uint64_t reserved[kHeaderReservedCount] = {};

  if (!ReadBytes(&fin, magic.data(), magic.size(), err) ||
      !ReadBytes(&fin, &version, 1, err) ||
      !ReadBytes(&fin, &flags, 1, err) ||
      !ReadBytes(&fin, &node_count_u64, 1, err) ||
      !ReadBytes(&fin, &edge_count, 1, err) ||
      !ReadBytes(&fin, &out_offsets_offset, 1, err) ||
      !ReadBytes(&fin, &out_neighbors_offset, 1, err) ||
      !ReadBytes(&fin, &indeg_offset, 1, err) ||
      !ReadBytes(&fin, &outdeg_offset, 1, err) ||
      !ReadBytes(&fin, &raw_values_offset, 1, err) ||
      !ReadBytes(&fin, reserved, kHeaderReservedCount, err)) {
    return false;
  }

  if (magic != kGraphBinaryMagic) {
    if (err) *err = "Invalid binary graph magic.";
    return false;
  }
  if (version != kGraphBinaryVersion) {
    if (err) *err = "Unsupported binary graph version: " + std::to_string(version);
    return false;
  }
  if (node_count_u64 >
      static_cast<std::uint64_t>(std::numeric_limits<NodeCount>::max())) {
    if (err) *err = "Binary graph node_count exceeds supported NodeCount range.";
    return false;
  }

  stats->num_nodes = static_cast<NodeCount>(node_count_u64);
  stats->num_edges = edge_count;
  stats->undirected_materialized =
      (flags & kGraphBinaryFlagUndirectedMaterialized) != 0;

  const std::uint64_t expected_out_offsets_size =
      sizeof(EdgeOffset) * (static_cast<std::uint64_t>(stats->num_nodes) + 1);
  const std::uint64_t expected_out_neighbors_size =
      sizeof(NodeId) * static_cast<std::uint64_t>(stats->num_edges);
  const std::uint64_t expected_indeg_size =
      sizeof(Degree) * static_cast<std::uint64_t>(stats->num_nodes);
  const std::uint64_t expected_outdeg_size =
      sizeof(Degree) * static_cast<std::uint64_t>(stats->num_nodes);

  if (out_offsets_offset < kHeaderSize ||
      out_neighbors_offset < out_offsets_offset + expected_out_offsets_size ||
      indeg_offset < out_neighbors_offset + expected_out_neighbors_size ||
      outdeg_offset < indeg_offset + expected_indeg_size ||
      outdeg_offset + expected_outdeg_size > file_size) {
    if (err) *err = "Invalid section offsets in binary graph file.";
    return false;
  }

  stats->indeg.assign(static_cast<std::size_t>(stats->num_nodes), 0);
  stats->outdeg.assign(static_cast<std::size_t>(stats->num_nodes), 0);
  if (!SeekTo(&fin, indeg_offset, err) ||
      !ReadBytes(&fin, stats->indeg.data(), stats->indeg.size(), err) ||
      !SeekTo(&fin, outdeg_offset, err) ||
      !ReadBytes(&fin, stats->outdeg.data(), stats->outdeg.size(), err)) {
    return false;
  }

  return ValidateGraphDegreeStats(*stats, err);
}

}  // namespace dcgs
