#include "graph.h"

#include "graph_binary.h"

#include <algorithm>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace dcgs {

namespace {

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

// HasBinaryGraphSuffix selects the binary load path based on the filename.
bool HasBinaryGraphSuffix(const std::string& path) {
  static const std::string kSuffix = ".dcgsbin";
  return path.size() >= kSuffix.size() &&
         path.compare(path.size() - kSuffix.size(), kSuffix.size(), kSuffix) == 0;
}

// ActiveRule picks the edge-value rule relevant to the currently selected model.
RuleKind ActiveRule(const RuntimeOptions& opts) {
  return (opts.model == DiffusionModel::kIC) ? opts.ic_rule : opts.lt_rule;
}

// ScanGraphFile is the common multi-pass text reader used during graph loading.
// The callback sees already validated edges, and undirected materialization is
// applied here so later passes operate on one consistent stream.
template <typename Fn>
bool ScanGraphFile(const RuntimeOptions& opts, Fn&& fn, std::string* err) {
  std::ifstream fin(opts.graph_path);
  if (!fin) {
    if (err) *err = "Failed to open graph file: " + opts.graph_path;
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
    if (opts.undirected && edge.from != edge.to) {
      ParsedEdge reverse = edge;
      std::swap(reverse.from, reverse.to);
      if (!fn(reverse)) return false;
    }
  }

  return true;
}

// ComputeBaseEdgeValue applies the user-selected rule before any optional LT
// incoming-normalization step.
bool ComputeBaseEdgeValue(Degree indeg_to,
                          bool has_input_value,
                          Real input_value,
                          const RuntimeOptions& opts,
                          RuleKind active_rule,
                          Real* value,
                          std::string* err) {
  if (!value) {
    if (err) *err = "Internal error: value is null.";
    return false;
  }

  Real base_value = 0.0;
  if (active_rule == RuleKind::kFile) {
    if (!has_input_value) {
      if (err) *err = "Rule file requires third column in graph input.";
      return false;
    }
    base_value = input_value;
  } else if (active_rule == RuleKind::kConst) {
    base_value = opts.p;
  } else if (active_rule == RuleKind::kCOverInDeg) {
    base_value = (indeg_to > 0) ? (opts.c / static_cast<Real>(indeg_to)) : 0.0;
  } else if (active_rule == RuleKind::kUniformIn) {
    base_value = (indeg_to > 0) ? (1.0 / static_cast<Real>(indeg_to)) : 0.0;
  }

  *value = std::clamp(base_value, static_cast<Real>(0.0), static_cast<Real>(1.0));
  return true;
}

// FinalizeEdgeValues converts raw third-column values or topology-derived rules
// into the runtime out_values array used by diffusion simulation.
bool FinalizeEdgeValues(const RuntimeOptions& opts,
                        const std::vector<Real>* raw_values,
                        Graph* graph,
                        std::string* err) {
  if (!graph) {
    if (err) *err = "Internal error: graph is null.";
    return false;
  }
  if (raw_values &&
      raw_values->size() != static_cast<std::size_t>(graph->num_edges)) {
    if (err) *err = "Internal error: raw_values size mismatch.";
    return false;
  }

  const RuleKind active_rule = ActiveRule(opts);
  if (active_rule == RuleKind::kFile && !raw_values) {
    if (err) *err = "Rule file requires third column in graph input.";
    return false;
  }

  graph->out_values.assign(static_cast<std::size_t>(graph->num_edges), 0.0);

  std::vector<Real> in_weight_sum;
  if (opts.model == DiffusionModel::kLT && opts.lt_normalize_in) {
    in_weight_sum.assign(static_cast<std::size_t>(graph->num_nodes), 0.0);
    for (NodeId from = 0; from < graph->num_nodes; ++from) {
      const EdgeOffset begin = graph->out_offsets[static_cast<std::size_t>(from)];
      const EdgeOffset end = graph->out_offsets[static_cast<std::size_t>(from) + 1];
      for (EdgeOffset pos = begin; pos < end; ++pos) {
        const std::size_t index = static_cast<std::size_t>(pos);
        const NodeId to = graph->out_neighbors[index];
        const bool has_input_value = raw_values != nullptr;
        const Real input_value = has_input_value ? (*raw_values)[index] : 0.0;
        Real base_value = 0.0;
        if (!ComputeBaseEdgeValue(graph->indeg[to], has_input_value, input_value,
                                  opts, active_rule, &base_value, err)) {
          return false;
        }
        in_weight_sum[to] += base_value;
      }
    }
  }

  for (NodeId from = 0; from < graph->num_nodes; ++from) {
    const EdgeOffset begin = graph->out_offsets[static_cast<std::size_t>(from)];
    const EdgeOffset end = graph->out_offsets[static_cast<std::size_t>(from) + 1];
    for (EdgeOffset pos = begin; pos < end; ++pos) {
      const std::size_t index = static_cast<std::size_t>(pos);
      const NodeId to = graph->out_neighbors[index];
      const bool has_input_value = raw_values != nullptr;
      const Real input_value = has_input_value ? (*raw_values)[index] : 0.0;
      Real value = 0.0;
      if (!ComputeBaseEdgeValue(graph->indeg[to], has_input_value, input_value,
                                opts, active_rule, &value, err)) {
        return false;
      }
      if (opts.model == DiffusionModel::kLT && opts.lt_normalize_in) {
        const Real denom = in_weight_sum[to];
        value = (denom > 0.0) ? (value / denom) : 0.0;
      }
      graph->out_values[index] = value;
    }
  }

  return true;
}

// LoadGraphFromText performs a few sequential passes over the text file:
// determine graph size, collect degrees, build CSR topology, then derive edge values.
bool LoadGraphFromText(const RuntimeOptions& opts, Graph* graph, std::string* err) {
  NodeId max_node = 0;
  bool saw_any_edge = false;
  bool has_raw_values = false;
  EdgeOffset edge_count = 0;
  if (!ScanGraphFile(
          opts,
          [&](const ParsedEdge& edge) {
            saw_any_edge = true;
            max_node = std::max(max_node, std::max(edge.from, edge.to));
            has_raw_values = edge.has_input_value;
            ++edge_count;
            return true;
          },
          err)) {
    return false;
  }

  graph->num_nodes = saw_any_edge ? static_cast<NodeCount>(max_node + 1) : 0;
  graph->num_edges = edge_count;
  graph->indeg.assign(static_cast<std::size_t>(graph->num_nodes), 0);
  graph->outdeg.assign(static_cast<std::size_t>(graph->num_nodes), 0);

  if (!ScanGraphFile(
          opts,
          [&](const ParsedEdge& edge) {
            ++graph->outdeg[edge.from];
            ++graph->indeg[edge.to];
            return true;
          },
          err)) {
    return false;
  }

  graph->out_offsets.assign(static_cast<std::size_t>(graph->num_nodes) + 1, 0);
  for (NodeId from = 0; from < graph->num_nodes; ++from) {
    graph->out_offsets[static_cast<std::size_t>(from) + 1] =
        graph->out_offsets[static_cast<std::size_t>(from)] +
        static_cast<EdgeOffset>(graph->outdeg[from]);
  }
  graph->num_edges = graph->out_offsets.back();
  graph->out_neighbors.assign(static_cast<std::size_t>(graph->num_edges), 0);

  std::vector<Real> raw_values;
  if (has_raw_values) {
    raw_values.assign(static_cast<std::size_t>(graph->num_edges), 0.0);
  }

  std::vector<EdgeOffset> cursor = graph->out_offsets;
  if (!ScanGraphFile(
          opts,
          [&](const ParsedEdge& edge) {
            const EdgeOffset position = cursor[edge.from]++;
            const std::size_t index = static_cast<std::size_t>(position);
            graph->out_neighbors[index] = edge.to;
            if (has_raw_values) {
              raw_values[index] = edge.input_value;
            }
            return true;
          },
          err)) {
    return false;
  }

  return FinalizeEdgeValues(opts, has_raw_values ? &raw_values : nullptr, graph, err);
}

// LoadGraphFromBinary reuses the preprocessed CSR topology and only recomputes
// the runtime edge values requested by the current CLI options.
bool LoadGraphFromBinary(const RuntimeOptions& opts, Graph* graph, std::string* err) {
  if (opts.undirected) {
    if (err) {
      *err = "Binary graph input does not accept --undirected at runtime; "
             "materialize reverse edges during preprocessing.";
    }
    return false;
  }

  BinaryGraphData binary_graph;
  if (!ReadBinaryGraphFile(opts.graph_path, &binary_graph, err)) {
    return false;
  }

  graph->num_nodes = binary_graph.num_nodes;
  graph->num_edges = binary_graph.num_edges;
  graph->out_offsets = binary_graph.out_offsets;
  graph->out_neighbors = binary_graph.out_neighbors;
  graph->indeg = binary_graph.indeg;
  graph->outdeg = binary_graph.outdeg;

  return FinalizeEdgeValues(
      opts,
      binary_graph.has_raw_values ? &binary_graph.raw_values : nullptr,
      graph,
      err);
}

}  // namespace

bool LoadGraph(const RuntimeOptions& opts, Graph* graph, std::string* err) {
  // The runtime loader keeps text and binary inputs behind one interface so the
  // rest of the pipeline is agnostic to preprocessing choices.
  if (!graph) {
    if (err) *err = "Internal error: graph is null.";
    return false;
  }
  if (HasBinaryGraphSuffix(opts.graph_path)) {
    return LoadGraphFromBinary(opts, graph, err);
  }
  return LoadGraphFromText(opts, graph, err);
}

}  // namespace dcgs
