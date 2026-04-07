#ifndef COMMON_H_
#define COMMON_H_

#include <cstdint>
#include <limits>
#include <string>

namespace dcgs {

// The project uses fixed-width integral types so large-graph experiments remain
// predictable across platforms and intermediate counters do not silently narrow.
using NodeId = std::uint32_t;
using NodeCount = std::uint32_t;
using Degree = std::uint32_t;
using EdgeIndex = std::uint64_t;
using EdgeOffset = std::uint64_t;
using SampleCount = std::uint64_t;
using Real = double;

constexpr NodeId kInvalidNode = std::numeric_limits<NodeId>::max();

// DiffusionModel selects which propagation process is simulated on the input
// graph for pilot, certified, baselines, and evaluation.
enum class DiffusionModel {
  kIC,
  kLT,
};

// AlgorithmKind selects the top-level experimental pipeline to execute.
enum class AlgorithmKind {
  kDcgs,
  kBestPair,
  kTopKActivationProb,
  kTopKCoactivationDegree,
  kGreedyAdd,
  kLocalSearch,
  kStructurePeeling,
  kEdgeCoactPeeling,
  kCertifiedCapNoEarlyStop,
};

// AlgorithmName converts the public enum to the CLI/logging name.
inline const char* AlgorithmName(AlgorithmKind algorithm) {
  switch (algorithm) {
    case AlgorithmKind::kDcgs:
      return "dcgs";
    case AlgorithmKind::kBestPair:
      return "best_pair";
    case AlgorithmKind::kTopKActivationProb:
      return "topk_activation_prob";
    case AlgorithmKind::kTopKCoactivationDegree:
      return "topk_coactivation_degree";
    case AlgorithmKind::kGreedyAdd:
      return "greedy_add";
    case AlgorithmKind::kLocalSearch:
      return "local_search";
    case AlgorithmKind::kStructurePeeling:
      return "structure_peeling";
    case AlgorithmKind::kEdgeCoactPeeling:
      return "edge_coact_peeling";
    case AlgorithmKind::kCertifiedCapNoEarlyStop:
      return "certified_cap_no_early_stop";
  }
  return "unknown";
}

// SubproblemSolverKind controls how the densest-subgraph subproblem is solved
// inside certified-style pipelines.
enum class SubproblemSolverKind {
  kPeeling,
  kExact,
};

// SubproblemSolverName converts the enum to the CLI/logging name.
inline const char* SubproblemSolverName(SubproblemSolverKind solver) {
  switch (solver) {
    case SubproblemSolverKind::kPeeling:
      return "peeling";
    case SubproblemSolverKind::kExact:
      return "exact";
  }
  return "unknown";
}

// RuleKind describes how edge probabilities or LT weights are derived from the
// raw graph topology and optional third column.
enum class RuleKind {
  kConst,
  kCOverInDeg,
  kUniformIn,
  kFile,
};

// RuntimeOptions gathers every CLI-controlled parameter that affects input
// loading, diffusion, solver choice, sampling budgets, and output.
struct RuntimeOptions {
  std::string graph_path;
  std::string seeds_path;
  std::string out_path = "result/output_u_hat.txt";
  std::string log_path = "result/output_log.txt";
  std::string trace_csv_path = "result/output_trace.csv";
  std::string trace_json_path = "result/output_trace.json";

  bool undirected = false;
  AlgorithmKind algorithm = AlgorithmKind::kDcgs;
  SubproblemSolverKind subproblem_solver = SubproblemSolverKind::kPeeling;
  DiffusionModel model = DiffusionModel::kIC;

  RuleKind ic_rule = RuleKind::kConst;
  RuleKind lt_rule = RuleKind::kUniformIn;

  Real p = 0.01;
  Real c = 1.0;
  bool lt_normalize_in = false;

  Real tau = 0.05;
  Real eps = 0.1;
  Real delta = 0.1;
  Real gamma = 1e-4;

  std::uint64_t rng_seed = 1;
  SampleCount sample_cap = 0;
  SampleCount eval_cap = 0;
  std::uint32_t local_search_max_iters = 1000;
  std::uint32_t local_search_swap_candidates = 32;
  std::uint64_t pilot_chunk_bytes = 256ull * 1024ull * 1024ull;
  std::uint64_t baseline_chunk_bytes = 256ull * 1024ull * 1024ull;
  std::uint32_t trials = 1;
  bool progress = false;
};

}  // namespace dcgs

#endif  // COMMON_H_
