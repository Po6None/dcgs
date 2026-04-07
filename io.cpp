#include "io.h"

#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace dcgs {

namespace {

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

// ParseDouble validates one floating-point CLI token.
bool ParseDouble(const std::string& s, Real* out) {
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

// ParseNodeIdToken validates one seed/node identifier token.
bool ParseNodeIdToken(const std::string& s, NodeId* out) {
  std::uint64_t value = 0;
  if (!ParseUInt64(s, &value)) return false;
  if (value > static_cast<std::uint64_t>(std::numeric_limits<NodeId>::max())) return false;
  *out = static_cast<NodeId>(value);
  return true;
}

// ParseRule converts the public CLI spelling to the internal rule enum.
RuleKind ParseRule(const std::string& value, bool for_lt, bool* ok) {
  if (ok) *ok = true;
  if (value == "const") return RuleKind::kConst;
  if (value == "c_over_indeg") return RuleKind::kCOverInDeg;
  if (value == "uniform_in" && for_lt) return RuleKind::kUniformIn;
  if (value == "file") return RuleKind::kFile;
  if (ok) *ok = false;
  return RuleKind::kConst;
}

// ParseAlgorithm converts the public CLI spelling to the internal algorithm enum.
AlgorithmKind ParseAlgorithm(const std::string& value, bool* ok) {
  if (ok) *ok = true;
  if (value == "dcgs") return AlgorithmKind::kDcgs;
  if (value == "best_pair") return AlgorithmKind::kBestPair;
  if (value == "topk_activation_prob") return AlgorithmKind::kTopKActivationProb;
  if (value == "topk_coactivation_degree") return AlgorithmKind::kTopKCoactivationDegree;
  if (value == "greedy_add") return AlgorithmKind::kGreedyAdd;
  if (value == "local_search") return AlgorithmKind::kLocalSearch;
  if (value == "structure_peeling") return AlgorithmKind::kStructurePeeling;
  if (value == "edge_coact_peeling") return AlgorithmKind::kEdgeCoactPeeling;
  if (value == "certified_cap_no_early_stop") {
    return AlgorithmKind::kCertifiedCapNoEarlyStop;
  }
  if (ok) *ok = false;
  return AlgorithmKind::kDcgs;
}

// ParseSubproblemSolver converts the public CLI spelling to the internal solver enum.
SubproblemSolverKind ParseSubproblemSolver(const std::string& value, bool* ok) {
  if (ok) *ok = true;
  if (value == "peeling") return SubproblemSolverKind::kPeeling;
  if (value == "exact") return SubproblemSolverKind::kExact;
  if (ok) *ok = false;
  return SubproblemSolverKind::kPeeling;
}

}  // namespace

bool ParseCommandLine(int argc, char** argv, RuntimeOptions* opts, std::string* err) {
  // Parsing is intentionally mechanical here. Cross-flag semantic checks are
  // deferred to ValidateOptions so error messages stay localized.
  if (!opts) {
    if (err) *err = "Internal error: opts is null.";
    return false;
  }

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto need_value = [&](const std::string& flag) -> const char* {
      if (i + 1 >= argc) {
        if (err) *err = "Missing value after " + flag;
        return nullptr;
      }
      ++i;
      return argv[i];
    };

    if (arg == "--graph") {
      const char* v = need_value(arg);
      if (!v) return false;
      opts->graph_path = v;
    } else if (arg == "--seeds") {
      const char* v = need_value(arg);
      if (!v) return false;
      opts->seeds_path = v;
    } else if (arg == "--out") {
      const char* v = need_value(arg);
      if (!v) return false;
      opts->out_path = v;
    } else if (arg == "--log") {
      const char* v = need_value(arg);
      if (!v) return false;
      opts->log_path = v;
    } else if (arg == "--trace_csv") {
      const char* v = need_value(arg);
      if (!v) return false;
      opts->trace_csv_path = v;
    } else if (arg == "--trace_json") {
      const char* v = need_value(arg);
      if (!v) return false;
      opts->trace_json_path = v;
    } else if (arg == "--progress") {
      opts->progress = true;
    } else if (arg == "--undirected") {
      opts->undirected = true;
    } else if (arg == "--model") {
      const char* v = need_value(arg);
      if (!v) return false;
      const std::string model = v;
      if (model == "ic") {
        opts->model = DiffusionModel::kIC;
      } else if (model == "lt") {
        opts->model = DiffusionModel::kLT;
      } else {
        if (err) *err = "Unknown model: " + model;
        return false;
      }
    } else if (arg == "--algorithm") {
      const char* v = need_value(arg);
      if (!v) return false;
      bool ok = false;
      opts->algorithm = ParseAlgorithm(v, &ok);
      if (!ok) {
        if (err) *err = "Invalid --algorithm value.";
        return false;
      }
    } else if (arg == "--subproblem_solver") {
      const char* v = need_value(arg);
      if (!v) return false;
      bool ok = false;
      opts->subproblem_solver = ParseSubproblemSolver(v, &ok);
      if (!ok) {
        if (err) *err = "Invalid --subproblem_solver value.";
        return false;
      }
    } else if (arg == "--ic_p_rule") {
      const char* v = need_value(arg);
      if (!v) return false;
      bool ok = false;
      opts->ic_rule = ParseRule(v, false, &ok);
      if (!ok) {
        if (err) *err = "Invalid --ic_p_rule value.";
        return false;
      }
    } else if (arg == "--lt_w_rule") {
      const char* v = need_value(arg);
      if (!v) return false;
      bool ok = false;
      opts->lt_rule = ParseRule(v, true, &ok);
      if (!ok) {
        if (err) *err = "Invalid --lt_w_rule value.";
        return false;
      }
    } else if (arg == "--lt_normalize_in") {
      opts->lt_normalize_in = true;
    } else if (arg == "--p") {
      const char* v = need_value(arg);
      if (!v) return false;
      if (!ParseDouble(v, &opts->p)) {
        if (err) *err = "Invalid --p value.";
        return false;
      }
    } else if (arg == "--c") {
      const char* v = need_value(arg);
      if (!v) return false;
      if (!ParseDouble(v, &opts->c)) {
        if (err) *err = "Invalid --c value.";
        return false;
      }
    } else if (arg == "--tau") {
      const char* v = need_value(arg);
      if (!v) return false;
      if (!ParseDouble(v, &opts->tau)) {
        if (err) *err = "Invalid --tau value.";
        return false;
      }
    } else if (arg == "--eps") {
      const char* v = need_value(arg);
      if (!v) return false;
      if (!ParseDouble(v, &opts->eps)) {
        if (err) *err = "Invalid --eps value.";
        return false;
      }
    } else if (arg == "--delta") {
      const char* v = need_value(arg);
      if (!v) return false;
      if (!ParseDouble(v, &opts->delta)) {
        if (err) *err = "Invalid --delta value.";
        return false;
      }
    } else if (arg == "--gamma") {
      const char* v = need_value(arg);
      if (!v) return false;
      if (!ParseDouble(v, &opts->gamma)) {
        if (err) *err = "Invalid --gamma value.";
        return false;
      }
    } else if (arg == "--rng_seed") {
      const char* v = need_value(arg);
      if (!v) return false;
      try {
        opts->rng_seed = static_cast<std::uint64_t>(std::stoull(v));
      } catch (...) {
        if (err) *err = "Invalid --rng_seed value.";
        return false;
      }
    } else if (arg == "--sample_cap") {
      const char* v = need_value(arg);
      if (!v) return false;
      if (!ParseUInt64(v, &opts->sample_cap)) {
        if (err) *err = "Invalid --sample_cap value.";
        return false;
      }
    } else if (arg == "--eval_cap") {
      const char* v = need_value(arg);
      if (!v) return false;
      if (!ParseUInt64(v, &opts->eval_cap)) {
        if (err) *err = "Invalid --eval_cap value.";
        return false;
      }
    } else if (arg == "--pilot_chunk_bytes") {
      const char* v = need_value(arg);
      if (!v) return false;
      if (!ParseUInt64(v, &opts->pilot_chunk_bytes)) {
        if (err) *err = "Invalid --pilot_chunk_bytes value.";
        return false;
      }
    } else if (arg == "--baseline_chunk_bytes") {
      const char* v = need_value(arg);
      if (!v) return false;
      if (!ParseUInt64(v, &opts->baseline_chunk_bytes)) {
        if (err) *err = "Invalid --baseline_chunk_bytes value.";
        return false;
      }
    } else if (arg == "--trials") {
      const char* v = need_value(arg);
      if (!v) return false;
      std::uint64_t parsed_trials = 0;
      if (!ParseUInt64(v, &parsed_trials) ||
          parsed_trials > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())) {
        if (err) *err = "Invalid --trials value.";
        return false;
      }
      opts->trials = static_cast<std::uint32_t>(parsed_trials);
    } else if (arg == "--local_search_max_iters") {
      const char* v = need_value(arg);
      if (!v) return false;
      std::uint64_t parsed_iters = 0;
      if (!ParseUInt64(v, &parsed_iters) ||
          parsed_iters > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())) {
        if (err) *err = "Invalid --local_search_max_iters value.";
        return false;
      }
      opts->local_search_max_iters = static_cast<std::uint32_t>(parsed_iters);
    } else if (arg == "--local_search_swap_candidates") {
      const char* v = need_value(arg);
      if (!v) return false;
      std::uint64_t parsed_candidates = 0;
      if (!ParseUInt64(v, &parsed_candidates) ||
          parsed_candidates > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())) {
        if (err) *err = "Invalid --local_search_swap_candidates value.";
        return false;
      }
      opts->local_search_swap_candidates = static_cast<std::uint32_t>(parsed_candidates);
    } else if (arg == "--help" || arg == "-h") {
      if (err) *err = "help";
      return false;
    } else {
      if (err) *err = "Unknown flag: " + arg;
      return false;
    }
  }

  return true;
}

bool ValidateOptions(const RuntimeOptions& opts, std::string* err) {
  // This step enforces argument combinations that cannot be checked while flags
  // are parsed independently.
  if (opts.graph_path.empty()) {
    if (err) *err = "Missing required flag: --graph";
    return false;
  }
  if (opts.seeds_path.empty()) {
    if (err) *err = "Missing required flag: --seeds";
    return false;
  }
  if (!(opts.tau > 0.0 && opts.tau < 1.0)) {
    if (err) *err = "tau must be in (0, 1).";
    return false;
  }
  if (!(opts.eps > 0.0)) {
    if (err) *err = "eps must be > 0.";
    return false;
  }
  if (!(opts.delta > 0.0 && opts.delta < 1.0)) {
    if (err) *err = "delta must be in (0, 1).";
    return false;
  }
  if (!(opts.gamma > 0.0)) {
    if (err) *err = "gamma must be > 0.";
    return false;
  }
  if (opts.pilot_chunk_bytes < sizeof(std::uint64_t)) {
    if (err) *err = "pilot_chunk_bytes must be at least 8.";
    return false;
  }
  if (opts.baseline_chunk_bytes < sizeof(std::uint64_t)) {
    if (err) *err = "baseline_chunk_bytes must be at least 8.";
    return false;
  }
  if (opts.trials == 0) {
    if (err) *err = "trials must be >= 1.";
    return false;
  }
  if (opts.local_search_max_iters == 0) {
    if (err) *err = "local_search_max_iters must be >= 1.";
    return false;
  }
  if ((opts.algorithm == AlgorithmKind::kBestPair ||
       opts.algorithm == AlgorithmKind::kTopKActivationProb ||
       opts.algorithm == AlgorithmKind::kTopKCoactivationDegree ||
       opts.algorithm == AlgorithmKind::kGreedyAdd ||
       opts.algorithm == AlgorithmKind::kLocalSearch ||
       opts.algorithm == AlgorithmKind::kEdgeCoactPeeling ||
       opts.algorithm == AlgorithmKind::kCertifiedCapNoEarlyStop) &&
      opts.sample_cap == 0) {
    if (err) {
      *err = std::string(AlgorithmName(opts.algorithm)) +
             " requires --sample_cap >= 1.";
    }
    return false;
  }
  if (opts.subproblem_solver == SubproblemSolverKind::kExact &&
      !(opts.algorithm == AlgorithmKind::kDcgs ||
        opts.algorithm == AlgorithmKind::kCertifiedCapNoEarlyStop)) {
    if (err) {
      *err = "--subproblem_solver=exact is only valid for dcgs and "
             "certified_cap_no_early_stop.";
    }
    return false;
  }
  return true;
}

bool LoadSeeds(const std::string& seeds_path, NodeCount n, std::vector<NodeId>* seeds, std::string* err) {
  // Seeds stay as a small text input: load once, validate against n, and
  // remove duplicates so all downstream modules can assume a clean seed set.
  if (!seeds) {
    if (err) *err = "Internal error: seeds is null.";
    return false;
  }

  std::ifstream fin(seeds_path);
  if (!fin) {
    if (err) *err = "Failed to open seeds file: " + seeds_path;
    return false;
  }

  std::vector<std::uint8_t> seen(static_cast<std::size_t>(n), 0);
  seeds->clear();

  std::string line;
  while (std::getline(fin, line)) {
    if (line.empty() || line[0] == '#') continue;
    NodeId u = kInvalidNode;
    if (!ParseNodeIdToken(line, &u)) {
      if (err) *err = "Invalid seed line: " + line;
      return false;
    }
    if (u >= n) {
      if (err) *err = "Seed id out of range: " + std::to_string(u);
      return false;
    }
    if (!seen[u]) {
      seen[u] = 1;
      seeds->push_back(u);
    }
  }
  return true;
}

}  // namespace dcgs
