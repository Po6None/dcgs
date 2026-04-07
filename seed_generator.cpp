#include "seed_generator.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <set>
#include <unordered_map>

namespace dcgs {

namespace {

// HasBinarySuffix keeps seed generation aligned with the main runtime's
// automatic .dcgsbin loading convention.
bool HasBinarySuffix(const std::string& path) {
  constexpr const char* kSuffix = ".dcgsbin";
  const std::size_t suffix_len = 8;
  return path.size() >= suffix_len &&
         path.compare(path.size() - suffix_len, suffix_len, kSuffix) == 0;
}

// EnsureParentDirectory creates the parent directory of one output path before
// the seed file itself is opened.
bool EnsureParentDirectory(const std::string& path, std::string* err) {
  if (path.empty()) return true;

  const std::filesystem::path fs_path(path);
  const std::filesystem::path parent = fs_path.parent_path();
  if (parent.empty()) return true;

  std::error_code ec;
  if (std::filesystem::exists(parent, ec)) {
    if (!ec) return true;
    if (err) *err = "Failed to inspect output directory: " + parent.string();
    return false;
  }
  if (std::filesystem::create_directories(parent, ec)) return true;
  if (!ec && std::filesystem::exists(parent)) return true;
  if (err) *err = "Failed to create output directory: " + parent.string();
  return false;
}

// BetterDegreeCandidate prefers larger degree and breaks ties by smaller node
// id so top-k selection stays deterministic across runs.
bool BetterDegreeCandidate(const std::pair<Degree, NodeId>& lhs,
                           const std::pair<Degree, NodeId>& rhs) {
  if (lhs.first != rhs.first) return lhs.first > rhs.first;
  return lhs.second < rhs.second;
}

// WorstDegreeCandidateOrder sorts the bounded multiset so begin() always points
// to the currently worst retained node.
struct WorstDegreeCandidateOrder {
  bool operator()(const std::pair<Degree, NodeId>& lhs,
                  const std::pair<Degree, NodeId>& rhs) const {
    if (lhs.first != rhs.first) return lhs.first < rhs.first;
    return lhs.second > rhs.second;
  }
};

// SelectTopKByDegree scans one degree array and maintains only the current top
// k nodes, avoiding a full O(n log n) sort.
std::vector<NodeId> SelectTopKByDegree(const std::vector<Degree>& degree,
                                       NodeCount k) {
  std::multiset<std::pair<Degree, NodeId>, WorstDegreeCandidateOrder> best;
  for (NodeId node = 0; node < static_cast<NodeId>(degree.size()); ++node) {
    const std::pair<Degree, NodeId> candidate(degree[node], node);
    if (best.size() < static_cast<std::size_t>(k)) {
      best.insert(candidate);
      continue;
    }
    const auto worst_it = best.begin();
    if (worst_it != best.end() && BetterDegreeCandidate(candidate, *worst_it)) {
      best.erase(worst_it);
      best.insert(candidate);
    }
  }

  std::vector<std::pair<Degree, NodeId>> ranked(best.begin(), best.end());
  std::sort(ranked.begin(), ranked.end(),
            [](const std::pair<Degree, NodeId>& lhs,
               const std::pair<Degree, NodeId>& rhs) {
              return BetterDegreeCandidate(lhs, rhs);
            });

  std::vector<NodeId> seeds;
  seeds.reserve(ranked.size());
  for (const auto& candidate : ranked) seeds.push_back(candidate.second);
  return seeds;
}

// LookupMappedValue is the helper used by Floyd's sampling algorithm to keep a
// compact remapping table instead of storing all nodes.
NodeId LookupMappedValue(const std::unordered_map<NodeId, NodeId>& remap,
                         NodeId key) {
  const auto it = remap.find(key);
  return (it == remap.end()) ? key : it->second;
}

// SelectRandomK uses Floyd sampling to draw k unique node ids without
// allocating a full permutation of [0, n).
std::vector<NodeId> SelectRandomK(NodeCount n, NodeCount k, std::mt19937_64* rng) {
  std::vector<NodeId> seeds;
  if (k == 0) return seeds;

  seeds.reserve(static_cast<std::size_t>(k));
  std::unordered_map<NodeId, NodeId> remap;
  remap.reserve(static_cast<std::size_t>(k) * 2);
  const std::uint64_t start = static_cast<std::uint64_t>(n) - static_cast<std::uint64_t>(k);

  for (std::uint64_t value = start; value < static_cast<std::uint64_t>(n); ++value) {
    const NodeId current = static_cast<NodeId>(value);
    std::uniform_int_distribution<std::uint64_t> dist(0, value);
    const NodeId picked = static_cast<NodeId>(dist(*rng));
    const NodeId selected = LookupMappedValue(remap, picked);
    const NodeId replacement = LookupMappedValue(remap, current);
    remap[picked] = replacement;
    seeds.push_back(selected);
  }

  std::sort(seeds.begin(), seeds.end());
  return seeds;
}

}  // namespace

const char* SeedSelectionStrategyName(SeedSelectionStrategy strategy) {
  switch (strategy) {
    case SeedSelectionStrategy::kRandom:
      return "random";
    case SeedSelectionStrategy::kTopInDegree:
      return "top_indeg";
    case SeedSelectionStrategy::kTopOutDegree:
      return "top_outdeg";
  }
  return "unknown";
}

bool ParseSeedSelectionStrategy(const std::string& text,
                                SeedSelectionStrategy* strategy) {
  if (!strategy) return false;
  if (text == "random") {
    *strategy = SeedSelectionStrategy::kRandom;
    return true;
  }
  if (text == "top_indeg") {
    *strategy = SeedSelectionStrategy::kTopInDegree;
    return true;
  }
  if (text == "top_outdeg") {
    *strategy = SeedSelectionStrategy::kTopOutDegree;
    return true;
  }
  return false;
}

bool LoadGraphDegreeStats(const SeedGeneratorOptions& opts,
                          GraphDegreeStats* stats,
                          std::string* err) {
  // Seed generation needs only topology summaries, so it deliberately avoids
  // loading the runtime Graph representation and its edge-value arrays.
  if (!stats) {
    if (err) *err = "Internal error: stats is null.";
    return false;
  }

  if (HasBinarySuffix(opts.graph_path)) {
    if (opts.undirected) {
      if (err) {
        *err = "Do not pass --undirected when loading a .dcgsbin file; "
               "undirected materialization must happen during preprocessing.";
      }
      return false;
    }
    return ReadBinaryGraphDegreeStats(opts.graph_path, stats, err);
  }
  return BuildGraphDegreeStatsFromTextEdgeList(opts.graph_path, opts.undirected,
                                               stats, err);
}

bool GenerateSeeds(const GraphDegreeStats& stats,
                   const SeedGeneratorOptions& opts,
                   std::mt19937_64* rng,
                   std::vector<NodeId>* seeds,
                   std::string* err) {
  // The generator always returns sorted unique node ids so the produced file is
  // deterministic and directly consumable by the main runtime.
  if (!seeds) {
    if (err) *err = "Internal error: seeds is null.";
    return false;
  }
  if (opts.k > stats.num_nodes) {
    if (err) *err = "Requested k exceeds the number of nodes in the graph.";
    return false;
  }
  if (opts.strategy == SeedSelectionStrategy::kRandom && !rng) {
    if (err) *err = "Internal error: rng is null for random seed selection.";
    return false;
  }

  switch (opts.strategy) {
    case SeedSelectionStrategy::kRandom:
      *seeds = SelectRandomK(stats.num_nodes, opts.k, rng);
      break;
    case SeedSelectionStrategy::kTopInDegree:
      *seeds = SelectTopKByDegree(stats.indeg, opts.k);
      break;
    case SeedSelectionStrategy::kTopOutDegree:
      *seeds = SelectTopKByDegree(stats.outdeg, opts.k);
      break;
  }

  if (seeds->size() != static_cast<std::size_t>(opts.k)) {
    if (err) *err = "Seed generation returned an unexpected number of nodes.";
    return false;
  }
  std::sort(seeds->begin(), seeds->end());
  return true;
}

bool SaveSeedsFile(const std::string& path,
                   const std::vector<NodeId>& seeds,
                   std::string* err) {
  if (!EnsureParentDirectory(path, err)) return false;

  std::ofstream fout(path);
  if (!fout) {
    if (err) *err = "Failed to open output seeds file: " + path;
    return false;
  }
  for (NodeId node : seeds) fout << node << "\n";
  return true;
}

}  // namespace dcgs
