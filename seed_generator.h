#ifndef SEED_GENERATOR_H_
#define SEED_GENERATOR_H_

#include "common.h"
#include "graph_binary.h"

#include <random>
#include <string>
#include <vector>

namespace dcgs {

// SeedSelectionStrategy enumerates the standalone seed-generation policies
// supported by the preprocessing tool.
enum class SeedSelectionStrategy {
  kRandom,
  kTopInDegree,
  kTopOutDegree,
};

// SeedGeneratorOptions carries only the flags relevant to offline seed
// generation and keeps them independent from the main DCGS runtime CLI.
struct SeedGeneratorOptions {
  std::string graph_path;
  std::string out_path = "result/generated_seeds.txt";
  SeedSelectionStrategy strategy = SeedSelectionStrategy::kRandom;
  NodeCount k = 0;
  bool undirected = false;
  std::uint64_t rng_seed = 1;
};

const char* SeedSelectionStrategyName(SeedSelectionStrategy strategy);
bool ParseSeedSelectionStrategy(const std::string& text,
                                SeedSelectionStrategy* strategy);

// LoadGraphDegreeStats loads only node counts and degree arrays so seed
// selection can avoid materializing probabilities or adjacency weights.
bool LoadGraphDegreeStats(const SeedGeneratorOptions& opts,
                          GraphDegreeStats* stats,
                          std::string* err);

// GenerateSeeds applies the selected policy and returns a sorted unique seed
// list ready to be written to disk or consumed by the main runtime.
bool GenerateSeeds(const GraphDegreeStats& stats,
                   const SeedGeneratorOptions& opts,
                   std::mt19937_64* rng,
                   std::vector<NodeId>* seeds,
                   std::string* err);

// SaveSeedsFile writes one node id per line and creates parent directories when
// needed.
bool SaveSeedsFile(const std::string& path,
                   const std::vector<NodeId>& seeds,
                   std::string* err);

}  // namespace dcgs

#endif  // SEED_GENERATOR_H_
