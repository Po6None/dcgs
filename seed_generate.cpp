#include "seed_generator.h"

#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {

// PrintUsage documents the standalone seed-generator CLI.
void PrintUsage() {
  std::cout
      << "Usage:\n"
      << "  ./seed_generate --graph <path> --strategy <name> --k <count> [options]\n\n"
      << "Strategies:\n"
      << "  random           Uniformly sample k distinct node ids\n"
      << "  top_indeg        Select the top-k nodes by in-degree\n"
      << "  top_outdeg       Select the top-k nodes by out-degree\n\n"
      << "Options:\n"
      << "  --graph <path>   Input text edge list or .dcgsbin graph\n"
      << "  --out <path>     Output seed file (default: result/generated_seeds.txt)\n"
      << "  --k <count>      Number of seeds to generate\n"
      << "  --strategy <s>   One of random|top_indeg|top_outdeg\n"
      << "  --rng_seed <u64> Base seed for random strategy (default: 1)\n"
      << "  --undirected     Materialize reverse edges for text graphs only\n"
      << "  --help           Show this message\n";
}

// ParseUInt64 parses one unsigned integer flag value.
bool ParseUInt64(const std::string& text, std::uint64_t* value) {
  if (!value || text.empty() || text[0] == '-') return false;
  try {
    std::size_t idx = 0;
    const std::uint64_t parsed = std::stoull(text, &idx);
    if (idx != text.size()) return false;
    *value = parsed;
    return true;
  } catch (...) {
    return false;
  }
}

// ParseCommandLine handles only the flags relevant to offline seed generation.
bool ParseCommandLine(int argc,
                      char** argv,
                      dcgs::SeedGeneratorOptions* opts,
                      std::string* err) {
  if (!opts) {
    if (err) *err = "Internal error: opts is null.";
    return false;
  }

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      if (err) *err = "help";
      return false;
    }
    if (arg == "--undirected") {
      opts->undirected = true;
      continue;
    }
    if (arg == "--graph" || arg == "--out" || arg == "--strategy" ||
        arg == "--k" || arg == "--rng_seed") {
      if (i + 1 >= argc) {
        if (err) *err = "Missing value after " + arg;
        return false;
      }
      const std::string value = argv[++i];
      if (arg == "--graph") {
        opts->graph_path = value;
      } else if (arg == "--out") {
        opts->out_path = value;
      } else if (arg == "--strategy") {
        if (!dcgs::ParseSeedSelectionStrategy(value, &opts->strategy)) {
          if (err) *err = "Unknown seed strategy: " + value;
          return false;
        }
      } else if (arg == "--k") {
        std::uint64_t parsed = 0;
        if (!ParseUInt64(value, &parsed) ||
            parsed > static_cast<std::uint64_t>(std::numeric_limits<dcgs::NodeCount>::max())) {
          if (err) *err = "Invalid --k value: " + value;
          return false;
        }
        opts->k = static_cast<dcgs::NodeCount>(parsed);
      } else if (arg == "--rng_seed") {
        std::uint64_t parsed = 0;
        if (!ParseUInt64(value, &parsed)) {
          if (err) *err = "Invalid --rng_seed value: " + value;
          return false;
        }
        opts->rng_seed = parsed;
      }
      continue;
    }

    if (err) *err = "Unknown argument: " + arg;
    return false;
  }

  if (opts->graph_path.empty()) {
    if (err) *err = "Missing required argument --graph";
    return false;
  }
  if (opts->k == 0) {
    if (err) *err = "k must be >= 1.";
    return false;
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  dcgs::SeedGeneratorOptions opts;
  std::string err;
  if (!ParseCommandLine(argc, argv, &opts, &err)) {
    if (err != "help") std::cerr << "Argument error: " << err << "\n\n";
    PrintUsage();
    return (err == "help") ? 0 : 1;
  }

  dcgs::GraphDegreeStats stats;
  if (!dcgs::LoadGraphDegreeStats(opts, &stats, &err)) {
    std::cerr << "Graph degree loading failed: " << err << "\n";
    return 1;
  }

  std::mt19937_64 rng(opts.rng_seed);
  std::vector<dcgs::NodeId> seeds;
  if (!dcgs::GenerateSeeds(stats, opts, &rng, &seeds, &err)) {
    std::cerr << "Seed generation failed: " << err << "\n";
    return 1;
  }
  if (!dcgs::SaveSeedsFile(opts.out_path, seeds, &err)) {
    std::cerr << "Seed file write failed: " << err << "\n";
    return 1;
  }

  std::cout << "graph=" << opts.graph_path << "\n"
            << "out=" << opts.out_path << "\n"
            << "strategy=" << dcgs::SeedSelectionStrategyName(opts.strategy) << "\n"
            << "k=" << opts.k << "\n"
            << "num_nodes=" << stats.num_nodes << "\n"
            << "num_edges=" << stats.num_edges << "\n"
            << "undirected_materialized="
            << (stats.undirected_materialized ? 1 : 0) << "\n"
            << "rng_seed=" << opts.rng_seed << "\n";
  return 0;
}
