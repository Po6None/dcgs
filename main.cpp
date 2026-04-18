#include "dcgs.hpp"

#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

// PrintUsage documents the public CLI expected by the main experiment runner.
void PrintUsage() {
  std::cout
      << "Usage:\n"
      << "  ./dcgs --graph <path> --seeds <path> [options]\n\n"
      << "Core options:\n"
      << "  --algorithm dcgs|pilot_greedy|pilot_peeling_cap|best_pair|topk_activation_prob|topk_coactivation_degree|\n"
      << "              greedy_add|local_search|structure_peeling|\n"
      << "              edge_coact_peeling|certified_cap_no_early_stop\n"
      << "  --subproblem_solver peeling|exact\n"
      << "  --model ic|lt\n"
      << "  --tau <float> --eps <float> --delta <float> --gamma <float>\n"
      << "  --rng_seed <uint64> --trials <uint32> --sample_cap <uint64> --eval_cap <uint64>\n"
      << "  --local_search_max_iters <uint32>\n"
      << "  --local_search_swap_candidates <uint32>   (0 disables swap)\n"
      << "  --pilot_chunk_bytes <uint64> --baseline_chunk_bytes <uint64>\n"
      << "  --progress\n"
      << "  --out <path> --log <path>\n"
      << "  --trace_csv <path> --trace_json <path>\n\n"
      << "Graph options:\n"
      << "  --undirected\n\n"
      << "IC edge rule:\n"
      << "  --ic_p_rule const|c_over_indeg|file\n"
      << "  --p <float> --c <float>\n\n"
      << "LT edge rule:\n"
      << "  --lt_w_rule uniform_in|c_over_indeg|file\n"
      << "  --lt_normalize_in\n";
}

}  // namespace

int main(int argc, char** argv) {
  // main wires together parsing, input loading, trial execution, and output
  // serialization. All algorithm-specific logic stays in library modules.
  dcgs::RuntimeOptions opts;
  std::string err;

  if (!dcgs::ParseCommandLine(argc, argv, &opts, &err)) {
    if (err != "help") std::cerr << "Argument error: " << err << "\n\n";
    PrintUsage();
    return (err == "help") ? 0 : 1;
  }
  if (!dcgs::ValidateOptions(opts, &err)) {
    std::cerr << "Option validation failed: " << err << "\n";
    return 1;
  }

  dcgs::Graph graph;
  if (!dcgs::LoadGraph(opts, &graph, &err)) {
    std::cerr << "Graph load failed: " << err << "\n";
    return 1;
  }

  std::vector<dcgs::NodeId> seeds;
  if (!dcgs::LoadSeeds(opts.seeds_path, graph.num_nodes, &seeds, &err)) {
    std::cerr << "Seed load failed: " << err << "\n";
    return 1;
  }

  dcgs::DcgsResult best;
  if (!dcgs::RunTrials(graph, seeds, opts, &best, &err)) {
    std::cerr << "Run failed: " << err << "\n";
    return 1;
  }

  dcgs::PrintSummary(opts, best);
  if (!dcgs::SaveNodeList(opts.out_path, best.u_hat, &err)) {
    std::cerr << "Write output failed: " << err << "\n";
    return 1;
  }

  if (!dcgs::SaveLog(opts.log_path, opts, best, &err)) {
    std::cerr << "Write log failed: " << err << "\n";
    return 1;
  }
  if (!dcgs::SaveCheckpointTraceCsv(opts.trace_csv_path, best, &err)) {
    std::cerr << "Write checkpoint CSV failed: " << err << "\n";
    return 1;
  }
  if (!dcgs::SaveCheckpointTraceJson(opts.trace_json_path, opts, best, &err)) {
    std::cerr << "Write checkpoint JSON failed: " << err << "\n";
    return 1;
  }

  return 0;
}
