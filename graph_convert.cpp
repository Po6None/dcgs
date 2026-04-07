#include "graph_binary.h"

#include <fstream>
#include <iostream>
#include <string>

namespace {

struct ConvertOptions {
  std::string input_path;
  std::string output_path;
  bool undirected = false;
};

// PrintUsage shows the converter-specific CLI.
void PrintUsage() {
  std::cout
      << "Usage:\n"
      << "  ./graph_convert --in <edge_list.txt> --out <graph.dcgsbin> [--undirected]\n\n"
      << "Options:\n"
      << "  --in <path>         Input text edge list\n"
      << "  --out <path>        Output binary CSR graph\n"
      << "  --undirected        Materialize reverse edges during preprocessing\n"
      << "  --help              Show this message\n";
}

// ParseCommandLine handles only preprocessing flags and leaves semantic graph
// validation to the conversion routines.
bool ParseCommandLine(int argc,
                      char** argv,
                      ConvertOptions* opts,
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
    if (arg == "--in" || arg == "--out") {
      if (i + 1 >= argc) {
        if (err) *err = "Missing value after " + arg;
        return false;
      }
      const std::string value = argv[++i];
      if (arg == "--in") {
        opts->input_path = value;
      } else {
        opts->output_path = value;
      }
      continue;
    }

    if (err) *err = "Unknown argument: " + arg;
    return false;
  }

  if (opts->input_path.empty()) {
    if (err) *err = "Missing required argument --in";
    return false;
  }
  if (opts->output_path.empty()) {
    if (err) *err = "Missing required argument --out";
    return false;
  }
  return true;
}

// ComputeFileSizeBytes reports the size of the generated binary graph for the
// conversion summary.
std::uint64_t ComputeFileSizeBytes(const std::string& path) {
  std::ifstream fin(path, std::ios::binary);
  if (!fin) return 0;
  fin.seekg(0, std::ios::end);
  return static_cast<std::uint64_t>(fin.tellg());
}

}  // namespace

int main(int argc, char** argv) {
  // The converter is a one-shot preprocessing tool: it validates the text edge
  // list, builds CSR topology, and writes the reusable binary payload.
  ConvertOptions opts;
  std::string err;
  if (!ParseCommandLine(argc, argv, &opts, &err)) {
    if (err != "help") std::cerr << "Argument error: " << err << "\n\n";
    PrintUsage();
    return (err == "help") ? 0 : 1;
  }

  dcgs::BinaryGraphData graph;
  if (!dcgs::BuildBinaryGraphFromTextEdgeList(
          opts.input_path, opts.undirected, &graph, &err)) {
    std::cerr << "Graph conversion failed: " << err << "\n";
    return 1;
  }

  if (!dcgs::WriteBinaryGraphFile(opts.output_path, graph, &err)) {
    std::cerr << "Binary graph write failed: " << err << "\n";
    return 1;
  }

  const std::uint64_t file_size_bytes = ComputeFileSizeBytes(opts.output_path);
  std::cout << "input=" << opts.input_path << "\n"
            << "output=" << opts.output_path << "\n"
            << "num_nodes=" << graph.num_nodes << "\n"
            << "num_edges=" << graph.num_edges << "\n"
            << "has_raw_values=" << (graph.has_raw_values ? 1 : 0) << "\n"
            << "undirected_materialized="
            << (graph.undirected_materialized ? 1 : 0) << "\n"
            << "file_size_bytes=" << file_size_bytes << "\n";
  return 0;
}
