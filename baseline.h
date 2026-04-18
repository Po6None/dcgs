#ifndef BASELINE_H_
#define BASELINE_H_

#include "dcgs_pipeline.h"
#include "diffusion_simulator.h"
#include "peeling.h"

#include <cstdint>
#include <random>
#include <vector>

namespace dcgs {

// CountedPairRun stores one sorted counted run of sampled pair occurrences.
struct CountedPairRun {
  std::vector<std::uint64_t> keys;
  std::vector<SampleCount> counts;
};

// BaselineWorkspace collects all reusable buffers shared by the sample-budget
// baselines.
struct BaselineWorkspace {
  DiffusionWorkspace diffusion_workspace;

  std::vector<std::uint32_t> seed_mark;
  std::uint32_t seed_epoch = 1;
  std::vector<std::uint32_t> candidate_mark;
  std::uint32_t candidate_epoch = 1;
  std::vector<NodeCount> rank_by_node;
  std::vector<NodeId> filtered_active_nodes;
  std::vector<SampleCount> activation_counts;
  std::vector<Real> node_scores;
  std::vector<NodeId> touched_activation_nodes;
  std::vector<Real> prefix_weight_starts;
  std::vector<EdgeOffset> adj_offsets;
  std::vector<NodeCount> adj_neighbors;
  std::vector<Real> adj_weights;
  std::vector<Real> gain_by_local;
  std::vector<std::uint8_t> selected_local;
  std::vector<std::pair<Real, NodeCount>> max_heap_entries;
  std::vector<NodeCount> candidate_rank_by_local;
  std::vector<NodeCount> touched_candidate_nodes;
  std::vector<std::pair<Real, NodeCount>> add_candidates;
  std::vector<std::pair<Real, NodeCount>> remove_candidates;
  std::vector<Real> swap_edge_weights;
  std::vector<std::uint8_t> active_mark;
  std::vector<NodeId> touched_active_nodes;
  std::vector<SampleCount> edge_activation_counts;
  std::vector<Real> edge_weights;
  PeelingWorkspace peeling_workspace;

  std::vector<std::uint64_t> pair_buffer;
  std::vector<std::uint64_t> chunk_buffer;

  std::vector<std::uint64_t> global_keys;
  std::vector<SampleCount> global_counts;

  std::vector<CountedPairRun> run_levels;
  std::vector<std::uint64_t> carry_keys;
  std::vector<SampleCount> carry_counts;
  std::vector<std::uint64_t> temp_keys;
  std::vector<SampleCount> temp_counts;
};

// RunBestPairBaseline returns the heaviest sampled thresholded pair.
DcgsResult RunBestPairBaseline(const Graph& graph,
                               const std::vector<NodeId>& seeds,
                               const RuntimeOptions& opts,
                               BaselineWorkspace* workspace,
                               std::mt19937_64* rng);

// Convenience overload that uses a temporary BaselineWorkspace.
DcgsResult RunBestPairBaseline(const Graph& graph,
                               const std::vector<NodeId>& seeds,
                               const RuntimeOptions& opts,
                               std::mt19937_64* rng);

// RunTopKActivationProbBaseline ranks nodes by activation probability and scans
// all prefixes under the sampled rho objective.
DcgsResult RunTopKActivationProbBaseline(const Graph& graph,
                                         const std::vector<NodeId>& seeds,
                                         const RuntimeOptions& opts,
                                         BaselineWorkspace* workspace,
                                         std::mt19937_64* rng);

DcgsResult RunTopKActivationProbBaseline(const Graph& graph,
                                         const std::vector<NodeId>& seeds,
                                         const RuntimeOptions& opts,
                                         std::mt19937_64* rng);

// RunTopKCoactivationDegreeBaseline ranks nodes by thresholded coactivation
// degree and scans all prefixes under the sampled rho objective.
DcgsResult RunTopKCoactivationDegreeBaseline(const Graph& graph,
                                            const std::vector<NodeId>& seeds,
                                            const RuntimeOptions& opts,
                                            BaselineWorkspace* workspace,
                                            std::mt19937_64* rng);

DcgsResult RunTopKCoactivationDegreeBaseline(const Graph& graph,
                                            const std::vector<NodeId>& seeds,
                                            const RuntimeOptions& opts,
                                            std::mt19937_64* rng);

// RunGreedyAddBaseline starts from the best sampled pair and greedily adds
// nodes while the sampled objective improves.
DcgsResult RunGreedyAddBaseline(const Graph& graph,
                               const std::vector<NodeId>& seeds,
                               const RuntimeOptions& opts,
                               BaselineWorkspace* workspace,
                               std::mt19937_64* rng);

DcgsResult RunGreedyAddBaseline(const Graph& graph,
                               const std::vector<NodeId>& seeds,
                               const RuntimeOptions& opts,
                               std::mt19937_64* rng);

// RunLocalSearchBaseline refines the greedy solution with add/remove/swap moves
// on the sampled thresholded graph.
DcgsResult RunLocalSearchBaseline(const Graph& graph,
                                  const std::vector<NodeId>& seeds,
                                  const RuntimeOptions& opts,
                                  BaselineWorkspace* workspace,
                                  std::mt19937_64* rng);

DcgsResult RunLocalSearchBaseline(const Graph& graph,
                                  const std::vector<NodeId>& seeds,
                                  const RuntimeOptions& opts,
                                  std::mt19937_64* rng);

// RunStructurePeelingBaseline applies weighted peeling on the original graph
// structure converted to a simple undirected graph (unit edge weights).
DcgsResult RunStructurePeelingBaseline(const Graph& graph,
                                       const std::vector<NodeId>& seeds,
                                       const RuntimeOptions& opts,
                                       BaselineWorkspace* workspace,
                                       std::mt19937_64* rng);

DcgsResult RunStructurePeelingBaseline(const Graph& graph,
                                       const std::vector<NodeId>& seeds,
                                       const RuntimeOptions& opts,
                                       std::mt19937_64* rng);

// RunEdgeCoactPeelingBaseline estimates co-activation similarity only on edges
// from the original undirected structure, then runs weighted peeling.
DcgsResult RunEdgeCoactPeelingBaseline(const Graph& graph,
                                       const std::vector<NodeId>& seeds,
                                       const RuntimeOptions& opts,
                                       BaselineWorkspace* workspace,
                                       std::mt19937_64* rng);

DcgsResult RunEdgeCoactPeelingBaseline(const Graph& graph,
                                       const std::vector<NodeId>& seeds,
                                       const RuntimeOptions& opts,
                                       std::mt19937_64* rng);

// RunPilotGreedyAblation reuses the pilot candidate set and then runs the
// sample-budget greedy add stage restricted to those candidate nodes.
DcgsResult RunPilotGreedyAblation(const Graph& graph,
                                  const std::vector<NodeId>& seeds,
                                  const std::vector<NodeId>& candidate_nodes,
                                  const RuntimeOptions& opts,
                                  BaselineWorkspace* workspace,
                                  std::mt19937_64* rng);

DcgsResult RunPilotGreedyAblation(const Graph& graph,
                                  const std::vector<NodeId>& seeds,
                                  const std::vector<NodeId>& candidate_nodes,
                                  const RuntimeOptions& opts,
                                  std::mt19937_64* rng);

// RunPilotPeelingCapAblation reuses pilot-discovered tracked candidates and
// performs fixed-cap sampling followed by one weighted peeling solve.
DcgsResult RunPilotPeelingCapAblation(const Graph& graph,
                                      const std::vector<NodeId>& seeds,
                                      const TrackedGraph& tracked_graph,
                                      const RuntimeOptions& opts,
                                      BaselineWorkspace* workspace,
                                      std::mt19937_64* rng);

DcgsResult RunPilotPeelingCapAblation(const Graph& graph,
                                      const std::vector<NodeId>& seeds,
                                      const TrackedGraph& tracked_graph,
                                      const RuntimeOptions& opts,
                                      std::mt19937_64* rng);

}  // namespace dcgs

#endif  // BASELINE_H_
