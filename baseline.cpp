#include "baseline.h"

#include <algorithm>
#include <chrono>
#include <functional>
#include <limits>
#include <utility>

namespace dcgs {

namespace {

// NowSeconds provides monotonic wall-clock timing for baseline profiling.
Real NowSeconds() {
  using clock = std::chrono::steady_clock;
  const auto now = clock::now().time_since_epoch();
  return std::chrono::duration<Real>(now).count();
}

// PackPairKey canonicalizes an unordered node pair into one sortable 64-bit key.
std::uint64_t PackPairKey(NodeId u, NodeId v) {
  if (u > v) std::swap(u, v);
  return (static_cast<std::uint64_t>(u) << 32) | static_cast<std::uint32_t>(v);
}

// UnpackPairKey restores a canonicalized pair from the packed 64-bit form.
EdgePair UnpackPairKey(std::uint64_t key) {
  return {static_cast<NodeId>(key >> 32), static_cast<NodeId>(key & 0xffffffffu)};
}

std::size_t ChunkCapacityKeys(std::uint64_t chunk_bytes) {
  const std::uint64_t bytes =
      std::max<std::uint64_t>(chunk_bytes, sizeof(std::uint64_t));
  const std::uint64_t keys = std::max<std::uint64_t>(1, bytes / sizeof(std::uint64_t));
  return static_cast<std::size_t>(
      std::min<std::uint64_t>(keys,
                              static_cast<std::uint64_t>(
                                  std::numeric_limits<std::size_t>::max())));
}

std::size_t PairBufferCapacityKeys(std::size_t chunk_capacity_keys) {
  constexpr std::size_t kPairBufferBytes = 8ull * 1024ull * 1024ull;
  const std::size_t batch_keys =
      std::max<std::size_t>(1, kPairBufferBytes / sizeof(std::uint64_t));
  return std::min(chunk_capacity_keys, batch_keys);
}

// Baseline runs are often repeated many times across trials. Seed membership is
// therefore tracked with an epoch counter so the code does not memset a full
// O(n) array between every invocation.
void AdvanceSeedEpoch(BaselineWorkspace* workspace) {
  if (!workspace) return;
  if (workspace->seed_epoch == std::numeric_limits<std::uint32_t>::max()) {
    std::fill(workspace->seed_mark.begin(), workspace->seed_mark.end(), 0);
    workspace->seed_epoch = 1;
    return;
  }
  ++workspace->seed_epoch;
}

// Candidate membership for pilot_greedy uses the same epoch trick as seeds so
// repeated trials avoid O(n) clears of a dense mark array.
void AdvanceCandidateEpoch(BaselineWorkspace* workspace) {
  if (!workspace) return;
  if (workspace->candidate_epoch == std::numeric_limits<std::uint32_t>::max()) {
    std::fill(workspace->candidate_mark.begin(), workspace->candidate_mark.end(), 0);
    workspace->candidate_epoch = 1;
    return;
  }
  ++workspace->candidate_epoch;
}

void ResetWorkspace(const Graph& graph,
                    const std::vector<NodeId>& seeds,
                    BaselineWorkspace* workspace) {
  // ResetWorkspace clears only the state that actually varies across runs while
  // reusing large allocated buffers whenever possible.
  if (!workspace) return;

  const std::size_t n = static_cast<std::size_t>(graph.num_nodes);
  if (workspace->seed_mark.size() != n) {
    workspace->seed_mark.assign(n, 0);
    workspace->seed_epoch = 1;
  } else {
    AdvanceSeedEpoch(workspace);
  }
  for (NodeId s : seeds) workspace->seed_mark[s] = workspace->seed_epoch;

  if (workspace->activation_counts.size() != n) {
    workspace->activation_counts.assign(n, 0);
  } else {
    for (NodeId node : workspace->touched_activation_nodes) {
      workspace->activation_counts[node] = 0;
    }
  }
  if (workspace->rank_by_node.size() != n) {
    workspace->rank_by_node.assign(n, 0);
  } else {
    for (NodeId node : workspace->touched_activation_nodes) {
      workspace->rank_by_node[node] = 0;
    }
  }
  if (workspace->node_scores.size() != n) {
    workspace->node_scores.assign(n, 0.0);
  } else {
    for (NodeId node : workspace->touched_activation_nodes) {
      workspace->node_scores[node] = 0.0;
    }
  }
  if (workspace->active_mark.size() != n) {
    workspace->active_mark.assign(n, 0);
  } else {
    for (NodeId node : workspace->touched_active_nodes) {
      workspace->active_mark[node] = 0;
    }
  }

  workspace->filtered_active_nodes.clear();
  workspace->touched_activation_nodes.clear();
  workspace->touched_active_nodes.clear();
  workspace->prefix_weight_starts.clear();
  workspace->adj_offsets.clear();
  workspace->adj_neighbors.clear();
  workspace->adj_weights.clear();
  workspace->gain_by_local.clear();
  workspace->selected_local.clear();
  workspace->max_heap_entries.clear();
  workspace->pair_buffer.clear();
  workspace->chunk_buffer.clear();
  workspace->global_keys.clear();
  workspace->global_counts.clear();
  for (CountedPairRun& run : workspace->run_levels) {
    run.keys.clear();
    run.counts.clear();
  }
  workspace->carry_keys.clear();
  workspace->carry_counts.clear();
  workspace->temp_keys.clear();
  workspace->temp_counts.clear();
  workspace->edge_activation_counts.clear();
  workspace->edge_weights.clear();
}

// PrepareSamplingBuffers reserves the front pair buffer under the configured
// baseline chunk budget.
void PrepareSamplingBuffers(std::size_t chunk_capacity_keys,
                            BaselineWorkspace* workspace) {
  if (!workspace) return;
  const std::size_t pair_buffer_capacity_keys = PairBufferCapacityKeys(chunk_capacity_keys);
  if (workspace->pair_buffer.capacity() < pair_buffer_capacity_keys) {
    workspace->pair_buffer.reserve(pair_buffer_capacity_keys);
  }
}

// CompressSortedKeysToCounts converts one sorted chunk of pair keys into
// distinct keys with exact multiplicities.
void CompressSortedKeysToCounts(const std::vector<std::uint64_t>& sorted_keys,
                                std::vector<std::uint64_t>* keys,
                                std::vector<SampleCount>* counts) {
  if (!keys || !counts) return;
  keys->clear();
  counts->clear();
  if (sorted_keys.empty()) return;

  keys->reserve(sorted_keys.size());
  counts->reserve(sorted_keys.size());

  std::uint64_t current_key = sorted_keys[0];
  SampleCount current_count = 1;
  for (std::size_t i = 1; i < sorted_keys.size(); ++i) {
    if (sorted_keys[i] == current_key) {
      ++current_count;
      continue;
    }
    keys->push_back(current_key);
    counts->push_back(current_count);
    current_key = sorted_keys[i];
    current_count = 1;
  }
  keys->push_back(current_key);
  counts->push_back(current_count);
}

// MergeSortedCounts merges two counted sorted runs exactly.
void MergeSortedCounts(const std::vector<std::uint64_t>& lhs_keys,
                       const std::vector<SampleCount>& lhs_counts,
                       const std::vector<std::uint64_t>& rhs_keys,
                       const std::vector<SampleCount>& rhs_counts,
                       std::vector<std::uint64_t>* out_keys,
                       std::vector<SampleCount>* out_counts) {
  if (!out_keys || !out_counts) return;
  out_keys->clear();
  out_counts->clear();
  out_keys->reserve(lhs_keys.size() + rhs_keys.size());
  out_counts->reserve(lhs_counts.size() + rhs_counts.size());

  std::size_t i = 0;
  std::size_t j = 0;
  while (i < lhs_keys.size() && j < rhs_keys.size()) {
    if (lhs_keys[i] < rhs_keys[j]) {
      out_keys->push_back(lhs_keys[i]);
      out_counts->push_back(lhs_counts[i]);
      ++i;
    } else if (rhs_keys[j] < lhs_keys[i]) {
      out_keys->push_back(rhs_keys[j]);
      out_counts->push_back(rhs_counts[j]);
      ++j;
    } else {
      out_keys->push_back(lhs_keys[i]);
      out_counts->push_back(lhs_counts[i] + rhs_counts[j]);
      ++i;
      ++j;
    }
  }

  while (i < lhs_keys.size()) {
    out_keys->push_back(lhs_keys[i]);
    out_counts->push_back(lhs_counts[i]);
    ++i;
  }
  while (j < rhs_keys.size()) {
    out_keys->push_back(rhs_keys[j]);
    out_counts->push_back(rhs_counts[j]);
    ++j;
  }
}

void InsertCarryRun(BaselineWorkspace* workspace) {
  if (!workspace || workspace->carry_keys.empty()) return;

  // Counted runs are merged in leveled slots, similar to a binary carry chain.
  // This avoids repeatedly merging every flushed chunk into one ever-growing
  // global vector after each realization.
  std::size_t level = 0;
  while (true) {
    if (level == workspace->run_levels.size()) {
      workspace->run_levels.emplace_back();
    }
    CountedPairRun& slot = workspace->run_levels[level];
    if (slot.keys.empty()) {
      slot.keys.swap(workspace->carry_keys);
      slot.counts.swap(workspace->carry_counts);
      return;
    }

    MergeSortedCounts(slot.keys,
                      slot.counts,
                      workspace->carry_keys,
                      workspace->carry_counts,
                      &workspace->temp_keys,
                      &workspace->temp_counts);
    slot.keys.clear();
    slot.counts.clear();
    workspace->carry_keys.swap(workspace->temp_keys);
    workspace->carry_counts.swap(workspace->temp_counts);
    workspace->temp_keys.clear();
    workspace->temp_counts.clear();
    ++level;
  }
}

void FlushChunk(BaselineWorkspace* workspace) {
  if (!workspace || workspace->chunk_buffer.empty()) return;

  // Flushing sorts the chunk once, compresses duplicates, and inserts the
  // result into the leveled merge structure.
  std::sort(workspace->chunk_buffer.begin(), workspace->chunk_buffer.end());
  CompressSortedKeysToCounts(workspace->chunk_buffer,
                             &workspace->carry_keys,
                             &workspace->carry_counts);
  workspace->chunk_buffer.clear();
  InsertCarryRun(workspace);
}

// FinalizeRuns collapses every leveled counted run into one global counted list.
void FinalizeRuns(BaselineWorkspace* workspace) {
  if (!workspace) return;

  workspace->global_keys.clear();
  workspace->global_counts.clear();
  for (CountedPairRun& run : workspace->run_levels) {
    if (run.keys.empty()) continue;
    if (workspace->global_keys.empty()) {
      workspace->global_keys.swap(run.keys);
      workspace->global_counts.swap(run.counts);
      continue;
    }
    MergeSortedCounts(workspace->global_keys,
                      workspace->global_counts,
                      run.keys,
                      run.counts,
                      &workspace->temp_keys,
                      &workspace->temp_counts);
    workspace->global_keys.swap(workspace->temp_keys);
    workspace->global_counts.swap(workspace->temp_counts);
    workspace->temp_keys.clear();
    workspace->temp_counts.clear();
    run.keys.clear();
    run.counts.clear();
  }
}

// DrainPairBufferToChunk moves one realization's pair emissions into the
// chunked run-merging pipeline.
void DrainPairBufferToChunk(BaselineWorkspace* workspace,
                            std::size_t chunk_capacity_keys) {
  if (!workspace || workspace->pair_buffer.empty()) return;

  std::size_t offset = 0;
  while (offset < workspace->pair_buffer.size()) {
    if (workspace->chunk_buffer.size() == chunk_capacity_keys) FlushChunk(workspace);
    const std::size_t available = chunk_capacity_keys - workspace->chunk_buffer.size();
    const std::size_t take = std::min(available, workspace->pair_buffer.size() - offset);
    workspace->chunk_buffer.insert(workspace->chunk_buffer.end(),
                                   workspace->pair_buffer.begin() + offset,
                                   workspace->pair_buffer.begin() + offset + take);
    offset += take;
    if (workspace->chunk_buffer.size() == chunk_capacity_keys) FlushChunk(workspace);
  }
  workspace->pair_buffer.clear();
}

// FallbackBestPairNodes supplies a deterministic non-seed pair when no sampled
// thresholded edge survives.
std::vector<NodeId> FallbackBestPairNodes(NodeCount num_nodes,
                                          const std::vector<std::uint32_t>& seed_mark,
                                          std::uint32_t seed_epoch) {
  std::vector<NodeId> fallback;
  fallback.reserve(2);
  for (NodeId node = 0; node < num_nodes && fallback.size() < 2; ++node) {
    if (seed_mark[node] != seed_epoch) fallback.push_back(node);
  }
  return fallback;
}

// FallbackSingleNode supplies a deterministic non-seed singleton for degenerate cases.
std::vector<NodeId> FallbackSingleNode(NodeCount num_nodes,
                                       const std::vector<std::uint32_t>& seed_mark,
                                       std::uint32_t seed_epoch) {
  for (NodeId node = 0; node < num_nodes; ++node) {
    if (seed_mark[node] != seed_epoch) return {node};
  }
  return {};
}

// FallbackCandidateSingleNode keeps pilot_greedy inside the pilot candidate
// scope when no positive sampled pair survives thresholding.
std::vector<NodeId> FallbackCandidateSingleNode(
    const std::vector<NodeId>& candidate_nodes,
    const std::vector<std::uint32_t>& seed_mark,
    std::uint32_t seed_epoch) {
  for (NodeId node : candidate_nodes) {
    if (node < seed_mark.size() && seed_mark[node] != seed_epoch) return {node};
  }
  return {};
}

void FinalizeSampleBudgetBaselineResult(const RuntimeOptions& opts,
                                        Real start,
                                        DcgsResult* result) {
  if (!result) return;
  // Sample-budget baselines bypass pilot/certified checkpointing, but they
  // still reuse the same output structure so downstream reporting remains
  // uniform across all algorithms.
  result->sample_cap_used = opts.sample_cap;
  result->pilot.t0 = 0;
  result->pilot.tracked_graph = TrackedGraph{};
  result->pilot.elapsed_sec = 0.0;
  result->certified.early_stop = false;
  result->certified.t_stop = opts.sample_cap;
  result->certified.checkpoints = 0;
  result->certified.t_max = opts.sample_cap;
  result->certified.checkpoint_trace.clear();
  result->certified.checkpoint_opt_sec = 0.0;
  result->certified.sampling_sec = NowSeconds() - start;
}

void FinalizeStructureBaselineResult(Real start, DcgsResult* result) {
  if (!result) return;
  result->sample_cap_used = 0;
  result->pilot.t0 = 0;
  result->pilot.tracked_graph = TrackedGraph{};
  result->pilot.elapsed_sec = 0.0;
  result->certified.early_stop = false;
  result->certified.t_stop = 0;
  result->certified.checkpoints = 0;
  result->certified.t_max = 0;
  result->certified.checkpoint_trace.clear();
  result->certified.sampling_sec = 0.0;
  result->certified.checkpoint_opt_sec = NowSeconds() - start;
}

void BuildUndirectedUniquePairKeysFromGraph(const Graph& graph,
                                            const RuntimeOptions& opts,
                                            BaselineWorkspace* workspace) {
  if (!workspace) return;

  workspace->pair_buffer.clear();
  workspace->chunk_buffer.clear();
  workspace->global_keys.clear();
  workspace->global_counts.clear();
  for (CountedPairRun& run : workspace->run_levels) {
    run.keys.clear();
    run.counts.clear();
  }
  workspace->carry_keys.clear();
  workspace->carry_counts.clear();
  workspace->temp_keys.clear();
  workspace->temp_counts.clear();

  const std::size_t chunk_capacity_keys = ChunkCapacityKeys(opts.baseline_chunk_bytes);
  PrepareSamplingBuffers(chunk_capacity_keys, workspace);

  for (NodeId u = 0; u < graph.num_nodes; ++u) {
    const EdgeOffset begin = graph.out_offsets[static_cast<std::size_t>(u)];
    const EdgeOffset end = graph.out_offsets[static_cast<std::size_t>(u) + 1];
    for (EdgeOffset pos = begin; pos < end; ++pos) {
      const NodeId v = graph.out_neighbors[static_cast<std::size_t>(pos)];
      if (u == v) continue;
      workspace->pair_buffer.push_back(PackPairKey(u, v));
      if (workspace->pair_buffer.size() == workspace->pair_buffer.capacity()) {
        DrainPairBufferToChunk(workspace, chunk_capacity_keys);
      }
    }
  }
  DrainPairBufferToChunk(workspace, chunk_capacity_keys);
  FlushChunk(workspace);
  FinalizeRuns(workspace);
}

void FilterNonSeedActivated(const std::vector<NodeId>& activated,
                            BaselineWorkspace* workspace) {
  // Baselines ignore seed nodes in their candidate construction logic.
  if (!workspace) return;
  workspace->filtered_active_nodes.clear();
  workspace->filtered_active_nodes.reserve(activated.size());
  for (NodeId node : activated) {
    if (workspace->seed_mark[node] != workspace->seed_epoch) {
      workspace->filtered_active_nodes.push_back(node);
    }
  }
}

// MarkPilotCandidateSet records the candidate-node membership used by the
// pilot_greedy ablation in an epoch-marked dense array.
void MarkPilotCandidateSet(const Graph& graph,
                           const std::vector<NodeId>& candidate_nodes,
                           BaselineWorkspace* workspace) {
  if (!workspace) return;
  const std::size_t n = static_cast<std::size_t>(graph.num_nodes);
  if (workspace->candidate_mark.size() != n) {
    workspace->candidate_mark.assign(n, 0);
    workspace->candidate_epoch = 1;
  } else {
    AdvanceCandidateEpoch(workspace);
  }
  for (NodeId node : candidate_nodes) {
    if (node < graph.num_nodes) {
      workspace->candidate_mark[node] = workspace->candidate_epoch;
    }
  }
}

// FilterPilotCandidateActivated keeps only activated nodes that are both
// non-seeds and members of the pilot candidate set.
void FilterPilotCandidateActivated(const std::vector<NodeId>& activated,
                                   BaselineWorkspace* workspace) {
  if (!workspace) return;
  workspace->filtered_active_nodes.clear();
  workspace->filtered_active_nodes.reserve(activated.size());
  for (NodeId node : activated) {
    if (workspace->seed_mark[node] == workspace->seed_epoch) continue;
    if (workspace->candidate_mark[node] == workspace->candidate_epoch) {
      workspace->filtered_active_nodes.push_back(node);
    }
  }
}

// UpdateActivationCounts tracks how often each non-seed node activates across samples.
void UpdateActivationCounts(BaselineWorkspace* workspace) {
  if (!workspace) return;
  for (NodeId node : workspace->filtered_active_nodes) {
    if (workspace->activation_counts[node] == 0) {
      workspace->touched_activation_nodes.push_back(node);
    }
    ++workspace->activation_counts[node];
  }
}

// EmitFilteredPairsToBuffers sends the realization's co-activated non-seed
// pairs into the chunked counting pipeline.
void EmitFilteredPairsToBuffers(BaselineWorkspace* workspace,
                                std::size_t chunk_capacity_keys) {
  if (!workspace) return;
  const std::size_t k = workspace->filtered_active_nodes.size();
  for (std::size_t i = 0; i < k; ++i) {
    const NodeId u = workspace->filtered_active_nodes[i];
    for (std::size_t j = i + 1; j < k; ++j) {
      const NodeId v = workspace->filtered_active_nodes[j];
      workspace->pair_buffer.push_back(PackPairKey(u, v));
      if (workspace->pair_buffer.size() == workspace->pair_buffer.capacity()) {
        DrainPairBufferToChunk(workspace, chunk_capacity_keys);
      }
    }
  }
  DrainPairBufferToChunk(workspace, chunk_capacity_keys);
}

void RunSamplingPass(const Graph& graph,
                     const std::vector<NodeId>& seeds,
                     SampleCount sample_cap,
                     const RuntimeOptions& opts,
                     bool record_activation_counts,
                     BaselineWorkspace* workspace,
                     std::mt19937_64* rng) {
  if (!workspace) return;
  // All sample-budget baselines share the same diffusion pass. The pass can
  // simultaneously collect activation counts and exact pair counts, so later
  // optimization stages do not have to resample for different baselines.
  const std::size_t chunk_capacity_keys = ChunkCapacityKeys(opts.baseline_chunk_bytes);
  PrepareSamplingBuffers(chunk_capacity_keys, workspace);

  for (SampleCount t = 0; t < sample_cap; ++t) {
    const std::vector<NodeId>& activated =
        SimulateDiffusion(graph, seeds, opts, &workspace->diffusion_workspace, rng);
    FilterNonSeedActivated(activated, workspace);
    if (record_activation_counts) UpdateActivationCounts(workspace);
    EmitFilteredPairsToBuffers(workspace, chunk_capacity_keys);
  }

  FlushChunk(workspace);
  FinalizeRuns(workspace);
}

void RunSamplingPassOnPilotCandidates(const Graph& graph,
                                      const std::vector<NodeId>& seeds,
                                      SampleCount sample_cap,
                                      const RuntimeOptions& opts,
                                      const std::vector<NodeId>& candidate_nodes,
                                      BaselineWorkspace* workspace,
                                      std::mt19937_64* rng) {
  if (!workspace) return;
  const std::size_t chunk_capacity_keys = ChunkCapacityKeys(opts.baseline_chunk_bytes);
  PrepareSamplingBuffers(chunk_capacity_keys, workspace);
  MarkPilotCandidateSet(graph, candidate_nodes, workspace);

  for (SampleCount t = 0; t < sample_cap; ++t) {
    const std::vector<NodeId>& activated =
        SimulateDiffusion(graph, seeds, opts, &workspace->diffusion_workspace, rng);
    FilterPilotCandidateActivated(activated, workspace);
    EmitFilteredPairsToBuffers(workspace, chunk_capacity_keys);
  }

  FlushChunk(workspace);
  FinalizeRuns(workspace);
}

bool ActivationCountLess(NodeId lhs,
                         NodeId rhs,
                         const std::vector<SampleCount>& activation_counts) {
  // Higher activation count is better; node id only breaks ties deterministically.
  if (activation_counts[lhs] != activation_counts[rhs]) {
    return activation_counts[lhs] > activation_counts[rhs];
  }
  return lhs < rhs;
}

// NodeScoreLess orders nodes by coactivation degree with deterministic tie-breaking.
bool NodeScoreLess(NodeId lhs,
                   NodeId rhs,
                   const std::vector<Real>& node_scores) {
  if (node_scores[lhs] != node_scores[rhs]) {
    return node_scores[lhs] > node_scores[rhs];
  }
  return lhs < rhs;
}

void BuildPrefixWeights(BaselineWorkspace* workspace,
                        SampleCount sample_cap,
                        Real tau) {
  if (!workspace) return;
  // After candidates are sorted, each thresholded pair contributes to exactly
  // one prefix boundary: the larger endpoint rank. A single linear scan over
  // prefix sums is then enough to evaluate every k.
  const std::size_t candidate_count = workspace->touched_activation_nodes.size();
  workspace->prefix_weight_starts.assign(candidate_count + 1, 0.0);
  for (std::size_t rank = 0; rank < candidate_count; ++rank) {
    const NodeId node = workspace->touched_activation_nodes[rank];
    workspace->rank_by_node[node] = static_cast<NodeCount>(rank + 1);
  }

  for (std::size_t i = 0; i < workspace->global_keys.size(); ++i) {
    const Real p_hat =
        static_cast<Real>(workspace->global_counts[i]) / static_cast<Real>(sample_cap);
    if (p_hat < tau) continue;
    const EdgePair edge = UnpackPairKey(workspace->global_keys[i]);
    const NodeCount rank_u = workspace->rank_by_node[edge.u];
    const NodeCount rank_v = workspace->rank_by_node[edge.v];
    if (rank_u == 0 || rank_v == 0) continue;
    const std::size_t max_rank = static_cast<std::size_t>(std::max(rank_u, rank_v));
    if (max_rank < workspace->prefix_weight_starts.size()) {
      workspace->prefix_weight_starts[max_rank] += p_hat;
    }
  }
}

void BuildNodeScoresFromThresholdedPairs(BaselineWorkspace* workspace,
                                         SampleCount sample_cap,
                                         Real tau) {
  if (!workspace) return;
  // topk_by_coactivation_degree ranks nodes by the total thresholded pair
  // weight incident on them, computed from the same sampled pair counts.
  workspace->touched_activation_nodes.clear();
  for (std::size_t i = 0; i < workspace->global_keys.size(); ++i) {
    const Real p_hat =
        static_cast<Real>(workspace->global_counts[i]) / static_cast<Real>(sample_cap);
    if (p_hat < tau) continue;
    const EdgePair edge = UnpackPairKey(workspace->global_keys[i]);
    if (workspace->node_scores[edge.u] == 0.0) {
      workspace->touched_activation_nodes.push_back(edge.u);
    }
    if (workspace->node_scores[edge.v] == 0.0) {
      workspace->touched_activation_nodes.push_back(edge.v);
    }
    workspace->node_scores[edge.u] += p_hat;
    workspace->node_scores[edge.v] += p_hat;
  }
}

bool BuildThresholdedAdjacency(BaselineWorkspace* workspace,
                               SampleCount sample_cap,
                               Real tau,
                               NodeCount* best_pair_u,
                               NodeCount* best_pair_v,
                               Real* best_pair_weight) {
  if (!workspace) return false;
  // Greedy-add and local-search operate on the thresholded sampled graph.
  // This helper extracts only positive-threshold edges and reindexes their
  // endpoints to a dense local graph once per baseline invocation.
  workspace->touched_activation_nodes.clear();

  bool found_pair = false;
  Real local_best_weight = -1.0;
  std::uint64_t local_best_key = 0;
  EdgeOffset positive_edges = 0;

  for (std::size_t i = 0; i < workspace->global_keys.size(); ++i) {
    const Real p_hat =
        static_cast<Real>(workspace->global_counts[i]) / static_cast<Real>(sample_cap);
    if (p_hat < tau) continue;
    const EdgePair edge = UnpackPairKey(workspace->global_keys[i]);
    if (workspace->rank_by_node[edge.u] == 0) {
      workspace->touched_activation_nodes.push_back(edge.u);
      workspace->rank_by_node[edge.u] = 1;
    }
    if (workspace->rank_by_node[edge.v] == 0) {
      workspace->touched_activation_nodes.push_back(edge.v);
      workspace->rank_by_node[edge.v] = 1;
    }
    ++positive_edges;
    if (!found_pair || p_hat > local_best_weight ||
        (p_hat == local_best_weight && workspace->global_keys[i] < local_best_key)) {
      found_pair = true;
      local_best_weight = p_hat;
      local_best_key = workspace->global_keys[i];
    }
  }

  if (!found_pair) return false;

  for (std::size_t i = 0; i < workspace->touched_activation_nodes.size(); ++i) {
    const NodeId node = workspace->touched_activation_nodes[i];
    workspace->rank_by_node[node] = static_cast<NodeCount>(i + 1);
  }

  const std::size_t local_n = workspace->touched_activation_nodes.size();
  workspace->adj_offsets.assign(local_n + 1, 0);
  for (std::size_t i = 0; i < workspace->global_keys.size(); ++i) {
    const Real p_hat =
        static_cast<Real>(workspace->global_counts[i]) / static_cast<Real>(sample_cap);
    if (p_hat < tau) continue;
    const EdgePair edge = UnpackPairKey(workspace->global_keys[i]);
    const NodeCount local_u = workspace->rank_by_node[edge.u] - 1;
    const NodeCount local_v = workspace->rank_by_node[edge.v] - 1;
    ++workspace->adj_offsets[static_cast<std::size_t>(local_u) + 1];
    ++workspace->adj_offsets[static_cast<std::size_t>(local_v) + 1];
  }
  for (std::size_t i = 1; i < workspace->adj_offsets.size(); ++i) {
    workspace->adj_offsets[i] += workspace->adj_offsets[i - 1];
  }

  workspace->adj_neighbors.assign(static_cast<std::size_t>(2 * positive_edges), 0);
  workspace->adj_weights.assign(static_cast<std::size_t>(2 * positive_edges), 0.0);
  std::vector<EdgeOffset> cursor = workspace->adj_offsets;
  for (std::size_t i = 0; i < workspace->global_keys.size(); ++i) {
    const Real p_hat =
        static_cast<Real>(workspace->global_counts[i]) / static_cast<Real>(sample_cap);
    if (p_hat < tau) continue;
    const EdgePair edge = UnpackPairKey(workspace->global_keys[i]);
    const NodeCount local_u = workspace->rank_by_node[edge.u] - 1;
    const NodeCount local_v = workspace->rank_by_node[edge.v] - 1;

    EdgeOffset pos = cursor[local_u]++;
    workspace->adj_neighbors[static_cast<std::size_t>(pos)] = local_v;
    workspace->adj_weights[static_cast<std::size_t>(pos)] = p_hat;

    pos = cursor[local_v]++;
    workspace->adj_neighbors[static_cast<std::size_t>(pos)] = local_u;
    workspace->adj_weights[static_cast<std::size_t>(pos)] = p_hat;
  }

  const EdgePair best_pair = UnpackPairKey(local_best_key);
  if (best_pair_u) *best_pair_u = workspace->rank_by_node[best_pair.u] - 1;
  if (best_pair_v) *best_pair_v = workspace->rank_by_node[best_pair.v] - 1;
  if (best_pair_weight) *best_pair_weight = local_best_weight;
  return true;
}

void UpdateHeapGain(NodeCount local_node,
                    Real gain,
                    BaselineWorkspace* workspace) {
  if (!workspace) return;
  workspace->max_heap_entries.push_back({gain, local_node});
  std::push_heap(workspace->max_heap_entries.begin(),
                 workspace->max_heap_entries.end());
}

void InitializeGreedyState(NodeCount start_u,
                           NodeCount start_v,
                           Real start_pair_weight,
                           BaselineWorkspace* workspace) {
  if (!workspace) return;
  const std::size_t local_n = workspace->touched_activation_nodes.size();
  workspace->gain_by_local.assign(local_n, 0.0);
  workspace->selected_local.assign(local_n, 0);
  workspace->max_heap_entries.clear();

  workspace->selected_local[start_u] = 1;
  workspace->selected_local[start_v] = 1;
  workspace->gain_by_local[start_u] = start_pair_weight;
  workspace->gain_by_local[start_v] = start_pair_weight;

  auto add_neighbors = [&](NodeCount selected_node) {
    for (EdgeOffset pos = workspace->adj_offsets[selected_node];
         pos < workspace->adj_offsets[static_cast<std::size_t>(selected_node) + 1];
         ++pos) {
      const NodeCount neighbor = workspace->adj_neighbors[static_cast<std::size_t>(pos)];
      if (workspace->selected_local[neighbor]) continue;
      workspace->gain_by_local[neighbor] +=
          workspace->adj_weights[static_cast<std::size_t>(pos)];
    }
  };

  add_neighbors(start_u);
  add_neighbors(start_v);

  for (NodeCount local_node = 0; local_node < static_cast<NodeCount>(local_n); ++local_node) {
    if (workspace->selected_local[local_node]) continue;
    UpdateHeapGain(local_node, workspace->gain_by_local[local_node], workspace);
  }
}

bool PopBestGain(NodeCount* local_node,
                 Real* gain,
                 BaselineWorkspace* workspace) {
  if (!workspace || !local_node || !gain) return false;
  while (!workspace->max_heap_entries.empty()) {
    std::pop_heap(workspace->max_heap_entries.begin(),
                  workspace->max_heap_entries.end());
    const std::pair<Real, NodeCount> entry = workspace->max_heap_entries.back();
    workspace->max_heap_entries.pop_back();
    if (workspace->selected_local[entry.second]) continue;
    if (entry.first != workspace->gain_by_local[entry.second]) continue;
    *local_node = entry.second;
    *gain = entry.first;
    return true;
  }
  return false;
}

void AddGreedyNode(NodeCount local_node, BaselineWorkspace* workspace) {
  if (!workspace) return;
  workspace->selected_local[local_node] = 1;
  for (EdgeOffset pos = workspace->adj_offsets[local_node];
       pos < workspace->adj_offsets[static_cast<std::size_t>(local_node) + 1];
       ++pos) {
    const NodeCount neighbor = workspace->adj_neighbors[static_cast<std::size_t>(pos)];
    workspace->gain_by_local[neighbor] +=
        workspace->adj_weights[static_cast<std::size_t>(pos)];
    if (!workspace->selected_local[neighbor]) {
      UpdateHeapGain(neighbor, workspace->gain_by_local[neighbor], workspace);
    }
  }
}

void RemoveLocalNode(NodeCount local_node, BaselineWorkspace* workspace) {
  if (!workspace) return;
  workspace->selected_local[local_node] = 0;
  for (EdgeOffset pos = workspace->adj_offsets[local_node];
       pos < workspace->adj_offsets[static_cast<std::size_t>(local_node) + 1];
       ++pos) {
    const NodeCount neighbor = workspace->adj_neighbors[static_cast<std::size_t>(pos)];
    const Real w = workspace->adj_weights[static_cast<std::size_t>(pos)];
    workspace->gain_by_local[neighbor] -= w;
    if (!workspace->selected_local[neighbor]) {
      UpdateHeapGain(neighbor, workspace->gain_by_local[neighbor], workspace);
    }
  }
  UpdateHeapGain(local_node, workspace->gain_by_local[local_node], workspace);
}

void CollectTopAddCandidates(std::size_t limit, BaselineWorkspace* workspace) {
  if (!workspace) return;
  workspace->add_candidates.clear();
  while (workspace->add_candidates.size() < limit) {
    NodeCount candidate = 0;
    Real gain = 0.0;
    if (!PopBestGain(&candidate, &gain, workspace)) break;
    workspace->add_candidates.push_back({gain, candidate});
  }
  for (const auto& entry : workspace->add_candidates) {
    UpdateHeapGain(entry.second, entry.first, workspace);
  }
}

void InsertSmallestRemoveCandidate(std::size_t limit,
                                   Real remove_gain,
                                   NodeCount local_node,
                                   BaselineWorkspace* workspace) {
  if (!workspace || limit == 0) return;
  auto& candidates = workspace->remove_candidates;
  std::size_t pos = 0;
  while (pos < candidates.size() && candidates[pos].first <= remove_gain) {
    ++pos;
  }
  if (pos < limit) {
    candidates.insert(candidates.begin() + static_cast<std::ptrdiff_t>(pos),
                      {remove_gain, local_node});
    if (candidates.size() > limit) {
      candidates.pop_back();
    }
  } else if (candidates.size() < limit) {
    candidates.push_back({remove_gain, local_node});
  }
}

}  // namespace

DcgsResult RunBestPairBaseline(const Graph& graph,
                               const std::vector<NodeId>& seeds,
                               const RuntimeOptions& opts,
                               BaselineWorkspace* workspace,
                               std::mt19937_64* rng) {
  DcgsResult result;
  const Real start = NowSeconds();

  BaselineWorkspace local_workspace;
  if (!workspace) workspace = &local_workspace;
  ResetWorkspace(graph, seeds, workspace);

  result.sample_cap_used = opts.sample_cap;
  if (opts.sample_cap == 0) {
    FinalizeSampleBudgetBaselineResult(opts, start, &result);
    return result;
  }
  RunSamplingPass(graph, seeds, opts.sample_cap, opts, false, workspace, rng);

  // best_pair is the strongest size-2 baseline under the sampled thresholded
  // objective: it simply returns the heaviest sampled pair after tau-thresholding.
  bool found_pair = false;
  std::uint64_t best_key = 0;
  Real best_thresholded_weight = -1.0;
  Real best_raw_weight = -1.0;
  for (std::size_t i = 0; i < workspace->global_keys.size(); ++i) {
    const Real p_hat =
        static_cast<Real>(workspace->global_counts[i]) / static_cast<Real>(opts.sample_cap);
    const Real thresholded_weight = (p_hat >= opts.tau) ? p_hat : 0.0;
    if (!found_pair ||
        thresholded_weight > best_thresholded_weight ||
        (thresholded_weight == best_thresholded_weight && p_hat > best_raw_weight) ||
        (thresholded_weight == best_thresholded_weight &&
         p_hat == best_raw_weight &&
         workspace->global_keys[i] < best_key)) {
      found_pair = true;
      best_key = workspace->global_keys[i];
      best_thresholded_weight = thresholded_weight;
      best_raw_weight = p_hat;
    }
  }

  if (found_pair) {
    const EdgePair best_pair = UnpackPairKey(best_key);
    result.u_hat = {best_pair.u, best_pair.v};
    result.certified.rho_final = best_thresholded_weight / 2.0;
  } else {
    result.u_hat = FallbackBestPairNodes(
        graph.num_nodes, workspace->seed_mark, workspace->seed_epoch);
    result.certified.rho_final = 0.0;
  }

  FinalizeSampleBudgetBaselineResult(opts, start, &result);
  return result;
}

DcgsResult RunBestPairBaseline(const Graph& graph,
                               const std::vector<NodeId>& seeds,
                               const RuntimeOptions& opts,
                               std::mt19937_64* rng) {
  BaselineWorkspace workspace;
  return RunBestPairBaseline(graph, seeds, opts, &workspace, rng);
}

DcgsResult RunTopKActivationProbBaseline(const Graph& graph,
                                         const std::vector<NodeId>& seeds,
                                         const RuntimeOptions& opts,
                                         BaselineWorkspace* workspace,
                                         std::mt19937_64* rng) {
  DcgsResult result;
  const Real start = NowSeconds();

  BaselineWorkspace local_workspace;
  if (!workspace) workspace = &local_workspace;
  ResetWorkspace(graph, seeds, workspace);

  result.sample_cap_used = opts.sample_cap;
  if (opts.sample_cap == 0) {
    FinalizeSampleBudgetBaselineResult(opts, start, &result);
    return result;
  }

  RunSamplingPass(graph, seeds, opts.sample_cap, opts, true, workspace, rng);

  if (workspace->touched_activation_nodes.empty()) {
    result.u_hat = FallbackSingleNode(graph.num_nodes,
                                      workspace->seed_mark,
                                      workspace->seed_epoch);
    result.certified.rho_final = 0.0;
  } else {
    std::sort(workspace->touched_activation_nodes.begin(),
              workspace->touched_activation_nodes.end(),
              [&](NodeId lhs, NodeId rhs) {
                return ActivationCountLess(lhs, rhs, workspace->activation_counts);
              });

    if (workspace->touched_activation_nodes.size() == 1) {
      result.u_hat = {workspace->touched_activation_nodes[0]};
      result.certified.rho_final = 0.0;
    } else {
      // Nodes are ranked by estimated activation probability, but the selected
      // k is chosen by scanning all prefixes under the sampled rho objective.
      BuildPrefixWeights(workspace, opts.sample_cap, opts.tau);
      Real cumulative_weight = 0.0;
      Real best_rho = -1.0;
      std::size_t best_k = 2;
      for (std::size_t k = 1; k <= workspace->touched_activation_nodes.size(); ++k) {
        cumulative_weight += workspace->prefix_weight_starts[k];
        if (k < 2) continue;
        const Real rho = cumulative_weight / static_cast<Real>(k);
        if (rho > best_rho) {
          best_rho = rho;
          best_k = k;
        }
      }
      result.u_hat.assign(workspace->touched_activation_nodes.begin(),
                          workspace->touched_activation_nodes.begin() + best_k);
      result.certified.rho_final = std::max(static_cast<Real>(0.0), best_rho);
    }
  }

  FinalizeSampleBudgetBaselineResult(opts, start, &result);
  return result;
}

DcgsResult RunTopKActivationProbBaseline(const Graph& graph,
                                         const std::vector<NodeId>& seeds,
                                         const RuntimeOptions& opts,
                                         std::mt19937_64* rng) {
  BaselineWorkspace workspace;
  return RunTopKActivationProbBaseline(graph, seeds, opts, &workspace, rng);
}

DcgsResult RunTopKCoactivationDegreeBaseline(const Graph& graph,
                                            const std::vector<NodeId>& seeds,
                                            const RuntimeOptions& opts,
                                            BaselineWorkspace* workspace,
                                            std::mt19937_64* rng) {
  DcgsResult result;
  const Real start = NowSeconds();

  BaselineWorkspace local_workspace;
  if (!workspace) workspace = &local_workspace;
  ResetWorkspace(graph, seeds, workspace);

  result.sample_cap_used = opts.sample_cap;
  if (opts.sample_cap == 0) {
    FinalizeSampleBudgetBaselineResult(opts, start, &result);
    return result;
  }

  RunSamplingPass(graph, seeds, opts.sample_cap, opts, false, workspace, rng);
  BuildNodeScoresFromThresholdedPairs(workspace, opts.sample_cap, opts.tau);

  if (workspace->touched_activation_nodes.empty()) {
    result.u_hat = FallbackSingleNode(graph.num_nodes,
                                      workspace->seed_mark,
                                      workspace->seed_epoch);
    result.certified.rho_final = 0.0;
  } else {
    std::sort(workspace->touched_activation_nodes.begin(),
              workspace->touched_activation_nodes.end(),
              [&](NodeId lhs, NodeId rhs) {
                return NodeScoreLess(lhs, rhs, workspace->node_scores);
              });

    if (workspace->touched_activation_nodes.size() == 1) {
      result.u_hat = {workspace->touched_activation_nodes[0]};
      result.certified.rho_final = 0.0;
    } else {
      // As in the activation baseline, k is not fixed in advance. The method
      // ranks by coactivation degree and then scans all sorted prefixes.
      BuildPrefixWeights(workspace, opts.sample_cap, opts.tau);
      Real cumulative_weight = 0.0;
      Real best_rho = -1.0;
      std::size_t best_k = 2;
      for (std::size_t k = 1; k <= workspace->touched_activation_nodes.size(); ++k) {
        cumulative_weight += workspace->prefix_weight_starts[k];
        if (k < 2) continue;
        const Real rho = cumulative_weight / static_cast<Real>(k);
        if (rho > best_rho) {
          best_rho = rho;
          best_k = k;
        }
      }
      result.u_hat.assign(workspace->touched_activation_nodes.begin(),
                          workspace->touched_activation_nodes.begin() + best_k);
      result.certified.rho_final = std::max(static_cast<Real>(0.0), best_rho);
    }
  }

  FinalizeSampleBudgetBaselineResult(opts, start, &result);
  return result;
}

DcgsResult RunTopKCoactivationDegreeBaseline(const Graph& graph,
                                            const std::vector<NodeId>& seeds,
                                            const RuntimeOptions& opts,
                                            std::mt19937_64* rng) {
  BaselineWorkspace workspace;
  return RunTopKCoactivationDegreeBaseline(graph, seeds, opts, &workspace, rng);
}

DcgsResult RunGreedyAddBaseline(const Graph& graph,
                               const std::vector<NodeId>& seeds,
                               const RuntimeOptions& opts,
                               BaselineWorkspace* workspace,
                               std::mt19937_64* rng) {
  DcgsResult result;
  const Real start = NowSeconds();

  BaselineWorkspace local_workspace;
  if (!workspace) workspace = &local_workspace;
  ResetWorkspace(graph, seeds, workspace);

  result.sample_cap_used = opts.sample_cap;
  if (opts.sample_cap == 0) {
    FinalizeSampleBudgetBaselineResult(opts, start, &result);
    return result;
  }

  RunSamplingPass(graph, seeds, opts.sample_cap, opts, false, workspace, rng);

  NodeCount best_pair_u = 0;
  NodeCount best_pair_v = 0;
  Real best_pair_weight = 0.0;
  if (!BuildThresholdedAdjacency(workspace,
                                 opts.sample_cap,
                                 opts.tau,
                                 &best_pair_u,
                                 &best_pair_v,
                                 &best_pair_weight)) {
    result.u_hat = FallbackSingleNode(graph.num_nodes,
                                      workspace->seed_mark,
                                      workspace->seed_epoch);
    result.certified.rho_final = 0.0;
  } else {
    // Start from the best sampled pair and keep adding the node with maximum
    // positive marginal gain while rho strictly improves.
    std::vector<NodeCount> selected_order;
    selected_order.reserve(workspace->touched_activation_nodes.size());
    selected_order.push_back(best_pair_u);
    selected_order.push_back(best_pair_v);
    InitializeGreedyState(best_pair_u, best_pair_v, best_pair_weight, workspace);

    Real current_weight = best_pair_weight;
    NodeCount current_size = 2;
    Real current_rho = current_weight / static_cast<Real>(current_size);

    while (true) {
      NodeCount candidate = 0;
      Real gain = 0.0;
      if (!PopBestGain(&candidate, &gain, workspace)) break;
      const Real candidate_rho =
          (current_weight + gain) / static_cast<Real>(current_size + 1);
      if (!(candidate_rho > current_rho)) break;

      selected_order.push_back(candidate);
      AddGreedyNode(candidate, workspace);
      current_weight += gain;
      ++current_size;
      current_rho = candidate_rho;
    }

    result.u_hat.reserve(selected_order.size());
    for (NodeCount local_node : selected_order) {
      result.u_hat.push_back(
          workspace->touched_activation_nodes[static_cast<std::size_t>(local_node)]);
    }
    result.certified.rho_final = current_rho;
  }

  FinalizeSampleBudgetBaselineResult(opts, start, &result);
  return result;
}

DcgsResult RunGreedyAddBaseline(const Graph& graph,
                               const std::vector<NodeId>& seeds,
                               const RuntimeOptions& opts,
                               std::mt19937_64* rng) {
  BaselineWorkspace workspace;
  return RunGreedyAddBaseline(graph, seeds, opts, &workspace, rng);
}

DcgsResult RunPilotGreedyAblation(const Graph& graph,
                                  const std::vector<NodeId>& seeds,
                                  const std::vector<NodeId>& candidate_nodes,
                                  const RuntimeOptions& opts,
                                  BaselineWorkspace* workspace,
                                  std::mt19937_64* rng) {
  DcgsResult result;
  const Real start = NowSeconds();

  BaselineWorkspace local_workspace;
  if (!workspace) workspace = &local_workspace;
  ResetWorkspace(graph, seeds, workspace);

  result.sample_cap_used = opts.sample_cap;
  if (opts.sample_cap == 0) {
    FinalizeSampleBudgetBaselineResult(opts, start, &result);
    return result;
  }

  RunSamplingPassOnPilotCandidates(graph,
                                   seeds,
                                   opts.sample_cap,
                                   opts,
                                   candidate_nodes,
                                   workspace,
                                   rng);

  NodeCount best_pair_u = 0;
  NodeCount best_pair_v = 0;
  Real best_pair_weight = 0.0;
  if (!BuildThresholdedAdjacency(workspace,
                                 opts.sample_cap,
                                 opts.tau,
                                 &best_pair_u,
                                 &best_pair_v,
                                 &best_pair_weight)) {
    result.u_hat = FallbackCandidateSingleNode(
        candidate_nodes, workspace->seed_mark, workspace->seed_epoch);
    if (result.u_hat.empty()) {
      result.u_hat = FallbackSingleNode(
          graph.num_nodes, workspace->seed_mark, workspace->seed_epoch);
    }
    result.certified.rho_final = 0.0;
  } else {
    std::vector<NodeCount> selected_order;
    selected_order.reserve(workspace->touched_activation_nodes.size());
    selected_order.push_back(best_pair_u);
    selected_order.push_back(best_pair_v);
    InitializeGreedyState(best_pair_u, best_pair_v, best_pair_weight, workspace);

    Real current_weight = best_pair_weight;
    NodeCount current_size = 2;
    Real current_rho = current_weight / static_cast<Real>(current_size);

    while (true) {
      NodeCount candidate = 0;
      Real gain = 0.0;
      if (!PopBestGain(&candidate, &gain, workspace)) break;
      const Real candidate_rho =
          (current_weight + gain) / static_cast<Real>(current_size + 1);
      if (!(candidate_rho > current_rho)) break;

      selected_order.push_back(candidate);
      AddGreedyNode(candidate, workspace);
      current_weight += gain;
      ++current_size;
      current_rho = candidate_rho;
    }

    result.u_hat.reserve(selected_order.size());
    for (NodeCount local_node : selected_order) {
      result.u_hat.push_back(
          workspace->touched_activation_nodes[static_cast<std::size_t>(local_node)]);
    }
    result.certified.rho_final = current_rho;
  }

  FinalizeSampleBudgetBaselineResult(opts, start, &result);
  return result;
}

DcgsResult RunPilotGreedyAblation(const Graph& graph,
                                  const std::vector<NodeId>& seeds,
                                  const std::vector<NodeId>& candidate_nodes,
                                  const RuntimeOptions& opts,
                                  std::mt19937_64* rng) {
  BaselineWorkspace workspace;
  return RunPilotGreedyAblation(graph, seeds, candidate_nodes, opts, &workspace, rng);
}

DcgsResult RunPilotPeelingCapAblation(const Graph& graph,
                                      const std::vector<NodeId>& seeds,
                                      const TrackedGraph& tracked_graph,
                                      const RuntimeOptions& opts,
                                      BaselineWorkspace* workspace,
                                      std::mt19937_64* rng) {
  DcgsResult result;
  const Real start = NowSeconds();

  BaselineWorkspace local_workspace;
  if (!workspace) workspace = &local_workspace;
  ResetWorkspace(graph, seeds, workspace);

  result.sample_cap_used = opts.sample_cap;
  if (opts.sample_cap == 0) {
    FinalizeSampleBudgetBaselineResult(opts, start, &result);
    return result;
  }

  const std::size_t tracked_edge_count =
      static_cast<std::size_t>(TrackedEdgeCount(tracked_graph));
  if (tracked_edge_count == 0) {
    result.u_hat = FallbackCandidateSingleNode(
        tracked_graph.local_to_global, workspace->seed_mark, workspace->seed_epoch);
    if (result.u_hat.empty()) {
      result.u_hat = FallbackSingleNode(
          graph.num_nodes, workspace->seed_mark, workspace->seed_epoch);
    }
    result.certified.rho_final = 0.0;
    FinalizeSampleBudgetBaselineResult(opts, start, &result);
    return result;
  }

  MarkPilotCandidateSet(graph, tracked_graph.local_to_global, workspace);
  workspace->edge_activation_counts.assign(tracked_edge_count, 0);
  workspace->edge_weights.assign(tracked_edge_count, 0.0);

  for (SampleCount t = 0; t < opts.sample_cap; ++t) {
    const std::vector<NodeId>& activated =
        SimulateDiffusion(graph, seeds, opts, &workspace->diffusion_workspace, rng);

    for (NodeId node : activated) {
      if (workspace->seed_mark[node] == workspace->seed_epoch) continue;
      if (workspace->candidate_mark[node] != workspace->candidate_epoch) continue;
      if (workspace->active_mark[node]) continue;
      workspace->active_mark[node] = 1;
      workspace->touched_active_nodes.push_back(node);
    }

    for (std::size_t edge_index = 0; edge_index < tracked_edge_count; ++edge_index) {
      const NodeCount local_u = tracked_graph.edge_u[edge_index];
      const NodeCount local_v = tracked_graph.edge_v[edge_index];
      const NodeId global_u = tracked_graph.local_to_global[static_cast<std::size_t>(local_u)];
      const NodeId global_v = tracked_graph.local_to_global[static_cast<std::size_t>(local_v)];
      if (workspace->active_mark[global_u] && workspace->active_mark[global_v]) {
        ++workspace->edge_activation_counts[edge_index];
      }
    }

    for (NodeId node : workspace->touched_active_nodes) {
      workspace->active_mark[node] = 0;
    }
    workspace->touched_active_nodes.clear();
  }

  for (std::size_t edge_index = 0; edge_index < tracked_edge_count; ++edge_index) {
    const Real p_hat = static_cast<Real>(workspace->edge_activation_counts[edge_index]) /
                       static_cast<Real>(opts.sample_cap);
    workspace->edge_weights[edge_index] = (p_hat >= opts.tau) ? p_hat : 0.0;
  }

  const PeelingResult peel =
      WeightedPeeling(tracked_graph, workspace->edge_weights, &workspace->peeling_workspace);
  result.u_hat = peel.u_max;
  result.certified.rho_final = peel.rho_max;
  FinalizeSampleBudgetBaselineResult(opts, start, &result);
  return result;
}

DcgsResult RunPilotPeelingCapAblation(const Graph& graph,
                                      const std::vector<NodeId>& seeds,
                                      const TrackedGraph& tracked_graph,
                                      const RuntimeOptions& opts,
                                      std::mt19937_64* rng) {
  BaselineWorkspace workspace;
  return RunPilotPeelingCapAblation(
      graph, seeds, tracked_graph, opts, &workspace, rng);
}

DcgsResult RunLocalSearchBaseline(const Graph& graph,
                                  const std::vector<NodeId>& seeds,
                                  const RuntimeOptions& opts,
                                  BaselineWorkspace* workspace,
                                  std::mt19937_64* rng) {
  DcgsResult result;
  const Real start = NowSeconds();

  BaselineWorkspace local_workspace;
  if (!workspace) workspace = &local_workspace;
  ResetWorkspace(graph, seeds, workspace);

  result.sample_cap_used = opts.sample_cap;
  if (opts.sample_cap == 0) {
    FinalizeSampleBudgetBaselineResult(opts, start, &result);
    return result;
  }

  RunSamplingPass(graph, seeds, opts.sample_cap, opts, false, workspace, rng);

  NodeCount best_pair_u = 0;
  NodeCount best_pair_v = 0;
  Real best_pair_weight = 0.0;
  if (!BuildThresholdedAdjacency(workspace,
                                 opts.sample_cap,
                                 opts.tau,
                                 &best_pair_u,
                                 &best_pair_v,
                                 &best_pair_weight)) {
    result.u_hat = FallbackSingleNode(graph.num_nodes,
                                      workspace->seed_mark,
                                      workspace->seed_epoch);
    result.certified.rho_final = 0.0;
  } else {
    const NodeCount invalid_local = std::numeric_limits<NodeCount>::max();
    const std::size_t swap_limit =
        static_cast<std::size_t>(opts.local_search_swap_candidates);
    std::vector<NodeCount> selected_order;
    std::vector<NodeCount> selected_pos(
        workspace->touched_activation_nodes.size(), invalid_local);
    selected_order.reserve(workspace->touched_activation_nodes.size());
    workspace->candidate_rank_by_local.assign(
        workspace->touched_activation_nodes.size(), invalid_local);

    selected_order.push_back(best_pair_u);
    selected_pos[best_pair_u] = 0;
    selected_order.push_back(best_pair_v);
    selected_pos[best_pair_v] = 1;
    InitializeGreedyState(best_pair_u, best_pair_v, best_pair_weight, workspace);

    // local_search reuses the same thresholded local graph as greedy_add, but
    // continues with add/remove moves and a bounded swap neighborhood.
    Real current_weight = best_pair_weight;
    NodeCount current_size = 2;
    Real current_rho = current_weight / static_cast<Real>(current_size);

    while (true) {
      NodeCount candidate = 0;
      Real gain = 0.0;
      if (!PopBestGain(&candidate, &gain, workspace)) break;
      const Real candidate_rho =
          (current_weight + gain) / static_cast<Real>(current_size + 1);
      if (!(candidate_rho > current_rho)) break;

      selected_pos[candidate] = static_cast<NodeCount>(selected_order.size());
      selected_order.push_back(candidate);
      AddGreedyNode(candidate, workspace);
      current_weight += gain;
      ++current_size;
      current_rho = candidate_rho;
    }

    for (std::uint32_t iter = 0; iter < opts.local_search_max_iters; ++iter) {
      CollectTopAddCandidates(std::max<std::size_t>(1, swap_limit), workspace);

      NodeCount best_add_node = invalid_local;
      Real best_add_gain = 0.0;
      Real best_add_rho = current_rho;
      if (!workspace->add_candidates.empty()) {
        best_add_node = workspace->add_candidates.front().second;
        best_add_gain = workspace->add_candidates.front().first;
        const Real candidate_rho =
            (current_weight + best_add_gain) / static_cast<Real>(current_size + 1);
        if (candidate_rho > best_add_rho) {
          best_add_rho = candidate_rho;
        } else {
          best_add_node = invalid_local;
          best_add_gain = 0.0;
        }
      }

      NodeCount best_remove_node = invalid_local;
      Real best_remove_gain = 0.0;
      Real best_remove_rho = current_rho;
      workspace->remove_candidates.clear();
      if (current_size > 2) {
        for (NodeCount local_node : selected_order) {
          const Real remove_gain = workspace->gain_by_local[local_node];
          const Real candidate_rho =
              (current_weight - remove_gain) / static_cast<Real>(current_size - 1);
          if (candidate_rho > best_remove_rho) {
            best_remove_node = local_node;
            best_remove_gain = remove_gain;
            best_remove_rho = candidate_rho;
          }
          InsertSmallestRemoveCandidate(swap_limit, remove_gain, local_node, workspace);
        }
      }

      NodeCount best_swap_out = invalid_local;
      NodeCount best_swap_in = invalid_local;
      Real best_swap_remove_gain = 0.0;
      Real best_swap_add_gain_after_remove = 0.0;
      Real best_swap_rho = current_rho;
      if (swap_limit > 0 &&
          !workspace->add_candidates.empty() &&
          !workspace->remove_candidates.empty()) {
        workspace->touched_candidate_nodes.clear();
        for (std::size_t i = 0; i < workspace->add_candidates.size(); ++i) {
          const NodeCount local_node = workspace->add_candidates[i].second;
          workspace->candidate_rank_by_local[local_node] = static_cast<NodeCount>(i + 1);
          workspace->touched_candidate_nodes.push_back(local_node);
        }
        workspace->swap_edge_weights.assign(workspace->add_candidates.size(), 0.0);

        for (const auto& remove_entry : workspace->remove_candidates) {
          std::fill(workspace->swap_edge_weights.begin(),
                    workspace->swap_edge_weights.end(),
                    0.0);
          const NodeCount remove_node = remove_entry.second;
          for (EdgeOffset pos = workspace->adj_offsets[remove_node];
               pos < workspace->adj_offsets[static_cast<std::size_t>(remove_node) + 1];
               ++pos) {
            const NodeCount neighbor =
                workspace->adj_neighbors[static_cast<std::size_t>(pos)];
            const NodeCount rank = workspace->candidate_rank_by_local[neighbor];
            if (rank == invalid_local) continue;
            workspace->swap_edge_weights[static_cast<std::size_t>(rank - 1)] =
                workspace->adj_weights[static_cast<std::size_t>(pos)];
          }
          for (std::size_t i = 0; i < workspace->add_candidates.size(); ++i) {
            const NodeCount add_node = workspace->add_candidates[i].second;
            const Real add_gain_after_remove =
                workspace->add_candidates[i].first - workspace->swap_edge_weights[i];
            const Real candidate_rho =
                (current_weight - remove_entry.first + add_gain_after_remove) /
                static_cast<Real>(current_size);
            if (candidate_rho > best_swap_rho) {
              best_swap_rho = candidate_rho;
              best_swap_out = remove_node;
              best_swap_in = add_node;
              best_swap_remove_gain = remove_entry.first;
              best_swap_add_gain_after_remove = add_gain_after_remove;
            }
          }
        }
        for (NodeCount local_node : workspace->touched_candidate_nodes) {
          workspace->candidate_rank_by_local[local_node] = invalid_local;
        }
      }

      enum class MoveKind { kNone, kAdd, kRemove, kSwap };
      MoveKind move = MoveKind::kNone;
      Real best_next_rho = current_rho;
      if (best_add_node != invalid_local && best_add_rho > best_next_rho) {
        move = MoveKind::kAdd;
        best_next_rho = best_add_rho;
      }
      if (best_remove_node != invalid_local && best_remove_rho > best_next_rho) {
        move = MoveKind::kRemove;
        best_next_rho = best_remove_rho;
      }
      if (best_swap_in != invalid_local && best_swap_rho > best_next_rho) {
        move = MoveKind::kSwap;
        best_next_rho = best_swap_rho;
      }

      if (move == MoveKind::kNone) break;
      if (move == MoveKind::kAdd) {
        selected_pos[best_add_node] = static_cast<NodeCount>(selected_order.size());
        selected_order.push_back(best_add_node);
        AddGreedyNode(best_add_node, workspace);
        current_weight += best_add_gain;
        ++current_size;
        current_rho = best_add_rho;
        continue;
      }
      if (move == MoveKind::kRemove) {
        RemoveLocalNode(best_remove_node, workspace);
        current_weight -= best_remove_gain;
        --current_size;
        current_rho = best_remove_rho;

        const NodeCount remove_pos = selected_pos[best_remove_node];
        const NodeCount last_node = selected_order.back();
        selected_order[remove_pos] = last_node;
        selected_pos[last_node] = remove_pos;
        selected_order.pop_back();
        selected_pos[best_remove_node] = invalid_local;
        continue;
      }
      RemoveLocalNode(best_swap_out, workspace);
      current_weight -= best_swap_remove_gain;

      const NodeCount remove_pos = selected_pos[best_swap_out];
      const NodeCount last_node = selected_order.back();
      selected_order[remove_pos] = last_node;
      selected_pos[last_node] = remove_pos;
      selected_order.pop_back();
      selected_pos[best_swap_out] = invalid_local;

      selected_pos[best_swap_in] = static_cast<NodeCount>(selected_order.size());
      selected_order.push_back(best_swap_in);
      AddGreedyNode(best_swap_in, workspace);
      current_weight += best_swap_add_gain_after_remove;
      current_rho = best_swap_rho;
    }

    result.u_hat.reserve(selected_order.size());
    for (NodeCount local_node : selected_order) {
      result.u_hat.push_back(
          workspace->touched_activation_nodes[static_cast<std::size_t>(local_node)]);
    }
    result.certified.rho_final = current_rho;
  }

  FinalizeSampleBudgetBaselineResult(opts, start, &result);
  return result;
}

DcgsResult RunLocalSearchBaseline(const Graph& graph,
                                  const std::vector<NodeId>& seeds,
                                  const RuntimeOptions& opts,
                                  std::mt19937_64* rng) {
  BaselineWorkspace workspace;
  return RunLocalSearchBaseline(graph, seeds, opts, &workspace, rng);
}

DcgsResult RunStructurePeelingBaseline(const Graph& graph,
                                       const std::vector<NodeId>& seeds,
                                       const RuntimeOptions& opts,
                                       BaselineWorkspace* workspace,
                                       std::mt19937_64* /*rng*/) {
  DcgsResult result;
  const Real start = NowSeconds();

  BaselineWorkspace local_workspace;
  if (!workspace) workspace = &local_workspace;

  ResetWorkspace(graph, seeds, workspace);
  BuildUndirectedUniquePairKeysFromGraph(graph, opts, workspace);
  const TrackedGraph tracked =
      BuildTrackedGraphFromSortedPairKeys(graph, seeds, workspace->global_keys);

  workspace->edge_weights.assign(static_cast<std::size_t>(TrackedEdgeCount(tracked)), 1.0);
  const PeelingResult peel =
      WeightedPeeling(tracked, workspace->edge_weights, &workspace->peeling_workspace);
  result.u_hat = peel.u_max;
  result.certified.rho_final = peel.rho_max;
  FinalizeStructureBaselineResult(start, &result);
  return result;
}

DcgsResult RunStructurePeelingBaseline(const Graph& graph,
                                       const std::vector<NodeId>& seeds,
                                       const RuntimeOptions& opts,
                                       std::mt19937_64* rng) {
  BaselineWorkspace workspace;
  return RunStructurePeelingBaseline(graph, seeds, opts, &workspace, rng);
}

DcgsResult RunEdgeCoactPeelingBaseline(const Graph& graph,
                                       const std::vector<NodeId>& seeds,
                                       const RuntimeOptions& opts,
                                       BaselineWorkspace* workspace,
                                       std::mt19937_64* rng) {
  DcgsResult result;
  const Real start = NowSeconds();

  BaselineWorkspace local_workspace;
  if (!workspace) workspace = &local_workspace;
  ResetWorkspace(graph, seeds, workspace);

  result.sample_cap_used = opts.sample_cap;
  if (opts.sample_cap == 0) {
    FinalizeSampleBudgetBaselineResult(opts, start, &result);
    return result;
  }

  BuildUndirectedUniquePairKeysFromGraph(graph, opts, workspace);
  const TrackedGraph tracked =
      BuildTrackedGraphFromSortedPairKeys(graph, seeds, workspace->global_keys);

  const std::size_t tracked_edge_count =
      static_cast<std::size_t>(TrackedEdgeCount(tracked));
  workspace->edge_activation_counts.assign(tracked_edge_count, 0);
  workspace->edge_weights.assign(tracked_edge_count, 0.0);

  for (SampleCount t = 0; t < opts.sample_cap; ++t) {
    const std::vector<NodeId>& activated =
        SimulateDiffusion(graph, seeds, opts, &workspace->diffusion_workspace, rng);

    for (NodeId node : activated) {
      if (workspace->seed_mark[node] == workspace->seed_epoch) continue;
      if (workspace->active_mark[node]) continue;
      workspace->active_mark[node] = 1;
      workspace->touched_active_nodes.push_back(node);
    }

    for (std::size_t edge_index = 0; edge_index < tracked_edge_count; ++edge_index) {
      const NodeCount local_u = tracked.edge_u[edge_index];
      const NodeCount local_v = tracked.edge_v[edge_index];
      const NodeId global_u = tracked.local_to_global[static_cast<std::size_t>(local_u)];
      const NodeId global_v = tracked.local_to_global[static_cast<std::size_t>(local_v)];
      if (workspace->active_mark[global_u] && workspace->active_mark[global_v]) {
        ++workspace->edge_activation_counts[edge_index];
      }
    }

    for (NodeId node : workspace->touched_active_nodes) {
      workspace->active_mark[node] = 0;
    }
    workspace->touched_active_nodes.clear();
  }

  for (std::size_t edge_index = 0; edge_index < tracked_edge_count; ++edge_index) {
    const Real p_hat = static_cast<Real>(workspace->edge_activation_counts[edge_index]) /
                       static_cast<Real>(opts.sample_cap);
    workspace->edge_weights[edge_index] = (p_hat >= opts.tau) ? p_hat : 0.0;
  }

  const PeelingResult peel =
      WeightedPeeling(tracked, workspace->edge_weights, &workspace->peeling_workspace);
  result.u_hat = peel.u_max;
  result.certified.rho_final = peel.rho_max;
  FinalizeSampleBudgetBaselineResult(opts, start, &result);
  return result;
}

DcgsResult RunEdgeCoactPeelingBaseline(const Graph& graph,
                                       const std::vector<NodeId>& seeds,
                                       const RuntimeOptions& opts,
                                       std::mt19937_64* rng) {
  BaselineWorkspace workspace;
  return RunEdgeCoactPeelingBaseline(graph, seeds, opts, &workspace, rng);
}

}  // namespace dcgs
