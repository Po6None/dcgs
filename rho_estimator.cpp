#include "rho_estimator.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <utility>

namespace dcgs {

namespace {

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

// ChunkCapacityKeys converts the byte budget into a key capacity.
std::size_t ChunkCapacityKeys(std::uint64_t chunk_bytes) {
  const std::uint64_t bytes =
      std::max<std::uint64_t>(chunk_bytes, sizeof(std::uint64_t));
  const std::uint64_t keys = std::max<std::uint64_t>(1, bytes / sizeof(std::uint64_t));
  return static_cast<std::size_t>(
      std::min<std::uint64_t>(keys,
                              static_cast<std::uint64_t>(
                                  std::numeric_limits<std::size_t>::max())));
}

// PairBufferCapacityKeys limits the front buffer so one realization does not
// monopolize the full chunk budget before flushing.
std::size_t PairBufferCapacityKeys(std::size_t chunk_capacity_keys) {
  constexpr std::size_t kPairBufferBytes = 8ull * 1024ull * 1024ull;
  const std::size_t batch_keys =
      std::max<std::size_t>(1, kPairBufferBytes / sizeof(std::uint64_t));
  return std::min(chunk_capacity_keys, batch_keys);
}

// AdvanceCandidateEpoch avoids clearing a full O(n) membership array between
// repeated rho-evaluation calls.
void AdvanceCandidateEpoch(RhoEstimationWorkspace* workspace) {
  if (!workspace) return;
  if (workspace->candidate_epoch == std::numeric_limits<std::uint32_t>::max()) {
    std::fill(workspace->candidate_mark.begin(), workspace->candidate_mark.end(), 0);
    workspace->candidate_epoch = 1;
    return;
  }
  ++workspace->candidate_epoch;
}

// ResetWorkspace marks the candidate set U and clears the counted-run state for
// a fresh rho(U) estimation pass.
void ResetWorkspace(const Graph& graph,
                    const std::vector<NodeId>& nodes,
                    RhoEstimationWorkspace* workspace) {
  if (!workspace) return;

  const std::size_t n = static_cast<std::size_t>(graph.num_nodes);
  if (workspace->candidate_mark.size() != n) {
    workspace->candidate_mark.assign(n, 0);
    workspace->candidate_epoch = 1;
  } else {
    AdvanceCandidateEpoch(workspace);
  }
  for (NodeId node : nodes) {
    if (node < graph.num_nodes) {
      workspace->candidate_mark[node] = workspace->candidate_epoch;
    }
  }

  workspace->filtered_active_nodes.clear();
  workspace->pair_buffer.clear();
  workspace->chunk_buffer.clear();
  workspace->global_keys.clear();
  workspace->global_counts.clear();
  for (CountedRhoRun& run : workspace->run_levels) {
    run.keys.clear();
    run.counts.clear();
  }
  workspace->carry_keys.clear();
  workspace->carry_counts.clear();
  workspace->temp_keys.clear();
  workspace->temp_counts.clear();
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

// InsertCarryRun performs leveled merging to keep counted runs compact.
void InsertCarryRun(RhoEstimationWorkspace* workspace) {
  if (!workspace || workspace->carry_keys.empty()) return;

  std::size_t level = 0;
  while (true) {
    if (level == workspace->run_levels.size()) {
      workspace->run_levels.emplace_back();
    }
    CountedRhoRun& slot = workspace->run_levels[level];
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

// FlushChunk sorts the current chunk and inserts its counted run into the
// leveled merge structure.
void FlushChunk(RhoEstimationWorkspace* workspace) {
  if (!workspace || workspace->chunk_buffer.empty()) return;

  std::sort(workspace->chunk_buffer.begin(), workspace->chunk_buffer.end());
  CompressSortedKeysToCounts(workspace->chunk_buffer,
                             &workspace->carry_keys,
                             &workspace->carry_counts);
  workspace->chunk_buffer.clear();
  InsertCarryRun(workspace);
}

// FinalizeRuns collapses the leveled counted runs into one global counted list.
void FinalizeRuns(RhoEstimationWorkspace* workspace) {
  if (!workspace) return;

  workspace->global_keys.clear();
  workspace->global_counts.clear();
  for (CountedRhoRun& run : workspace->run_levels) {
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
void DrainPairBufferToChunk(RhoEstimationWorkspace* workspace,
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

}  // namespace

RhoEstimationResult EstimateRhoBySampling(const Graph& graph,
                                          const std::vector<NodeId>& seeds,
                                          const std::vector<NodeId>& nodes,
                                          SampleCount eval_cap,
                                          const RuntimeOptions& opts,
                                          RhoEstimationWorkspace* workspace,
                                          std::mt19937_64* rng) {
  // The estimator mirrors the sample-budget counting logic used by baselines:
  // sample realizations, count co-activated pairs inside U, threshold by tau,
  // then divide the surviving total weight by |U|.
  RhoEstimationResult result;
  const Real start = NowSeconds();

  if (eval_cap == 0 || nodes.size() < 2) {
    result.elapsed_sec = NowSeconds() - start;
    return result;
  }

  RhoEstimationWorkspace local_workspace;
  if (!workspace) workspace = &local_workspace;
  ResetWorkspace(graph, nodes, workspace);

  const std::size_t chunk_capacity_keys = ChunkCapacityKeys(opts.baseline_chunk_bytes);
  const std::size_t pair_buffer_capacity_keys = PairBufferCapacityKeys(chunk_capacity_keys);
  if (workspace->pair_buffer.capacity() < pair_buffer_capacity_keys) {
    workspace->pair_buffer.reserve(pair_buffer_capacity_keys);
  }

  for (SampleCount t = 0; t < eval_cap; ++t) {
    const std::vector<NodeId>& activated =
        SimulateDiffusion(graph, seeds, opts, &workspace->diffusion_workspace, rng);

    workspace->filtered_active_nodes.clear();
    workspace->filtered_active_nodes.reserve(activated.size());
    for (NodeId node : activated) {
      if (workspace->candidate_mark[node] == workspace->candidate_epoch) {
        workspace->filtered_active_nodes.push_back(node);
      }
    }

    const std::size_t k = workspace->filtered_active_nodes.size();
    for (std::size_t i = 0; i < k; ++i) {
      const NodeId u = workspace->filtered_active_nodes[i];
      for (std::size_t j = i + 1; j < k; ++j) {
        const NodeId v = workspace->filtered_active_nodes[j];
        workspace->pair_buffer.push_back(PackPairKey(u, v));
        if (workspace->pair_buffer.size() == pair_buffer_capacity_keys) {
          DrainPairBufferToChunk(workspace, chunk_capacity_keys);
        }
      }
    }
    DrainPairBufferToChunk(workspace, chunk_capacity_keys);
  }

  FlushChunk(workspace);
  FinalizeRuns(workspace);

  Real total_weight = 0.0;
  for (std::size_t i = 0; i < workspace->global_counts.size(); ++i) {
    const Real p_hat =
        static_cast<Real>(workspace->global_counts[i]) / static_cast<Real>(eval_cap);
    if (p_hat >= opts.tau) total_weight += p_hat;
  }

  result.rho_hat = total_weight / static_cast<Real>(nodes.size());
  result.eval_cap_used = eval_cap;
  result.elapsed_sec = NowSeconds() - start;
  return result;
}

RhoEstimationResult EstimateRhoBySampling(const Graph& graph,
                                          const std::vector<NodeId>& seeds,
                                          const std::vector<NodeId>& nodes,
                                          SampleCount eval_cap,
                                          const RuntimeOptions& opts,
                                          std::mt19937_64* rng) {
  // Convenience overload that allocates a temporary workspace.
  RhoEstimationWorkspace workspace;
  return EstimateRhoBySampling(graph, seeds, nodes, eval_cap, opts, &workspace, rng);
}

}  // namespace dcgs
