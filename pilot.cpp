#include "pilot.h"

#include "progress.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <vector>

namespace dcgs {

namespace {

Real NowSeconds() {
  using clock = std::chrono::steady_clock;
  const auto now = clock::now().time_since_epoch();
  return std::chrono::duration<Real>(now).count();
}

SampleCount Comb2(SampleCount x) {
  if (x < 2) return 0;
  return x * (x - 1) / 2;
}

std::uint64_t PackPairKey(NodeId u, NodeId v) {
  if (u > v) std::swap(u, v);
  return (static_cast<std::uint64_t>(u) << 32) | static_cast<std::uint32_t>(v);
}

// The pilot stage emits many repeated co-activated pairs across realizations.
// Keeping them in a global hash table is exact but expensive at scale, so we
// instead accumulate sorted runs under a fixed memory budget.
void SortAndUnique(std::vector<std::uint64_t>* keys) {
  if (!keys || keys->empty()) return;
  std::sort(keys->begin(), keys->end());
  keys->erase(std::unique(keys->begin(), keys->end()), keys->end());
}

void MergeSortedUnique(const std::vector<std::uint64_t>& lhs,
                       const std::vector<std::uint64_t>& rhs,
                       std::vector<std::uint64_t>* out) {
  if (!out) return;
  out->clear();
  out->reserve(lhs.size() + rhs.size());

  std::size_t i = 0;
  std::size_t j = 0;
  while (i < lhs.size() && j < rhs.size()) {
    if (lhs[i] < rhs[j]) {
      out->push_back(lhs[i++]);
    } else if (rhs[j] < lhs[i]) {
      out->push_back(rhs[j++]);
    } else {
      out->push_back(lhs[i]);
      ++i;
      ++j;
    }
  }
  out->insert(out->end(), lhs.begin() + i, lhs.end());
  out->insert(out->end(), rhs.begin() + j, rhs.end());
}

std::size_t ChunkCapacityKeys(const RuntimeOptions& opts) {
  const std::uint64_t bytes = std::max<std::uint64_t>(opts.pilot_chunk_bytes, sizeof(std::uint64_t));
  const std::uint64_t keys = std::max<std::uint64_t>(1, bytes / sizeof(std::uint64_t));
  return static_cast<std::size_t>(
      std::min<std::uint64_t>(keys, static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())));
}

// Pair generation is bursty: one realization may emit a large quadratic batch.
// A smaller front buffer keeps those writes contiguous, and the chunk buffer
// enforces the user-visible memory budget before global deduplication.
std::size_t PairBufferCapacityKeys(std::size_t chunk_capacity_keys) {
  constexpr std::size_t kPairBufferBytes = 8ull * 1024ull * 1024ull;
  const std::size_t batch_keys = std::max<std::size_t>(1, kPairBufferBytes / sizeof(std::uint64_t));
  return std::min(chunk_capacity_keys, batch_keys);
}

void ResetWorkspace(const Graph& graph,
                    const std::vector<NodeId>& seeds,
                    PilotWorkspace* workspace) {
  if (!workspace) return;

  // Seeds are marked once per pilot invocation and then reused by every
  // realization. This avoids repeated seed filtering work inside the inner
  // diffusion loop.
  const std::size_t n = static_cast<std::size_t>(graph.num_nodes);
  if (workspace->is_seed.size() != n) {
    workspace->is_seed.assign(n, 0);
  } else {
    std::fill(workspace->is_seed.begin(), workspace->is_seed.end(), 0);
  }
  for (NodeId s : seeds) workspace->is_seed[s] = 1;

  workspace->filtered_active_nodes.clear();
  workspace->pair_buffer.clear();
  workspace->chunk_buffer.clear();
  workspace->global_unique_keys.clear();
  workspace->merge_buffer.clear();
  if (workspace->tracked_node_seen.size() != n) {
    workspace->tracked_node_seen.assign(n, 0);
  } else {
    std::fill(workspace->tracked_node_seen.begin(),
              workspace->tracked_node_seen.end(),
              0);
  }
  workspace->tracked_nodes_count = 0;
}

void FlushChunk(PilotWorkspace* workspace) {
  if (!workspace || workspace->chunk_buffer.empty()) return;

  // Each flushed chunk becomes globally sorted and unique before it is merged
  // into the accumulated exact result. The code keeps only unique keys alive
  // after a flush instead of preserving every emitted pair instance.
  SortAndUnique(&workspace->chunk_buffer);
  MergeSortedUnique(workspace->global_unique_keys, workspace->chunk_buffer, &workspace->merge_buffer);
  workspace->global_unique_keys.swap(workspace->merge_buffer);
  workspace->chunk_buffer.clear();
}

void DrainPairBufferToChunk(PilotWorkspace* workspace, std::size_t chunk_capacity_keys) {
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

void MarkTrackedNode(NodeId u, PilotWorkspace* workspace) {
  if (!workspace || u >= workspace->tracked_node_seen.size()) return;
  if (workspace->tracked_node_seen[u]) return;
  workspace->tracked_node_seen[u] = 1;
  ++workspace->tracked_nodes_count;
}

SampleCount CurrentTrackedEdgeCountForProgress(const PilotWorkspace& workspace) {
  // global_unique_keys is always sorted+unique. chunk/pair buffers are pending
  // and need a local dedup to compute an exact live edge count for progress.
  if (workspace.chunk_buffer.empty() && workspace.pair_buffer.empty()) {
    return static_cast<SampleCount>(workspace.global_unique_keys.size());
  }

  std::vector<std::uint64_t> pending;
  pending.reserve(workspace.chunk_buffer.size() + workspace.pair_buffer.size());
  pending.insert(pending.end(), workspace.chunk_buffer.begin(), workspace.chunk_buffer.end());
  pending.insert(pending.end(), workspace.pair_buffer.begin(), workspace.pair_buffer.end());
  SortAndUnique(&pending);

  std::size_t i = 0;
  std::size_t j = 0;
  SampleCount count = 0;
  while (i < workspace.global_unique_keys.size() && j < pending.size()) {
    if (workspace.global_unique_keys[i] < pending[j]) {
      ++count;
      ++i;
    } else if (pending[j] < workspace.global_unique_keys[i]) {
      ++count;
      ++j;
    } else {
      ++count;
      ++i;
      ++j;
    }
  }
  count += static_cast<SampleCount>(workspace.global_unique_keys.size() - i);
  count += static_cast<SampleCount>(pending.size() - j);
  return count;
}

}  // namespace

PilotResult PilotStage(const Graph& graph,
                       const std::vector<NodeId>& seeds,
                       Real delta0,
                       const RuntimeOptions& opts,
                       PilotWorkspace* workspace,
                       std::mt19937_64* rng) {
  PilotResult result;
  const Real start = NowSeconds();

  PilotWorkspace local_workspace;
  if (!workspace) workspace = &local_workspace;
  ResetWorkspace(graph, seeds, workspace);

  const SampleCount n0 =
      static_cast<SampleCount>(graph.num_nodes) - static_cast<SampleCount>(seeds.size());
  const SampleCount pair_count = Comb2(n0);
  if (pair_count <= 0) {
    result.t0 = 0;
    result.elapsed_sec = NowSeconds() - start;
    PrintProgressLine(opts,
                      "pilot",
                      "skipped T0=0 because fewer than two non-seed nodes remain");
    return result;
  }

  // The pilot sample size follows the paper's union-bound guarantee over all
  // unordered non-seed pairs. The caller passes delta0 explicitly so that the
  // pilot stage does not silently assume a particular failure-budget split.
  const Real numerator = std::log(static_cast<Real>(pair_count) / delta0);
  const Real denominator = std::log(1.0 / (1.0 - opts.tau));
  result.t0 = static_cast<SampleCount>(std::ceil(numerator / denominator));
  PrintProgressLine(opts,
                    "pilot",
                    "start T0=" + std::to_string(result.t0) +
                        " tracked_nodes=0 tracked_edges=0");

  const std::size_t chunk_capacity_keys = ChunkCapacityKeys(opts);
  const std::size_t pair_buffer_capacity_keys = PairBufferCapacityKeys(chunk_capacity_keys);
  if (workspace->pair_buffer.capacity() < pair_buffer_capacity_keys) {
    workspace->pair_buffer.reserve(pair_buffer_capacity_keys);
  }
  SampleCount next_progress = ProgressReportInterval(result.t0);

  for (SampleCount t = 0; t < result.t0; ++t) {
    const std::vector<NodeId>& activated =
        SimulateDiffusion(graph, seeds, opts, &workspace->diffusion_workspace, rng);
    workspace->filtered_active_nodes.clear();
    workspace->filtered_active_nodes.reserve(activated.size());
    for (NodeId u : activated) {
      if (!workspace->is_seed[u]) workspace->filtered_active_nodes.push_back(u);
    }

    const std::size_t k = workspace->filtered_active_nodes.size();
    // Enumerating all co-activated pairs is unavoidable if we want the exact
    // tracked set for this realization. The surrounding buffering logic keeps
    // that unavoidable O(k^2) emission from turning into random global writes.
    for (std::size_t i = 0; i < k; ++i) {
        const NodeId u = workspace->filtered_active_nodes[i];
        for (std::size_t j = i + 1; j < k; ++j) {
          const NodeId v = workspace->filtered_active_nodes[j];
          workspace->pair_buffer.push_back(PackPairKey(u, v));
          MarkTrackedNode(u, workspace);
          MarkTrackedNode(v, workspace);
          if (workspace->pair_buffer.size() == pair_buffer_capacity_keys) {
            DrainPairBufferToChunk(workspace, chunk_capacity_keys);
          }
      }
    }
    DrainPairBufferToChunk(workspace, chunk_capacity_keys);

    const SampleCount current = t + 1;
    if (ShouldReportProgress(current, result.t0, &next_progress)) {
      const SampleCount tracked_edges = CurrentTrackedEdgeCountForProgress(*workspace);
      PrintProgressLine(opts,
                        "pilot",
                        "sample " + FormatProgressFraction(current, result.t0) +
                            " tracked_nodes=" +
                            std::to_string(workspace->tracked_nodes_count) +
                            " tracked_edges=" + std::to_string(tracked_edges));
    }
  }

  FlushChunk(workspace);
  // The pilot stage now hands downstream modules a canonical tracked graph
  // instead of an intermediate EdgePair list. This removes a long-lived
  // duplicate representation of E_hat_tau from the pipeline.
  result.tracked_graph =
      BuildTrackedGraphFromSortedPairKeys(graph, seeds, workspace->global_unique_keys);

  result.elapsed_sec = NowSeconds() - start;
  PrintProgressLine(opts,
                    "pilot",
                    "done T0=" + std::to_string(result.t0) +
                        " tracked_nodes=" +
                        std::to_string(TrackedNodeCount(result.tracked_graph)) +
                        " tracked_edges=" +
                        std::to_string(TrackedEdgeCount(result.tracked_graph)) +
                        " elapsed_sec=" + std::to_string(result.elapsed_sec));
  return result;
}

PilotResult PilotStage(const Graph& graph,
                       const std::vector<NodeId>& seeds,
                       Real delta0,
                       const RuntimeOptions& opts,
                       std::mt19937_64* rng) {
  PilotWorkspace workspace;
  return PilotStage(graph, seeds, delta0, opts, &workspace, rng);
}

}  // namespace dcgs
