#include "diffusion_simulator.h"

namespace dcgs {

namespace {

DiffusionWorkspace* GetWorkspaceOrFallback(DiffusionWorkspace* workspace) {
  static thread_local DiffusionWorkspace fallback_workspace;
  return workspace ? workspace : &fallback_workspace;
}

// GetRngOrFallback keeps the convenience overloads deterministic even when the
// caller does not provide an RNG object.
std::mt19937_64* GetRngOrFallback(std::mt19937_64* rng) {
  static thread_local std::mt19937_64 fallback_rng(1);
  return rng ? rng : &fallback_rng;
}

// EnsureActiveWorkspaceCapacity prepares the generic active/frontier buffers.
void EnsureActiveWorkspaceCapacity(NodeCount num_nodes, DiffusionWorkspace* workspace) {
  workspace->active.resize(static_cast<std::size_t>(num_nodes), 0);
  workspace->frontier.clear();
  workspace->activated_nodes.clear();
}

// EnsureLtWorkspaceCapacity prepares the LT-specific scratch arrays.
void EnsureLtWorkspaceCapacity(NodeCount num_nodes, DiffusionWorkspace* workspace) {
  workspace->lt_seen.resize(static_cast<std::size_t>(num_nodes), 0);
  workspace->lt_sum_in.resize(static_cast<std::size_t>(num_nodes), 0.0);
  workspace->lt_theta.resize(static_cast<std::size_t>(num_nodes), 0.0);
  workspace->lt_touched_nodes.clear();
}

// ResetActiveFlags clears only the nodes touched by the current realization.
void ResetActiveFlags(DiffusionWorkspace* workspace) {
  for (NodeId node : workspace->activated_nodes) {
    workspace->active[node] = 0;
  }
}

// ResetLtScratch clears only the LT nodes that were touched in this realization.
void ResetLtScratch(DiffusionWorkspace* workspace) {
  for (NodeId node : workspace->lt_touched_nodes) {
    workspace->lt_seen[node] = 0;
    workspace->lt_sum_in[node] = 0.0;
  }
  workspace->lt_touched_nodes.clear();
}

}  // namespace

const std::vector<NodeId>& SimulateIC(const Graph& graph,
                                      const std::vector<NodeId>& seeds,
                                      DiffusionWorkspace* workspace,
                                      std::mt19937_64* rng) {
  // The IC implementation pays only for nodes reached in this realization.
  workspace = GetWorkspaceOrFallback(workspace);
  rng = GetRngOrFallback(rng);
  EnsureActiveWorkspaceCapacity(graph.num_nodes, workspace);

  for (NodeId seed : seeds) {
    if (seed >= graph.num_nodes || workspace->active[seed]) continue;
    workspace->active[seed] = 1;
    workspace->frontier.push_back(seed);
    workspace->activated_nodes.push_back(seed);
  }

  std::uniform_real_distribution<Real> unif(0.0, 1.0);
  std::size_t frontier_head = 0;
  while (frontier_head < workspace->frontier.size()) {
    const NodeId u = workspace->frontier[frontier_head++];
    for (EdgeOffset edge_index = graph.out_offsets[u];
         edge_index < graph.out_offsets[static_cast<std::size_t>(u) + 1];
         ++edge_index) {
      const NodeId v = graph.out_neighbors[static_cast<std::size_t>(edge_index)];
      if (workspace->active[v]) continue;
      const Real p_uv = graph.out_values[static_cast<std::size_t>(edge_index)];
      if (unif(*rng) <= p_uv) {
        workspace->active[v] = 1;
        workspace->frontier.push_back(v);
        workspace->activated_nodes.push_back(v);
      }
    }
  }

  ResetActiveFlags(workspace);
  return workspace->activated_nodes;
}

const std::vector<NodeId>& SimulateLT(const Graph& graph,
                                      const std::vector<NodeId>& seeds,
                                      DiffusionWorkspace* workspace,
                                      std::mt19937_64* rng) {
  // LT uses lazy threshold generation: a node receives theta only when it is
  // first touched by incoming influence.
  workspace = GetWorkspaceOrFallback(workspace);
  rng = GetRngOrFallback(rng);
  EnsureActiveWorkspaceCapacity(graph.num_nodes, workspace);
  EnsureLtWorkspaceCapacity(graph.num_nodes, workspace);

  for (NodeId seed : seeds) {
    if (seed >= graph.num_nodes || workspace->active[seed]) continue;
    workspace->active[seed] = 1;
    workspace->frontier.push_back(seed);
    workspace->activated_nodes.push_back(seed);
  }

  std::uniform_real_distribution<Real> unif(0.0, 1.0);
  std::size_t frontier_head = 0;
  while (frontier_head < workspace->frontier.size()) {
    const NodeId u = workspace->frontier[frontier_head++];
    for (EdgeOffset edge_index = graph.out_offsets[u];
         edge_index < graph.out_offsets[static_cast<std::size_t>(u) + 1];
         ++edge_index) {
      const NodeId v = graph.out_neighbors[static_cast<std::size_t>(edge_index)];
      if (workspace->active[v]) continue;

      if (!workspace->lt_seen[v]) {
        workspace->lt_seen[v] = 1;
        workspace->lt_sum_in[v] = 0.0;
        workspace->lt_theta[v] = unif(*rng);
        workspace->lt_touched_nodes.push_back(v);
      }

      workspace->lt_sum_in[v] += graph.out_values[static_cast<std::size_t>(edge_index)];
      if (workspace->lt_sum_in[v] >= workspace->lt_theta[v]) {
        workspace->active[v] = 1;
        workspace->frontier.push_back(v);
        workspace->activated_nodes.push_back(v);
      }
    }
  }

  ResetActiveFlags(workspace);
  ResetLtScratch(workspace);
  return workspace->activated_nodes;
}

const std::vector<NodeId>& SimulateDiffusion(const Graph& graph,
                                             const std::vector<NodeId>& seeds,
                                             const RuntimeOptions& opts,
                                             DiffusionWorkspace* workspace,
                                             std::mt19937_64* rng) {
  // The dispatch keeps the caller-side code independent of the diffusion model.
  if (opts.model == DiffusionModel::kLT) {
    return SimulateLT(graph, seeds, workspace, rng);
  }
  return SimulateIC(graph, seeds, workspace, rng);
}

std::vector<NodeId> SimulateIC(const Graph& graph,
                               const std::vector<NodeId>& seeds,
                               std::mt19937_64* rng) {
  // Convenience overload that allocates a temporary workspace.
  DiffusionWorkspace workspace;
  const std::vector<NodeId>& activated = SimulateIC(graph, seeds, &workspace, rng);
  return activated;
}

std::vector<NodeId> SimulateLT(const Graph& graph,
                               const std::vector<NodeId>& seeds,
                               std::mt19937_64* rng) {
  // Convenience overload that allocates a temporary workspace.
  DiffusionWorkspace workspace;
  const std::vector<NodeId>& activated = SimulateLT(graph, seeds, &workspace, rng);
  return activated;
}

std::vector<NodeId> SimulateDiffusion(const Graph& graph,
                                      const std::vector<NodeId>& seeds,
                                      const RuntimeOptions& opts,
                                      std::mt19937_64* rng) {
  // Convenience overload that allocates a temporary workspace.
  DiffusionWorkspace workspace;
  const std::vector<NodeId>& activated = SimulateDiffusion(graph, seeds, opts, &workspace, rng);
  return activated;
}

}  // namespace dcgs
