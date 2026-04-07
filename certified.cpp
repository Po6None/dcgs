#include "certified.h"

#include "progress.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <utility>

namespace dcgs {

namespace {

// NowSeconds provides monotonic wall-clock timing for stage-level profiling.
Real NowSeconds() {
  using clock = std::chrono::steady_clock;
  const auto now = clock::now().time_since_epoch();
  return std::chrono::duration<Real>(now).count();
}

// ClampCeilToSampleCount safely converts a long-double upper bound into the
// unsigned sample-count type used throughout the project.
SampleCount ClampCeilToSampleCount(long double value) {
  if (!(value > 0.0L)) return 0;
  const long double ceil_value = std::ceil(value);
  const long double limit =
      static_cast<long double>(std::numeric_limits<SampleCount>::max());
  if (ceil_value >= limit) return std::numeric_limits<SampleCount>::max();
  return static_cast<SampleCount>(ceil_value);
}

// InitialCheckpoint computes the first certified checkpoint time.
SampleCount InitialCheckpoint(SampleCount tracked_edge_count, Real delta1) {
  const long double numerator =
      2.0L * std::log((6.0L * static_cast<long double>(tracked_edge_count)) /
                      static_cast<long double>(delta1));
  return std::max<SampleCount>(1, ClampCeilToSampleCount(numerator));
}

// MaxCheckpointCount computes how many geometric checkpoints can occur before
// the certified cap is reached.
std::uint32_t MaxCheckpointCount(SampleCount first_checkpoint_t, SampleCount t_max) {
  if (first_checkpoint_t == 0 || first_checkpoint_t > t_max) return 0;
  std::uint32_t count = 0;
  SampleCount current = first_checkpoint_t;
  while (current <= t_max) {
    ++count;
    if (current > t_max / 2) break;
    current *= 2;
  }
  return count;
}

// This is the paper's worst-case sample cap for the certified stage. The cap is
// intentionally conservative: it only prevents unbounded sampling and is not
// the condition that triggers early stopping.
SampleCount ComputeTMax(SampleCount n0,
                        SampleCount tracked_edge_count,
                        const RuntimeOptions& opts,
                        Real delta1) {
  if (tracked_edge_count == 0) return 0;

  const long double eps_ld = static_cast<long double>(opts.eps);
  const long double tau_ld = static_cast<long double>(opts.tau);
  const long double gamma_ld = static_cast<long double>(opts.gamma);
  const long double tracked_edge_count_ld =
      static_cast<long double>(tracked_edge_count);
  const long double n0_minus_1_ld =
      (n0 > 0) ? static_cast<long double>(n0 - 1) : 0.0L;

  const long double log_term =
      0.5L * std::log((6.0L * tracked_edge_count_ld) /
                      static_cast<long double>(delta1));
  const long double gamma_term = 1.0L / (gamma_ld * gamma_ld);
  const long double size_term =
      std::min(n0_minus_1_ld * n0_minus_1_ld,
               tracked_edge_count_ld * tracked_edge_count_ld);
  long double solver_term = 0.0L;
  if (opts.subproblem_solver == SubproblemSolverKind::kPeeling) {
    solver_term =
        (9.0L * (2.0L + eps_ld) * (2.0L + eps_ld) /
         (tau_ld * tau_ld * eps_ld * eps_ld)) *
        size_term;
  } else {
    solver_term =
        (((1.0L + eps_ld) * (1.0L + eps_ld)) /
         (tau_ld * tau_ld * eps_ld * eps_ld)) *
        size_term;
  }
  return std::max<SampleCount>(
      1, ClampCeilToSampleCount(log_term * std::max(gamma_term, solver_term)));
}

// ShouldEarlyStop applies the solver-specific multiplicative certificate.
bool ShouldEarlyStop(Real rho_upper,
                     Real rho_lower,
                     const RuntimeOptions& opts) {
  const long double upper = static_cast<long double>(rho_upper);
  const long double lower = static_cast<long double>(rho_lower);
  const long double eps = static_cast<long double>(opts.eps);
  if (opts.subproblem_solver == SubproblemSolverKind::kPeeling) {
    return 2.0L * upper <= (2.0L + eps) * lower;
  }
  return upper <= (1.0L + eps) * lower;
}

Real ComputeCheckpointRadius(SampleCount t,
                             std::uint32_t checkpoint_index,
                             SampleCount tracked_edge_count,
                             Real delta1) {
  // The confidence radius shrinks with both the current checkpoint index and
  // the sample count t. It is shared by the lower/upper confidence bounds used
  // in the early-stop certificate.
  const long double t_ld = static_cast<long double>(t);
  const long double checkpoint_factor =
      static_cast<long double>(checkpoint_index + 1) *
      static_cast<long double>(checkpoint_index + 2);
  const long double log_term =
      std::log((3.0L * static_cast<long double>(tracked_edge_count) *
                checkpoint_factor) /
               static_cast<long double>(delta1));
  const long double radius = std::sqrt(log_term / (2.0L * t_ld));
  return static_cast<Real>(radius);
}

// EarlyStopTargetFactor returns the solver-specific multiplicative threshold
// used by the checkpoint certificate.
Real EarlyStopTargetFactor(const RuntimeOptions& opts) {
  if (opts.subproblem_solver == SubproblemSolverKind::kPeeling) {
    return 2.0 + opts.eps;
  }
  return 1.0 + opts.eps;
}

// CurrentApproximationFactor reports the real-time factor implied by the
// current upper/lower certificate pair. Early stopping triggers once this
// factor drops below the corresponding target threshold.
Real CurrentApproximationFactor(Real rho_upper,
                                Real rho_lower,
                                const RuntimeOptions& opts) {
  if (rho_lower <= 0.0) {
    if (rho_upper <= 0.0) return 0.0;
    return std::numeric_limits<Real>::infinity();
  }
  if (opts.subproblem_solver == SubproblemSolverKind::kPeeling) {
    return 2.0 * rho_upper / rho_lower;
  }
  return rho_upper / rho_lower;
}

void ResetWorkspace(const Graph& graph,
                    const TrackedGraph& tracked_graph,
                    CertifiedWorkspace* workspace) {
  if (!workspace) return;
  workspace->tracked_graph = &tracked_graph;

  // tracked_node_mask and active_mark live in global node space because the
  // diffusion simulator also returns global node ids. This keeps the hot path
  // free of global-to-local lookups during every realization.
  const std::size_t num_nodes = static_cast<std::size_t>(graph.num_nodes);
  workspace->tracked_node_mask.assign(num_nodes, 0);
  workspace->active_mark.assign(num_nodes, 0);

  for (NodeId node : tracked_graph.local_to_global) {
    workspace->tracked_node_mask[node] = 1;
  }

  const std::size_t num_edges = static_cast<std::size_t>(TrackedEdgeCount(tracked_graph));
  workspace->counters.assign(num_edges, 0);
  workspace->weight_buffer.assign(num_edges, 0.0);

  const std::size_t num_local_nodes = tracked_graph.local_to_global.size();
  workspace->local_membership.assign(num_local_nodes, 0);
  workspace->touched_local_nodes.clear();
  workspace->touched_local_nodes.reserve(num_local_nodes);
}

void MarkActivatedTrackedNodes(const std::vector<NodeId>& activated,
                               CertifiedWorkspace* workspace) {
  // Only tracked endpoints matter for certified counters. Nodes outside the
  // tracked graph are ignored immediately so they do not affect later scans.
  for (NodeId node : activated) {
    if (workspace->tracked_node_mask[node]) workspace->active_mark[node] = 1;
  }
}

void ClearActivatedTrackedNodes(const std::vector<NodeId>& activated,
                                CertifiedWorkspace* workspace) {
  // The cleanup cost is proportional to the current realization's activated
  // set, not to the total graph size.
  for (NodeId node : activated) {
    if (workspace->tracked_node_mask[node]) workspace->active_mark[node] = 0;
  }
}

void UpdateCounters(CertifiedWorkspace* workspace) {
  const TrackedGraph& tracked_graph = *workspace->tracked_graph;
  // We scan only the tracked pairs discovered by the pilot stage. This avoids
  // rebuilding pair sets from the activated nodes during certified sampling.
  for (EdgeIndex edge_index = 0;
       edge_index < static_cast<EdgeIndex>(tracked_graph.edge_u.size());
       ++edge_index) {
    const NodeId u = tracked_graph.local_to_global[
        static_cast<std::size_t>(tracked_graph.edge_u[static_cast<std::size_t>(edge_index)])];
    const NodeId v = tracked_graph.local_to_global[
        static_cast<std::size_t>(tracked_graph.edge_v[static_cast<std::size_t>(edge_index)])];
    if (workspace->active_mark[u] && workspace->active_mark[v]) {
      ++workspace->counters[static_cast<std::size_t>(edge_index)];
    }
  }
}

void BuildTildeWeights(SampleCount t,
                       const RuntimeOptions& opts,
                       CertifiedWorkspace* workspace) {
  // weight_buffer is reused across checkpoint solves. We first materialize the
  // thresholded empirical weights \tilde{W}, solve once, then overwrite the
  // same buffer with the optimistic upper weights W^+.
  const Real inv_t = 1.0 / static_cast<Real>(t);
  for (std::size_t edge_index = 0; edge_index < workspace->counters.size(); ++edge_index) {
    const Real p_hat =
        static_cast<Real>(workspace->counters[edge_index]) * inv_t;
    workspace->weight_buffer[edge_index] = (p_hat >= opts.tau) ? p_hat : 0.0;
  }
}

void BuildUpperWeights(SampleCount t,
                       Real radius,
                       const RuntimeOptions& opts,
                       CertifiedWorkspace* workspace) {
  // The upper-confidence graph is only needed for the second solve at a
  // checkpoint, so there is no reason to keep a separate permanent array.
  const Real inv_t = 1.0 / static_cast<Real>(t);
  for (std::size_t edge_index = 0; edge_index < workspace->counters.size(); ++edge_index) {
    const Real p_hat =
        static_cast<Real>(workspace->counters[edge_index]) * inv_t;
    const Real upper = std::min(static_cast<Real>(1.0), p_hat + radius);
    workspace->weight_buffer[edge_index] = (upper >= opts.tau) ? upper : 0.0;
  }
}

void BuildFinalWeights(SampleCount t,
                       const RuntimeOptions& opts,
                       CertifiedWorkspace* workspace) {
  BuildTildeWeights(t, opts, workspace);
}
}

void FillFinalExactTrace(CertifiedFinalExactTrace* trace,
                         SampleCount t,
                         const PeelingResult& peel,
                         const ExactSolverCallStats& exact_stats,
                         Real opt_sec) {
  // The final exact trace is optional and only populated when an exact solve is
  // actually executed at the end of the certified stage.
  if (!trace) return;
  *trace = CertifiedFinalExactTrace{};
  if (exact_stats.solve_calls == 0) return;

  trace->used = true;
  trace->t = t;
  trace->u_hat_size = static_cast<NodeCount>(peel.u_max.size());
  trace->rho = peel.rho_max;
  trace->opt_sec = opt_sec;
  trace->exact_solve_calls = exact_stats.solve_calls;
  trace->exact_flow_calls = exact_stats.flow_calls;
  trace->exact_binary_search_iters = exact_stats.binary_search_iters;
  trace->exact_topology_rebuilds = exact_stats.topology_rebuilt ? 1u : 0u;
  trace->exact_capacity_scale_last = exact_stats.capacity_scale_used;
  trace->exact_capacity_scale_max = exact_stats.capacity_scale_used;
  trace->exact_solver_sec = exact_stats.solver_sec;
  trace->exact_avg_flow_calls =
      static_cast<Real>(exact_stats.flow_calls) /
      static_cast<Real>(exact_stats.solve_calls);
  trace->exact_avg_binary_search_iters =
      static_cast<Real>(exact_stats.binary_search_iters) /
      static_cast<Real>(exact_stats.solve_calls);
  trace->exact_avg_solver_time_sec =
      exact_stats.solver_sec / static_cast<Real>(exact_stats.solve_calls);
}

// SolveSubproblem dispatches the tracked weighted graph to either peeling or
// exact min-cut and accumulates exact-specific statistics when relevant.
PeelingResult SolveSubproblem(const CertifiedWorkspace* workspace,
                              const RuntimeOptions& opts,
                              const std::vector<Real>& weights,
                              CertifiedResult* result,
                              ExactSolverCallStats* call_stats,
                              PeelingWorkspace* peeling_workspace,
                              ExactSolverWorkspace* exact_workspace) {
  if (call_stats) *call_stats = ExactSolverCallStats{};
  if (!workspace) return PeelingResult{};
  if (opts.subproblem_solver == SubproblemSolverKind::kExact) {
    ExactSolverCallStats exact_stats;
    const PeelingResult exact_result =
        ExactMinCutSolver(*workspace->tracked_graph, weights, exact_workspace,
                          &exact_stats);
    if (call_stats) *call_stats = exact_stats;
    if (result) {
      result->exact_solve_calls += exact_stats.solve_calls;
      result->exact_flow_calls += exact_stats.flow_calls;
      result->exact_binary_search_iters += exact_stats.binary_search_iters;
      result->exact_topology_rebuilds += exact_stats.topology_rebuilt ? 1u : 0u;
      result->exact_capacity_scale_last = exact_stats.capacity_scale_used;
      result->exact_capacity_scale_max =
          std::max(result->exact_capacity_scale_max,
                   exact_stats.capacity_scale_used);
      result->exact_solver_sec += exact_stats.solver_sec;
    }
    return exact_result;
  }
  return WeightedPeeling(*workspace->tracked_graph, weights, peeling_workspace);
}

Real ComputeDensity(const TrackedGraph& prepared,
                    const std::vector<NodeId>& nodes,
                    CertifiedWorkspace* workspace,
                    const std::vector<Real>* weights,
                    SampleCount t,
                    Real radius,
                    Real tau) {
  if (nodes.empty() || prepared.local_to_global.empty()) return 0.0;

  // local_membership is reset only on the nodes touched by the queried set.
  // This keeps repeated density checks cheap even when the tracked graph is
  // large but the candidate solution is small.
  workspace->touched_local_nodes.clear();
  for (NodeId node : nodes) {
    const auto it = std::lower_bound(prepared.local_to_global.begin(),
                                     prepared.local_to_global.end(), node);
    if (it == prepared.local_to_global.end() || *it != node) continue;

    const NodeCount local_node =
        static_cast<NodeCount>(it - prepared.local_to_global.begin());
    if (!workspace->local_membership[local_node]) {
      workspace->local_membership[local_node] = 1;
      workspace->touched_local_nodes.push_back(local_node);
    }
  }

  const std::size_t set_size = workspace->touched_local_nodes.size();
  if (set_size == 0) return 0.0;

  Real total_weight = 0.0;
  for (EdgeIndex edge_index = 0;
       edge_index < static_cast<EdgeIndex>(prepared.edge_u.size());
       ++edge_index) {
    const NodeCount local_u = prepared.edge_u[static_cast<std::size_t>(edge_index)];
    const NodeCount local_v = prepared.edge_v[static_cast<std::size_t>(edge_index)];
    if (!workspace->local_membership[local_u] ||
        !workspace->local_membership[local_v]) {
      continue;
    }

    Real edge_weight = 0.0;
    if (weights) {
      if (edge_index >= static_cast<EdgeIndex>(weights->size())) continue;
      edge_weight =
          std::max(static_cast<Real>(0.0),
                   (*weights)[static_cast<std::size_t>(edge_index)]);
    } else {
      if (edge_index >= static_cast<EdgeIndex>(workspace->counters.size())) continue;
      const Real p_hat =
          static_cast<Real>(workspace->counters[static_cast<std::size_t>(edge_index)]) /
          static_cast<Real>(t);
      const Real lower = p_hat - radius;
      edge_weight = (lower >= tau) ? lower : 0.0;
    }
    total_weight += edge_weight;
  }

  for (NodeCount local_node : workspace->touched_local_nodes) {
    workspace->local_membership[local_node] = 0;
  }
  workspace->touched_local_nodes.clear();

  return total_weight / static_cast<Real>(set_size);
}

CertifiedResult RunCertifiedStageImpl(const Graph& graph,
                                      const std::vector<NodeId>& seeds,
                                      const TrackedGraph& tracked_graph,
                                      Real delta1,
                                      const RuntimeOptions& opts,
                                      SampleCount capped_t_max,
                                      bool enable_early_stop,
                                      CertifiedWorkspace* workspace,
                                      std::mt19937_64* rng) {
  CertifiedResult result;

  CertifiedWorkspace local_workspace;
  if (!workspace) workspace = &local_workspace;
  ResetWorkspace(graph, tracked_graph, workspace);

  const SampleCount tracked_edge_count =
      static_cast<SampleCount>(TrackedEdgeCount(tracked_graph));
  const SampleCount n0 =
      static_cast<SampleCount>(graph.num_nodes) -
      static_cast<SampleCount>(seeds.size());

  result.t_max = (capped_t_max > 0)
                     ? capped_t_max
                     : ComputeTMax(n0, tracked_edge_count, opts, delta1);
  result.first_checkpoint_t = InitialCheckpoint(tracked_edge_count, delta1);
  result.max_checkpoints =
      MaxCheckpointCount(result.first_checkpoint_t, result.t_max);
  PrintProgressLine(opts,
                    "certified",
                    "start tracked_edges=" +
                        std::to_string(tracked_edge_count) + " T_max=" +
                        std::to_string(result.t_max) + " first_checkpoint_t=" +
                        std::to_string(result.first_checkpoint_t) +
                        " max_checkpoints=" +
                        std::to_string(result.max_checkpoints) +
                        " early_stop=" +
                        std::to_string(enable_early_stop ? 1 : 0));

  if (tracked_edge_count == 0 || result.t_max == 0) {
    // With no tracked edges, the certified subproblem is trivial: solve once on
    // the empty tracked graph and return the solver's fallback behavior.
    const Real final_opt_start = NowSeconds();
    ExactSolverCallStats final_exact_stats;
    const PeelingResult peel =
        SolveSubproblem(workspace, opts, workspace->weight_buffer, &result,
                        &final_exact_stats,
                        &workspace->peeling_workspace, &workspace->exact_workspace);
    FillFinalExactTrace(&result.final_exact_trace, 0, peel, final_exact_stats,
                        NowSeconds() - final_opt_start);
    result.u_hat = peel.u_max;
    result.t_stop = 0;
    result.rho_final = peel.rho_max;
    PrintProgressLine(opts,
                      "certified",
                      "done tracked_edges=0 t_stop=0 rho_final=" +
                          std::to_string(result.rho_final));
    return result;
  }

  SampleCount t_check = result.first_checkpoint_t;
  std::uint32_t checkpoint_index = 0;
  SampleCount next_progress = ProgressReportInterval(result.t_max);

  for (SampleCount t = 1; t <= result.t_max; ++t) {
    const Real sampling_start = NowSeconds();
    const std::vector<NodeId>& activated =
        SimulateDiffusion(graph, seeds, opts, &workspace->diffusion_workspace, rng);
    MarkActivatedTrackedNodes(activated, workspace);
    UpdateCounters(workspace);
    ClearActivatedTrackedNodes(activated, workspace);
    result.sampling_sec += NowSeconds() - sampling_start;

    if (ShouldReportProgress(t, result.t_max, &next_progress)) {
      PrintProgressLine(opts,
                        "certified",
                        "sample " +
                            FormatProgressFraction(t, result.t_max));
    }

    if (t != t_check) continue;

    const Real checkpoint_start = NowSeconds();
    const Real radius =
        ComputeCheckpointRadius(t, checkpoint_index, tracked_edge_count, delta1);
    // First solve on the thresholded empirical graph \tilde{W}(t).
    BuildTildeWeights(t, opts, workspace);

    ExactSolverCallStats exact_tilde_stats;
    ExactSolverCallStats exact_plus_stats;
    const PeelingResult peel_tilde =
        SolveSubproblem(workspace, opts, workspace->weight_buffer, &result,
                        &exact_tilde_stats,
                        &workspace->peeling_workspace, &workspace->exact_workspace);
    // Then overwrite the same buffer with the optimistic graph W^+(t) and
    // solve again for the upper-certificate side.
    BuildUpperWeights(t, radius, opts, workspace);
    const PeelingResult peel_plus =
        SolveSubproblem(workspace, opts, workspace->weight_buffer, &result,
                        &exact_plus_stats,
                        &workspace->peeling_workspace, &workspace->exact_workspace);
    // rho_lower is evaluated on peel_tilde.u_max under the lower confidence
    // weights W^-(t). The lower graph is never materialized permanently.
    const Real rho_lower =
        ComputeDensity(*workspace->tracked_graph, peel_tilde.u_max, workspace,
                       nullptr, t, radius, opts.tau);
    const Real approx_factor =
        CurrentApproximationFactor(peel_plus.rho_max, rho_lower, opts);
    const bool stop_now = ShouldEarlyStop(peel_plus.rho_max, rho_lower, opts);
    const Real checkpoint_opt_sec = NowSeconds() - checkpoint_start;

    CertifiedCheckpointTrace trace;
    trace.t = t;
    trace.radius = radius;
    trace.rho_tilde = peel_tilde.rho_max;
    trace.rho_lower = rho_lower;
    trace.rho_upper = peel_plus.rho_max;
    trace.u_hat_size = static_cast<NodeCount>(peel_tilde.u_max.size());
    trace.u_plus_size = static_cast<NodeCount>(peel_plus.u_max.size());
    trace.opt_sec = checkpoint_opt_sec;
    trace.exact_solve_calls =
        exact_tilde_stats.solve_calls + exact_plus_stats.solve_calls;
    trace.exact_flow_calls =
        exact_tilde_stats.flow_calls + exact_plus_stats.flow_calls;
    trace.exact_binary_search_iters =
        exact_tilde_stats.binary_search_iters + exact_plus_stats.binary_search_iters;
    trace.exact_topology_rebuilds =
        (exact_tilde_stats.topology_rebuilt ? 1u : 0u) +
        (exact_plus_stats.topology_rebuilt ? 1u : 0u);
    trace.exact_capacity_scale_last = exact_plus_stats.capacity_scale_used;
    trace.exact_capacity_scale_max =
        std::max(exact_tilde_stats.capacity_scale_used,
                 exact_plus_stats.capacity_scale_used);
    trace.exact_solver_sec =
        exact_tilde_stats.solver_sec + exact_plus_stats.solver_sec;
    if (trace.exact_solve_calls > 0) {
      trace.exact_avg_flow_calls =
          static_cast<Real>(trace.exact_flow_calls) /
          static_cast<Real>(trace.exact_solve_calls);
      trace.exact_avg_binary_search_iters =
          static_cast<Real>(trace.exact_binary_search_iters) /
          static_cast<Real>(trace.exact_solve_calls);
      trace.exact_avg_solver_time_sec =
          trace.exact_solver_sec / static_cast<Real>(trace.exact_solve_calls);
    }
    trace.early_stop = enable_early_stop && stop_now;
    result.checkpoint_trace.push_back(trace);
    PrintProgressLine(
        opts,
        "certified",
        "checkpoint " + std::to_string(checkpoint_index + 1) +
            " t=" + std::to_string(t) + " rho_tilde=" +
            std::to_string(peel_tilde.rho_max) + " rho_lower=" +
            std::to_string(rho_lower) + " rho_upper=" +
            std::to_string(peel_plus.rho_max) + " approx_factor=" +
            std::to_string(approx_factor) + " target=" +
            std::to_string(EarlyStopTargetFactor(opts)));

    ++result.checkpoints;
    result.checkpoint_opt_sec += checkpoint_opt_sec;

    if (enable_early_stop && stop_now) {
      // The current empirical solution satisfies the solver-specific
      // multiplicative certificate, so certified sampling can stop early.
      result.u_hat = peel_tilde.u_max;
      result.early_stop = true;
      result.t_stop = t;
      result.rho_final = peel_tilde.rho_max;
      PrintProgressLine(opts,
                        "certified",
                        "done early_stop=1 t_stop=" +
                            std::to_string(result.t_stop) + " checkpoints=" +
                            std::to_string(result.checkpoints) +
                            " rho_final=" + std::to_string(result.rho_final));
      return result;
    }

    ++checkpoint_index;
    if (t_check > result.t_max / 2) {
      t_check = std::numeric_limits<SampleCount>::max();
    } else {
      t_check *= 2;
    }
  }

  const Real checkpoint_start = NowSeconds();
  // If no checkpoint certified an early stop, the stage returns the final
  // solution obtained from the empirical graph at the sampling cap.
  BuildFinalWeights(result.t_max, opts, workspace);
  ExactSolverCallStats final_exact_stats;
  const PeelingResult peel =
      SolveSubproblem(workspace, opts, workspace->weight_buffer, &result,
                      &final_exact_stats,
                      &workspace->peeling_workspace, &workspace->exact_workspace);
  const Real final_opt_sec = NowSeconds() - checkpoint_start;
  result.checkpoint_opt_sec += final_opt_sec;
  FillFinalExactTrace(&result.final_exact_trace, result.t_max, peel, final_exact_stats,
                      final_opt_sec);

  result.u_hat = peel.u_max;
  result.early_stop = false;
  result.t_stop = result.t_max;
  result.rho_final = peel.rho_max;
  PrintProgressLine(opts,
                    "certified",
                    "done early_stop=0 t_stop=" +
                        std::to_string(result.t_stop) + " checkpoints=" +
                        std::to_string(result.checkpoints) + " rho_final=" +
                        std::to_string(result.rho_final));
  return result;
}

}  // namespace

dcgs::CertifiedResult dcgs::CertifiedStage(
    const Graph& graph,
    const std::vector<NodeId>& seeds,
    const TrackedGraph& tracked_graph,
    Real delta1,
    const RuntimeOptions& opts,
    CertifiedWorkspace* workspace,
    std::mt19937_64* rng) {
  // Public entry point for the standard early-stopping certified stage.
  return RunCertifiedStageImpl(graph, seeds, tracked_graph, delta1, opts, 0, true,
                               workspace, rng);
}

dcgs::CertifiedResult dcgs::CertifiedStage(
    const Graph& graph,
    const std::vector<NodeId>& seeds,
    const TrackedGraph& tracked_graph,
    Real delta1,
    const RuntimeOptions& opts,
    std::mt19937_64* rng) {
  // Convenience overload that allocates a temporary workspace.
  CertifiedWorkspace workspace;
  return CertifiedStage(graph, seeds, tracked_graph, delta1, opts, &workspace, rng);
}

dcgs::CertifiedResult dcgs::CertifiedStageCappedNoEarlyStop(
    const Graph& graph,
    const std::vector<NodeId>& seeds,
    const TrackedGraph& tracked_graph,
    Real delta1,
    SampleCount sample_cap,
    const RuntimeOptions& opts,
    CertifiedWorkspace* workspace,
    std::mt19937_64* rng) {
  // Public entry point for the fixed-cap certified baseline without early stop.
  return RunCertifiedStageImpl(graph, seeds, tracked_graph, delta1, opts,
                               sample_cap, false, workspace, rng);
}

dcgs::CertifiedResult dcgs::CertifiedStageCappedNoEarlyStop(
    const Graph& graph,
    const std::vector<NodeId>& seeds,
    const TrackedGraph& tracked_graph,
    Real delta1,
    SampleCount sample_cap,
    const RuntimeOptions& opts,
    std::mt19937_64* rng) {
  // Convenience overload that allocates a temporary workspace.
  CertifiedWorkspace workspace;
  return CertifiedStageCappedNoEarlyStop(graph, seeds, tracked_graph, delta1,
                                         sample_cap, opts, &workspace, rng);
}
