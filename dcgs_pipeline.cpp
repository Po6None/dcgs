#include "dcgs_pipeline.h"

#include "baseline.h"
#include "progress.h"
#include "rho_estimator.h"

namespace dcgs {

namespace {

// MixSeed deterministically derives one independent-looking seed per trial or
// evaluation pass from the user-provided base seed.
std::uint64_t MixSeed(std::uint64_t base_seed, std::uint64_t trial_index) {
  std::uint64_t z = base_seed + 0x9e3779b97f4a7c15ull + trial_index;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  return z ^ (z >> 31);
}

// TotalCoreTime aggregates the pipeline times that are measured separately in
// the result object.
Real TotalCoreTime(const DcgsResult& result) {
  return result.pilot.elapsed_sec +
         result.certified.sampling_sec +
         result.certified.checkpoint_opt_sec +
         result.eval_time_sec;
}

// AverageExactBinarySearchIters derives a per-solve average from cumulative stats.
Real AverageExactBinarySearchIters(std::uint64_t total_iters,
                                   std::uint64_t solve_calls) {
  if (solve_calls == 0) return 0.0;
  return static_cast<Real>(total_iters) / static_cast<Real>(solve_calls);
}

// AverageExactFlowCalls derives a per-solve average from cumulative stats.
Real AverageExactFlowCalls(std::uint64_t total_flow_calls,
                           std::uint64_t solve_calls) {
  if (solve_calls == 0) return 0.0;
  return static_cast<Real>(total_flow_calls) / static_cast<Real>(solve_calls);
}

// AverageExactSolverTime derives a per-solve average from cumulative stats.
Real AverageExactSolverTime(Real total_solver_sec,
                            std::uint64_t solve_calls) {
  if (solve_calls == 0) return 0.0;
  return total_solver_sec / static_cast<Real>(solve_calls);
}

// MakeTrialSummary converts one DcgsResult into the compact record used for
// trial reporting and aggregate statistics.
TrialSummary MakeTrialSummary(const DcgsResult& result,
                              std::uint32_t trial_index,
                              std::uint64_t trial_seed) {
  TrialSummary summary;
  summary.trial_index = trial_index;
  summary.trial_seed = trial_seed;
  summary.sample_cap_used = result.sample_cap_used;
  summary.eval_cap_used = result.eval_cap_used;
  summary.u_hat_size = static_cast<NodeCount>(result.u_hat.size());
  summary.tracked_nodes = TrackedNodeCount(result.pilot.tracked_graph);
  summary.e_hat_tau_size =
      static_cast<NodeCount>(TrackedEdgeCount(result.pilot.tracked_graph));
  summary.t0 = result.pilot.t0;
  summary.t_stop = result.certified.t_stop;
  summary.t_max = result.certified.t_max;
  summary.checkpoints = result.certified.checkpoints;
  summary.rho_final = result.certified.rho_final;
  summary.rho_eval = result.rho_eval;
  summary.early_stop = result.certified.early_stop;
  summary.pilot_time_sec = result.pilot.elapsed_sec;
  summary.sampling_sec = result.certified.sampling_sec;
  summary.checkpoint_opt_sec = result.certified.checkpoint_opt_sec;
  summary.exact_solve_calls = result.certified.exact_solve_calls;
  summary.exact_flow_calls = result.certified.exact_flow_calls;
  summary.exact_avg_flow_calls =
      AverageExactFlowCalls(result.certified.exact_flow_calls,
                            result.certified.exact_solve_calls);
  summary.exact_binary_search_iters = result.certified.exact_binary_search_iters;
  summary.exact_avg_binary_search_iters =
      AverageExactBinarySearchIters(result.certified.exact_binary_search_iters,
                                    result.certified.exact_solve_calls);
  summary.exact_topology_rebuilds = result.certified.exact_topology_rebuilds;
  summary.exact_capacity_scale_last = result.certified.exact_capacity_scale_last;
  summary.exact_capacity_scale_max = result.certified.exact_capacity_scale_max;
  summary.exact_solver_sec = result.certified.exact_solver_sec;
  summary.exact_avg_solver_time_sec =
      AverageExactSolverTime(result.certified.exact_solver_sec,
                             result.certified.exact_solve_calls);
  summary.eval_time_sec = result.eval_time_sec;
  summary.total_time_sec = TotalCoreTime(result);
  return summary;
}

// UpdateMetric maintains mean accumulation and running min/max.
void UpdateMetric(Real value,
                  std::uint32_t count,
                  TrialAggregateMetric* metric) {
  if (!metric) return;
  if (count == 1) {
    metric->mean = value;
    metric->min = value;
    metric->max = value;
    return;
  }
  metric->mean += value;
  if (value < metric->min) metric->min = value;
  if (value > metric->max) metric->max = value;
}

// UpdateAggregate folds one finished trial into the global aggregate stats.
void UpdateAggregate(const TrialSummary& summary, TrialAggregateStats* aggregate) {
  if (!aggregate) return;
  ++aggregate->trials;
  const std::uint32_t count = aggregate->trials;
  if (summary.early_stop) ++aggregate->early_stop_count;

  UpdateMetric(summary.rho_final, count, &aggregate->rho_final);
  UpdateMetric(summary.rho_eval, count, &aggregate->rho_eval);
  UpdateMetric(static_cast<Real>(summary.u_hat_size), count, &aggregate->u_hat_size);
  UpdateMetric(static_cast<Real>(summary.tracked_nodes), count, &aggregate->tracked_nodes);
  UpdateMetric(static_cast<Real>(summary.e_hat_tau_size), count, &aggregate->e_hat_tau_size);
  UpdateMetric(static_cast<Real>(summary.sample_cap_used), count, &aggregate->sample_cap_used);
  UpdateMetric(static_cast<Real>(summary.eval_cap_used), count, &aggregate->eval_cap_used);
  UpdateMetric(static_cast<Real>(summary.t0), count, &aggregate->t0);
  UpdateMetric(static_cast<Real>(summary.t_stop), count, &aggregate->t_stop);
  UpdateMetric(static_cast<Real>(summary.t_max), count, &aggregate->t_max);
  UpdateMetric(static_cast<Real>(summary.checkpoints), count, &aggregate->checkpoints);
  UpdateMetric(summary.pilot_time_sec, count, &aggregate->pilot_time_sec);
  UpdateMetric(summary.sampling_sec, count, &aggregate->sampling_sec);
  UpdateMetric(summary.checkpoint_opt_sec, count, &aggregate->checkpoint_opt_sec);
  UpdateMetric(static_cast<Real>(summary.exact_solve_calls), count,
               &aggregate->exact_solve_calls);
  UpdateMetric(static_cast<Real>(summary.exact_flow_calls), count,
               &aggregate->exact_flow_calls);
  UpdateMetric(summary.exact_avg_flow_calls, count,
               &aggregate->exact_avg_flow_calls);
  UpdateMetric(static_cast<Real>(summary.exact_binary_search_iters), count,
               &aggregate->exact_binary_search_iters);
  UpdateMetric(summary.exact_avg_binary_search_iters, count,
               &aggregate->exact_avg_binary_search_iters);
  UpdateMetric(static_cast<Real>(summary.exact_topology_rebuilds), count,
               &aggregate->exact_topology_rebuilds);
  UpdateMetric(summary.exact_solver_sec, count, &aggregate->exact_solver_sec);
  UpdateMetric(summary.exact_avg_solver_time_sec, count,
               &aggregate->exact_avg_solver_time_sec);
  UpdateMetric(summary.eval_time_sec, count, &aggregate->eval_time_sec);
  UpdateMetric(summary.total_time_sec, count, &aggregate->total_time_sec);
}

// FinalizeAggregate turns accumulated sums into means once all trials finish.
void FinalizeAggregate(TrialAggregateStats* aggregate) {
  if (!aggregate || aggregate->trials == 0) return;
  const Real inv_trials = 1.0 / static_cast<Real>(aggregate->trials);
  aggregate->early_stop_rate =
      static_cast<Real>(aggregate->early_stop_count) * inv_trials;
  aggregate->rho_final.mean *= inv_trials;
  aggregate->rho_eval.mean *= inv_trials;
  aggregate->u_hat_size.mean *= inv_trials;
  aggregate->tracked_nodes.mean *= inv_trials;
  aggregate->e_hat_tau_size.mean *= inv_trials;
  aggregate->sample_cap_used.mean *= inv_trials;
  aggregate->eval_cap_used.mean *= inv_trials;
  aggregate->t0.mean *= inv_trials;
  aggregate->t_stop.mean *= inv_trials;
  aggregate->t_max.mean *= inv_trials;
  aggregate->checkpoints.mean *= inv_trials;
  aggregate->pilot_time_sec.mean *= inv_trials;
  aggregate->sampling_sec.mean *= inv_trials;
  aggregate->checkpoint_opt_sec.mean *= inv_trials;
  aggregate->exact_solve_calls.mean *= inv_trials;
  aggregate->exact_flow_calls.mean *= inv_trials;
  aggregate->exact_avg_flow_calls.mean *= inv_trials;
  aggregate->exact_binary_search_iters.mean *= inv_trials;
  aggregate->exact_avg_binary_search_iters.mean *= inv_trials;
  aggregate->exact_topology_rebuilds.mean *= inv_trials;
  aggregate->exact_solver_sec.mean *= inv_trials;
  aggregate->exact_avg_solver_time_sec.mean *= inv_trials;
  aggregate->eval_time_sec.mean *= inv_trials;
  aggregate->total_time_sec.mean *= inv_trials;
}

// MaybeEvaluateRho performs the optional common post-hoc evaluation of rho(U).
void MaybeEvaluateRho(const Graph& graph,
                      const std::vector<NodeId>& seeds,
                      const RuntimeOptions& opts,
                      std::uint64_t trial_seed,
                      RhoEstimationWorkspace* eval_workspace,
                      DcgsResult* result) {
  if (!result || opts.eval_cap == 0) return;
  PrintProgressLine(opts,
                    "evaluation",
                    "start eval_cap=" + std::to_string(opts.eval_cap));
  const std::uint64_t eval_seed =
      MixSeed(trial_seed, 0xd6e8feb86659fd93ull);
  std::mt19937_64 eval_rng(eval_seed);
  const RhoEstimationResult eval = EstimateRhoBySampling(graph,
                                                         seeds,
                                                         result->u_hat,
                                                         opts.eval_cap,
                                                         opts,
                                                         eval_workspace,
                                                         &eval_rng);
  result->eval_cap_used = eval.eval_cap_used;
  result->rho_eval = eval.rho_hat;
  result->eval_time_sec = eval.elapsed_sec;
  PrintProgressLine(opts,
                    "evaluation",
                    "done eval_cap_used=" +
                        std::to_string(result->eval_cap_used) +
                        " rho_eval=" + std::to_string(result->rho_eval));
}

}  // namespace

DcgsResult MainDcgs(const Graph& graph, const std::vector<NodeId>& seeds,
                    const RuntimeOptions& opts, std::mt19937_64* rng) {
  // The main method splits delta into pilot and certified budgets and then runs
  // the two-stage DCGS pipeline once.
  DcgsResult result;
  const Real delta0 = opts.delta * 0.5;
  const Real delta1 = opts.delta * 0.5;
  result.pilot = PilotStage(graph, seeds, delta0, opts, rng);
  result.certified =
      CertifiedStage(graph, seeds, result.pilot.tracked_graph, delta1, opts, rng);
  result.u_hat = result.certified.u_hat;
  return result;
}

DcgsResult MainCertifiedCapNoEarlyStop(const Graph& graph,
                                       const std::vector<NodeId>& seeds,
                                       const RuntimeOptions& opts,
                                       std::mt19937_64* rng) {
  // This baseline shares the pilot stage with DCGS but disables certified
  // early stopping in favor of a fixed sample cap.
  DcgsResult result;
  const Real delta0 = opts.delta * 0.5;
  const Real delta1 = opts.delta * 0.5;
  result.pilot = PilotStage(graph, seeds, delta0, opts, rng);
  result.certified = CertifiedStageCappedNoEarlyStop(
      graph, seeds, result.pilot.tracked_graph, delta1, opts.sample_cap, opts, rng);
  result.u_hat = result.certified.u_hat;
  result.sample_cap_used = opts.sample_cap;
  return result;
}

bool RunSingleTrial(const Graph& graph,
                    const std::vector<NodeId>& seeds,
                    const RuntimeOptions& opts,
                    BaselineWorkspace* baseline_workspace,
                    std::mt19937_64* rng,
                    DcgsResult* result,
                    std::string* err) {
  // The dispatch layer keeps the rest of the trial loop independent of which
  // algorithm family is selected at the CLI level.
  if (!result) {
    if (err) *err = "Internal error: result is null.";
    return false;
  }

  switch (opts.algorithm) {
    case AlgorithmKind::kDcgs:
      *result = MainDcgs(graph, seeds, opts, rng);
      return true;
    case AlgorithmKind::kBestPair:
      *result = RunBestPairBaseline(graph, seeds, opts, baseline_workspace, rng);
      return true;
    case AlgorithmKind::kTopKActivationProb:
      *result = RunTopKActivationProbBaseline(graph, seeds, opts, baseline_workspace, rng);
      return true;
    case AlgorithmKind::kTopKCoactivationDegree:
      *result = RunTopKCoactivationDegreeBaseline(graph, seeds, opts, baseline_workspace, rng);
      return true;
    case AlgorithmKind::kGreedyAdd:
      *result = RunGreedyAddBaseline(graph, seeds, opts, baseline_workspace, rng);
      return true;
    case AlgorithmKind::kLocalSearch:
      *result = RunLocalSearchBaseline(graph, seeds, opts, baseline_workspace, rng);
      return true;
    case AlgorithmKind::kStructurePeeling:
      *result = RunStructurePeelingBaseline(graph, seeds, opts, baseline_workspace, rng);
      return true;
    case AlgorithmKind::kEdgeCoactPeeling:
      *result = RunEdgeCoactPeelingBaseline(graph, seeds, opts, baseline_workspace, rng);
      return true;
    case AlgorithmKind::kCertifiedCapNoEarlyStop:
      *result = MainCertifiedCapNoEarlyStop(graph, seeds, opts, rng);
      return true;
  }

  if (err) {
    *err = std::string("Unknown algorithm kind: ") + AlgorithmName(opts.algorithm);
  }
  return false;
}

bool RunTrials(const Graph& graph,
               const std::vector<NodeId>& seeds,
               const RuntimeOptions& opts,
               DcgsResult* result,
               std::string* err) {
  // Trials are intentionally independent and deterministic once rng_seed is
  // fixed, which makes outer-script parallelization straightforward.
  if (!result) {
    if (err) *err = "Internal error: result is null.";
    return false;
  }

  DcgsResult representative;
  TrialAggregateStats aggregate;
  std::vector<TrialSummary> summaries;
  summaries.reserve(opts.trials);
  BaselineWorkspace baseline_workspace;
  RhoEstimationWorkspace eval_workspace;

  for (std::uint32_t trial_index = 0; trial_index < opts.trials; ++trial_index) {
    const std::uint64_t trial_seed = MixSeed(opts.rng_seed, trial_index);
    std::mt19937_64 trial_rng(trial_seed);
    DcgsResult trial_result;
    PrintProgressLine(opts,
                      "trial",
                      std::to_string(trial_index + 1) + "/" +
                          std::to_string(opts.trials) + " start algorithm=" +
                          AlgorithmName(opts.algorithm) + " seed=" +
                          std::to_string(trial_seed));
    if (!RunSingleTrial(graph,
                        seeds,
                        opts,
                        &baseline_workspace,
                        &trial_rng,
                        &trial_result,
                        err)) {
      return false;
    }
    MaybeEvaluateRho(graph, seeds, opts, trial_seed, &eval_workspace, &trial_result);
    const TrialSummary summary = MakeTrialSummary(trial_result, trial_index, trial_seed);
    UpdateAggregate(summary, &aggregate);
    summaries.push_back(summary);
    PrintProgressLine(opts,
                      "trial",
                      std::to_string(trial_index + 1) + "/" +
                          std::to_string(opts.trials) + " done u_hat_size=" +
                          std::to_string(summary.u_hat_size) + " rho_final=" +
                          std::to_string(summary.rho_final) + " total_time_sec=" +
                          std::to_string(summary.total_time_sec));

    representative = std::move(trial_result);
    representative.representative_trial_index = trial_index;
  }

  FinalizeAggregate(&aggregate);
  representative.aggregate = aggregate;
  representative.trial_summaries = std::move(summaries);
  *result = std::move(representative);
  return true;
}

}  // namespace dcgs
