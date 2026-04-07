#ifndef DCGS_PIPELINE_H_
#define DCGS_PIPELINE_H_

#include "certified.h"
#include "pilot.h"

#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace dcgs {

// TrialSummary is the per-trial compact record used by multi-trial reporting.
struct TrialSummary {
  std::uint32_t trial_index = 0;
  std::uint64_t trial_seed = 0;
  SampleCount sample_cap_used = 0;
  SampleCount eval_cap_used = 0;
  NodeCount u_hat_size = 0;
  NodeCount tracked_nodes = 0;
  NodeCount e_hat_tau_size = 0;
  SampleCount t0 = 0;
  SampleCount t_stop = 0;
  SampleCount t_max = 0;
  std::uint32_t checkpoints = 0;
  Real rho_final = 0.0;
  Real rho_eval = 0.0;
  bool early_stop = false;
  Real pilot_time_sec = 0.0;
  Real sampling_sec = 0.0;
  Real checkpoint_opt_sec = 0.0;
  std::uint64_t exact_solve_calls = 0;
  std::uint64_t exact_flow_calls = 0;
  Real exact_avg_flow_calls = 0.0;
  std::uint64_t exact_binary_search_iters = 0;
  Real exact_avg_binary_search_iters = 0.0;
  std::uint64_t exact_topology_rebuilds = 0;
  std::int64_t exact_capacity_scale_last = 0;
  std::int64_t exact_capacity_scale_max = 0;
  Real exact_solver_sec = 0.0;
  Real exact_avg_solver_time_sec = 0.0;
  Real eval_time_sec = 0.0;
  Real total_time_sec = 0.0;
};

// TrialAggregateMetric stores mean/min/max for one scalar metric across trials.
struct TrialAggregateMetric {
  Real mean = 0.0;
  Real min = 0.0;
  Real max = 0.0;
};

// TrialAggregateStats stores aggregate statistics over all completed trials.
struct TrialAggregateStats {
  std::uint32_t trials = 0;
  std::uint32_t early_stop_count = 0;
  Real early_stop_rate = 0.0;
  TrialAggregateMetric rho_final;
  TrialAggregateMetric rho_eval;
  TrialAggregateMetric u_hat_size;
  TrialAggregateMetric tracked_nodes;
  TrialAggregateMetric e_hat_tau_size;
  TrialAggregateMetric sample_cap_used;
  TrialAggregateMetric eval_cap_used;
  TrialAggregateMetric t0;
  TrialAggregateMetric t_stop;
  TrialAggregateMetric t_max;
  TrialAggregateMetric checkpoints;
  TrialAggregateMetric pilot_time_sec;
  TrialAggregateMetric sampling_sec;
  TrialAggregateMetric checkpoint_opt_sec;
  TrialAggregateMetric exact_solve_calls;
  TrialAggregateMetric exact_flow_calls;
  TrialAggregateMetric exact_avg_flow_calls;
  TrialAggregateMetric exact_binary_search_iters;
  TrialAggregateMetric exact_avg_binary_search_iters;
  TrialAggregateMetric exact_topology_rebuilds;
  TrialAggregateMetric exact_solver_sec;
  TrialAggregateMetric exact_avg_solver_time_sec;
  TrialAggregateMetric eval_time_sec;
  TrialAggregateMetric total_time_sec;
};

// DcgsResult is the top-level result object returned by all algorithms.
struct DcgsResult {
  std::vector<NodeId> u_hat;
  PilotResult pilot;
  CertifiedResult certified;
  SampleCount sample_cap_used = 0;
  SampleCount eval_cap_used = 0;
  Real rho_eval = 0.0;
  Real eval_time_sec = 0.0;
  std::uint32_t representative_trial_index = 0;
  TrialAggregateStats aggregate;
  std::vector<TrialSummary> trial_summaries;
};

// MainDcgs runs the standard pilot + certified pipeline for one trial.
DcgsResult MainDcgs(const Graph& graph,
                    const std::vector<NodeId>& seeds,
                    const RuntimeOptions& opts,
                    std::mt19937_64* rng);

// RunTrials executes the selected algorithm for opts.trials independent trials
// and fills aggregate statistics plus the representative result.
bool RunTrials(const Graph& graph,
               const std::vector<NodeId>& seeds,
               const RuntimeOptions& opts,
               DcgsResult* result,
               std::string* err);

}  // namespace dcgs

#endif  // DCGS_PIPELINE_H_
