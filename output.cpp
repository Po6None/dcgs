#include "output.h"

#include "dcgs_pipeline.h"
#include "resource_usage.h"

#include <filesystem>
#include <fstream>
#include <iostream>

namespace dcgs {

namespace {

// EnsureParentDirectory creates the parent directory of one output file path
// before the file itself is opened.
bool EnsureParentDirectory(const std::string& path, std::string* err) {
  if (path.empty()) return true;

  const std::filesystem::path fs_path(path);
  const std::filesystem::path parent = fs_path.parent_path();
  if (parent.empty()) return true;

  std::error_code ec;
  if (std::filesystem::exists(parent, ec)) {
    if (!ec) return true;
    if (err) *err = "Failed to inspect output directory: " + parent.string();
    return false;
  }
  if (std::filesystem::create_directories(parent, ec)) return true;
  if (!ec && std::filesystem::exists(parent)) return true;
  if (err) *err = "Failed to create output directory: " + parent.string();
  return false;
}

// AlgorithmFamilyName groups algorithms by their reporting semantics.
const char* AlgorithmFamilyName(AlgorithmKind algorithm) {
  switch (algorithm) {
    case AlgorithmKind::kDcgs:
      return "dcgs";
    case AlgorithmKind::kBestPair:
    case AlgorithmKind::kTopKActivationProb:
    case AlgorithmKind::kTopKCoactivationDegree:
    case AlgorithmKind::kGreedyAdd:
    case AlgorithmKind::kLocalSearch:
      return "sample_budget_baseline";
    case AlgorithmKind::kStructurePeeling:
      return "structure_baseline";
    case AlgorithmKind::kEdgeCoactPeeling:
      return "edge_coact_baseline";
    case AlgorithmKind::kCertifiedCapNoEarlyStop:
      return "certified_baseline";
  }
  return "unknown";
}

// UsesPilot marks algorithms that execute the pilot stage and therefore expose
// tracked-edge statistics in a meaningful way.
bool UsesPilot(AlgorithmKind algorithm) {
  return algorithm == AlgorithmKind::kDcgs ||
         algorithm == AlgorithmKind::kCertifiedCapNoEarlyStop;
}

// UsesCheckpointing marks algorithms that execute certified-style checkpoints.
bool UsesCheckpointing(AlgorithmKind algorithm) {
  return algorithm == AlgorithmKind::kDcgs ||
         algorithm == AlgorithmKind::kCertifiedCapNoEarlyStop;
}

// AverageExactBinarySearchIters derives a per-solve average from cumulative stats.
Real AverageExactBinarySearchIters(const CertifiedResult& result) {
  if (result.exact_solve_calls == 0) return 0.0;
  return static_cast<Real>(result.exact_binary_search_iters) /
         static_cast<Real>(result.exact_solve_calls);
}

// AverageExactFlowCalls derives a per-solve average from cumulative stats.
Real AverageExactFlowCalls(const CertifiedResult& result) {
  if (result.exact_solve_calls == 0) return 0.0;
  return static_cast<Real>(result.exact_flow_calls) /
         static_cast<Real>(result.exact_solve_calls);
}

// AverageExactSolverTime derives a per-solve average from cumulative stats.
Real AverageExactSolverTime(const CertifiedResult& result) {
  if (result.exact_solve_calls == 0) return 0.0;
  return result.exact_solver_sec / static_cast<Real>(result.exact_solve_calls);
}

// WriteAggregateMetric emits one aggregate metric in the plain-text log format.
void WriteAggregateMetric(std::ostream& out,
                          const std::string& name,
                          const TrialAggregateMetric& metric) {
  out << "aggregate." << name << ".mean=" << metric.mean << "\n";
  out << "aggregate." << name << ".min=" << metric.min << "\n";
  out << "aggregate." << name << ".max=" << metric.max << "\n";
}

// WriteSummary prints the representative-trial summary shared by stdout and the
// textual log file.
void WriteSummary(std::ostream& out,
                  const RuntimeOptions& opts,
                  const DcgsResult& result) {
  const ProcessMemorySnapshot memory = ReadProcessMemorySnapshot();
  out << "trials=" << result.aggregate.trials << "\n";
  out << "representative_trial_index=" << result.representative_trial_index << "\n";
  out << "algorithm=" << AlgorithmName(opts.algorithm) << "\n";
  out << "subproblem_solver=" << SubproblemSolverName(opts.subproblem_solver) << "\n";
  out << "algorithm_family=" << AlgorithmFamilyName(opts.algorithm) << "\n";
  out << "model=" << ((opts.model == DiffusionModel::kIC) ? "ic" : "lt") << "\n";
  out << "pilot_used=" << (UsesPilot(opts.algorithm) ? 1 : 0) << "\n";
  out << "checkpointing_used=" << (UsesCheckpointing(opts.algorithm) ? 1 : 0) << "\n";
  out << "sample_cap=" << opts.sample_cap << "\n";
  out << "sample_cap_used=" << result.sample_cap_used << "\n";
  out << "eval_cap=" << opts.eval_cap << "\n";
  out << "eval_cap_used=" << result.eval_cap_used << "\n";
  out << "n_hat=" << result.u_hat.size() << "\n";
  out << "tracked_nodes=" << TrackedNodeCount(result.pilot.tracked_graph) << "\n";
  out << "rho_final=" << result.certified.rho_final << "\n";
  out << "rho_eval=" << result.rho_eval << "\n";
  out << "early_stop=" << (result.certified.early_stop ? 1 : 0) << "\n";
  out << "t_stop=" << result.certified.t_stop << "\n";
  out << "checkpoints=" << result.certified.checkpoints << "\n";
  out << "max_checkpoints=" << result.certified.max_checkpoints << "\n";
  out << "first_checkpoint_t=" << result.certified.first_checkpoint_t << "\n";
  out << "checkpoint_trace_count=" << result.certified.checkpoint_trace.size() << "\n";
  out << "e_hat_tau_size=" << TrackedEdgeCount(result.pilot.tracked_graph) << "\n";
  out << "T0=" << result.pilot.t0 << "\n";
  out << "T_max=" << result.certified.t_max << "\n";
  out << "tau=" << opts.tau << "\n";
  out << "eps=" << opts.eps << "\n";
  out << "delta=" << opts.delta << "\n";
  out << "gamma=" << opts.gamma << "\n";
  out << "pilot_chunk_bytes=" << opts.pilot_chunk_bytes << "\n";
  out << "baseline_chunk_bytes=" << opts.baseline_chunk_bytes << "\n";
  out << "pilot_time_sec=" << result.pilot.elapsed_sec << "\n";
  out << "sampling_time_sec=" << result.certified.sampling_sec << "\n";
  out << "checkpoint_opt_time_sec=" << result.certified.checkpoint_opt_sec << "\n";
  out << "exact_solve_calls=" << result.certified.exact_solve_calls << "\n";
  out << "exact_flow_calls=" << result.certified.exact_flow_calls << "\n";
  out << "exact_avg_flow_calls="
      << AverageExactFlowCalls(result.certified) << "\n";
  out << "exact_binary_search_iters=" << result.certified.exact_binary_search_iters << "\n";
  out << "exact_avg_binary_search_iters="
      << AverageExactBinarySearchIters(result.certified) << "\n";
  out << "exact_topology_rebuilds=" << result.certified.exact_topology_rebuilds << "\n";
  out << "exact_capacity_scale_last=" << result.certified.exact_capacity_scale_last << "\n";
  out << "exact_capacity_scale_max=" << result.certified.exact_capacity_scale_max << "\n";
  out << "exact_solver_time_sec=" << result.certified.exact_solver_sec << "\n";
  out << "exact_avg_solver_time_sec="
      << AverageExactSolverTime(result.certified) << "\n";
  out << "final_exact_trace_used="
      << (result.certified.final_exact_trace.used ? 1 : 0) << "\n";
  out << "eval_time_sec=" << result.eval_time_sec << "\n";
  out << "memory_stats_available=" << (memory.available ? 1 : 0) << "\n";
  out << "memory_rss_bytes=" << memory.rss_bytes << "\n";
  out << "memory_rss_mib=" << BytesToMiB(memory.rss_bytes) << "\n";
  out << "memory_peak_rss_bytes=" << memory.peak_rss_bytes << "\n";
  out << "memory_peak_rss_mib=" << BytesToMiB(memory.peak_rss_bytes) << "\n";
}

// WriteAggregateText appends multi-trial aggregates to the text log.
void WriteAggregateText(std::ostream& out, const DcgsResult& result) {
  if (result.aggregate.trials <= 1) return;
  out << "aggregate.early_stop_count=" << result.aggregate.early_stop_count << "\n";
  out << "aggregate.early_stop_rate=" << result.aggregate.early_stop_rate << "\n";
  WriteAggregateMetric(out, "rho_final", result.aggregate.rho_final);
  WriteAggregateMetric(out, "rho_eval", result.aggregate.rho_eval);
  WriteAggregateMetric(out, "u_hat_size", result.aggregate.u_hat_size);
  WriteAggregateMetric(out, "tracked_nodes", result.aggregate.tracked_nodes);
  WriteAggregateMetric(out, "e_hat_tau_size", result.aggregate.e_hat_tau_size);
  WriteAggregateMetric(out, "sample_cap_used", result.aggregate.sample_cap_used);
  WriteAggregateMetric(out, "eval_cap_used", result.aggregate.eval_cap_used);
  WriteAggregateMetric(out, "t0", result.aggregate.t0);
  WriteAggregateMetric(out, "t_stop", result.aggregate.t_stop);
  WriteAggregateMetric(out, "t_max", result.aggregate.t_max);
  WriteAggregateMetric(out, "checkpoints", result.aggregate.checkpoints);
  WriteAggregateMetric(out, "pilot_time_sec", result.aggregate.pilot_time_sec);
  WriteAggregateMetric(out, "sampling_sec", result.aggregate.sampling_sec);
  WriteAggregateMetric(out, "checkpoint_opt_sec", result.aggregate.checkpoint_opt_sec);
  WriteAggregateMetric(out, "exact_solve_calls", result.aggregate.exact_solve_calls);
  WriteAggregateMetric(out, "exact_flow_calls", result.aggregate.exact_flow_calls);
  WriteAggregateMetric(out, "exact_avg_flow_calls",
                       result.aggregate.exact_avg_flow_calls);
  WriteAggregateMetric(out, "exact_binary_search_iters",
                       result.aggregate.exact_binary_search_iters);
  WriteAggregateMetric(out, "exact_avg_binary_search_iters",
                       result.aggregate.exact_avg_binary_search_iters);
  WriteAggregateMetric(out, "exact_topology_rebuilds",
                       result.aggregate.exact_topology_rebuilds);
  WriteAggregateMetric(out, "exact_solver_sec", result.aggregate.exact_solver_sec);
  WriteAggregateMetric(out, "exact_avg_solver_time_sec",
                       result.aggregate.exact_avg_solver_time_sec);
  WriteAggregateMetric(out, "eval_time_sec", result.aggregate.eval_time_sec);
  WriteAggregateMetric(out, "total_time_sec", result.aggregate.total_time_sec);
}

// WriteTrialSummariesText prints one compact line per trial.
void WriteTrialSummariesText(std::ostream& out, const DcgsResult& result) {
  for (const TrialSummary& summary : result.trial_summaries) {
    out << "trial[" << summary.trial_index << "]"
        << " seed=" << summary.trial_seed
        << " sample_cap_used=" << summary.sample_cap_used
        << " eval_cap_used=" << summary.eval_cap_used
        << " u_hat_size=" << summary.u_hat_size
        << " tracked_nodes=" << summary.tracked_nodes
        << " e_hat_tau_size=" << summary.e_hat_tau_size
        << " rho_final=" << summary.rho_final
        << " rho_eval=" << summary.rho_eval
        << " early_stop=" << (summary.early_stop ? 1 : 0)
        << " t0=" << summary.t0
        << " t_stop=" << summary.t_stop
        << " t_max=" << summary.t_max
        << " checkpoints=" << summary.checkpoints
        << " pilot_time_sec=" << summary.pilot_time_sec
        << " sampling_sec=" << summary.sampling_sec
        << " checkpoint_opt_sec=" << summary.checkpoint_opt_sec
        << " exact_solve_calls=" << summary.exact_solve_calls
        << " exact_flow_calls=" << summary.exact_flow_calls
        << " exact_avg_flow_calls=" << summary.exact_avg_flow_calls
        << " exact_binary_search_iters=" << summary.exact_binary_search_iters
        << " exact_avg_binary_search_iters=" << summary.exact_avg_binary_search_iters
        << " exact_topology_rebuilds=" << summary.exact_topology_rebuilds
        << " exact_capacity_scale_last=" << summary.exact_capacity_scale_last
        << " exact_capacity_scale_max=" << summary.exact_capacity_scale_max
        << " exact_solver_sec=" << summary.exact_solver_sec
        << " exact_avg_solver_time_sec=" << summary.exact_avg_solver_time_sec
        << " eval_time_sec=" << summary.eval_time_sec
        << " total_time_sec=" << summary.total_time_sec
        << "\n";
  }
}

// WriteCheckpointTraceText prints checkpoint-level traces in a readable text form.
void WriteCheckpointTraceText(std::ostream& out, const DcgsResult& result) {
  for (std::size_t i = 0; i < result.certified.checkpoint_trace.size(); ++i) {
    const CertifiedCheckpointTrace& trace = result.certified.checkpoint_trace[i];
    out << "checkpoint[" << i << "]"
        << " t=" << trace.t
        << " radius=" << trace.radius
        << " rho_tilde=" << trace.rho_tilde
        << " rho_lower=" << trace.rho_lower
        << " rho_upper=" << trace.rho_upper
        << " u_hat_size=" << trace.u_hat_size
        << " u_plus_size=" << trace.u_plus_size
        << " opt_sec=" << trace.opt_sec
        << " exact_solve_calls=" << trace.exact_solve_calls
        << " exact_flow_calls=" << trace.exact_flow_calls
        << " exact_avg_flow_calls=" << trace.exact_avg_flow_calls
        << " exact_binary_search_iters=" << trace.exact_binary_search_iters
        << " exact_avg_binary_search_iters=" << trace.exact_avg_binary_search_iters
        << " exact_topology_rebuilds=" << trace.exact_topology_rebuilds
        << " exact_capacity_scale_last=" << trace.exact_capacity_scale_last
        << " exact_capacity_scale_max=" << trace.exact_capacity_scale_max
        << " exact_solver_sec=" << trace.exact_solver_sec
        << " exact_avg_solver_time_sec=" << trace.exact_avg_solver_time_sec
        << " early_stop=" << (trace.early_stop ? 1 : 0)
        << "\n";
  }
}

// WriteFinalExactTraceText prints the final exact solve trace when present.
void WriteFinalExactTraceText(std::ostream& out, const DcgsResult& result) {
  const CertifiedFinalExactTrace& trace = result.certified.final_exact_trace;
  if (!trace.used) return;
  out << "final_exact"
      << " t=" << trace.t
      << " u_hat_size=" << trace.u_hat_size
      << " rho=" << trace.rho
      << " opt_sec=" << trace.opt_sec
      << " exact_solve_calls=" << trace.exact_solve_calls
      << " exact_flow_calls=" << trace.exact_flow_calls
      << " exact_avg_flow_calls=" << trace.exact_avg_flow_calls
      << " exact_binary_search_iters=" << trace.exact_binary_search_iters
      << " exact_avg_binary_search_iters=" << trace.exact_avg_binary_search_iters
      << " exact_topology_rebuilds=" << trace.exact_topology_rebuilds
      << " exact_capacity_scale_last=" << trace.exact_capacity_scale_last
      << " exact_capacity_scale_max=" << trace.exact_capacity_scale_max
      << " exact_solver_sec=" << trace.exact_solver_sec
      << " exact_avg_solver_time_sec=" << trace.exact_avg_solver_time_sec
      << "\n";
}

}  // namespace

bool SaveNodeList(const std::string& path,
                  const std::vector<NodeId>& nodes,
                  std::string* err) {
  // The node list is intentionally plain text so downstream scripts can diff or
  // post-process it easily.
  if (path.empty()) return true;
  if (!EnsureParentDirectory(path, err)) return false;
  std::ofstream fout(path);
  if (!fout) {
    if (err) *err = "Failed to open output file: " + path;
    return false;
  }
  for (NodeId u : nodes) fout << u << "\n";
  return true;
}

bool SaveLog(const std::string& path,
             const RuntimeOptions& opts,
             const DcgsResult& result,
             std::string* err) {
  // The text log is the human-readable experiment artifact.
  if (path.empty()) return true;
  if (!EnsureParentDirectory(path, err)) return false;

  std::ofstream fout(path);
  if (!fout) {
    if (err) *err = "Failed to open log file: " + path;
    return false;
  }

  WriteSummary(fout, opts, result);
  WriteAggregateText(fout, result);
  WriteTrialSummariesText(fout, result);
  WriteCheckpointTraceText(fout, result);
  WriteFinalExactTraceText(fout, result);
  return true;
}

bool SaveCheckpointTraceCsv(const std::string& path,
                            const DcgsResult& result,
                            std::string* err) {
  // CSV is intended for direct import into spreadsheets or plotting scripts.
  if (path.empty()) return true;
  if (!EnsureParentDirectory(path, err)) return false;

  std::ofstream fout(path);
  if (!fout) {
    if (err) *err = "Failed to open checkpoint CSV file: " + path;
    return false;
  }

  fout << "checkpoint_index,t,radius,rho_tilde,rho_lower,rho_upper,u_hat_size,u_plus_size,opt_sec,exact_solve_calls,exact_flow_calls,exact_avg_flow_calls,exact_binary_search_iters,exact_avg_binary_search_iters,exact_topology_rebuilds,exact_capacity_scale_last,exact_capacity_scale_max,exact_solver_sec,exact_avg_solver_time_sec,early_stop\n";
  for (std::size_t i = 0; i < result.certified.checkpoint_trace.size(); ++i) {
    const CertifiedCheckpointTrace& trace = result.certified.checkpoint_trace[i];
    fout << i
         << "," << trace.t
         << "," << trace.radius
         << "," << trace.rho_tilde
         << "," << trace.rho_lower
         << "," << trace.rho_upper
         << "," << trace.u_hat_size
         << "," << trace.u_plus_size
         << "," << trace.opt_sec
         << "," << trace.exact_solve_calls
         << "," << trace.exact_flow_calls
         << "," << trace.exact_avg_flow_calls
         << "," << trace.exact_binary_search_iters
         << "," << trace.exact_avg_binary_search_iters
         << "," << trace.exact_topology_rebuilds
         << "," << trace.exact_capacity_scale_last
         << "," << trace.exact_capacity_scale_max
         << "," << trace.exact_solver_sec
         << "," << trace.exact_avg_solver_time_sec
         << "," << (trace.early_stop ? "true" : "false")
         << "\n";
  }
  return true;
}

bool SaveCheckpointTraceJson(const std::string& path,
                             const RuntimeOptions& opts,
                             const DcgsResult& result,
                             std::string* err) {
  // JSON preserves the full structured result, including aggregates and traces.
  if (path.empty()) return true;
  if (!EnsureParentDirectory(path, err)) return false;

  std::ofstream fout(path);
  if (!fout) {
    if (err) *err = "Failed to open checkpoint JSON file: " + path;
    return false;
  }

  const ProcessMemorySnapshot memory = ReadProcessMemorySnapshot();
  fout << "{\n";
  fout << "  \"trials\": " << result.aggregate.trials << ",\n";
  fout << "  \"representative_trial_index\": " << result.representative_trial_index << ",\n";
  fout << "  \"algorithm\": \"" << AlgorithmName(opts.algorithm) << "\",\n";
  fout << "  \"subproblem_solver\": \"" << SubproblemSolverName(opts.subproblem_solver)
       << "\",\n";
  fout << "  \"algorithm_family\": \"" << AlgorithmFamilyName(opts.algorithm) << "\",\n";
  fout << "  \"model\": \"" << ((opts.model == DiffusionModel::kIC) ? "ic" : "lt") << "\",\n";
  fout << "  \"pilot_used\": " << (UsesPilot(opts.algorithm) ? "true" : "false") << ",\n";
  fout << "  \"checkpointing_used\": "
       << (UsesCheckpointing(opts.algorithm) ? "true" : "false") << ",\n";
  fout << "  \"sample_cap\": " << opts.sample_cap << ",\n";
  fout << "  \"sample_cap_used\": " << result.sample_cap_used << ",\n";
  fout << "  \"eval_cap\": " << opts.eval_cap << ",\n";
  fout << "  \"eval_cap_used\": " << result.eval_cap_used << ",\n";
  fout << "  \"n_hat\": " << result.u_hat.size() << ",\n";
  fout << "  \"tracked_nodes\": "
       << TrackedNodeCount(result.pilot.tracked_graph) << ",\n";
  fout << "  \"rho_final\": " << result.certified.rho_final << ",\n";
  fout << "  \"rho_eval\": " << result.rho_eval << ",\n";
  fout << "  \"early_stop\": " << (result.certified.early_stop ? "true" : "false") << ",\n";
  fout << "  \"t_stop\": " << result.certified.t_stop << ",\n";
  fout << "  \"checkpoints\": " << result.certified.checkpoints << ",\n";
  fout << "  \"max_checkpoints\": " << result.certified.max_checkpoints << ",\n";
  fout << "  \"first_checkpoint_t\": " << result.certified.first_checkpoint_t << ",\n";
  fout << "  \"e_hat_tau_size\": "
       << TrackedEdgeCount(result.pilot.tracked_graph) << ",\n";
  fout << "  \"T0\": " << result.pilot.t0 << ",\n";
  fout << "  \"T_max\": " << result.certified.t_max << ",\n";
  fout << "  \"tau\": " << opts.tau << ",\n";
  fout << "  \"eps\": " << opts.eps << ",\n";
  fout << "  \"delta\": " << opts.delta << ",\n";
  fout << "  \"gamma\": " << opts.gamma << ",\n";
  fout << "  \"pilot_chunk_bytes\": " << opts.pilot_chunk_bytes << ",\n";
  fout << "  \"baseline_chunk_bytes\": " << opts.baseline_chunk_bytes << ",\n";
  fout << "  \"pilot_time_sec\": " << result.pilot.elapsed_sec << ",\n";
  fout << "  \"sampling_time_sec\": " << result.certified.sampling_sec << ",\n";
  fout << "  \"checkpoint_opt_time_sec\": " << result.certified.checkpoint_opt_sec << ",\n";
  fout << "  \"exact_solve_calls\": " << result.certified.exact_solve_calls << ",\n";
  fout << "  \"exact_flow_calls\": " << result.certified.exact_flow_calls << ",\n";
  fout << "  \"exact_avg_flow_calls\": "
       << AverageExactFlowCalls(result.certified) << ",\n";
  fout << "  \"exact_binary_search_iters\": "
       << result.certified.exact_binary_search_iters << ",\n";
  fout << "  \"exact_avg_binary_search_iters\": "
       << AverageExactBinarySearchIters(result.certified) << ",\n";
  fout << "  \"exact_topology_rebuilds\": "
       << result.certified.exact_topology_rebuilds << ",\n";
  fout << "  \"exact_capacity_scale_last\": "
       << result.certified.exact_capacity_scale_last << ",\n";
  fout << "  \"exact_capacity_scale_max\": "
       << result.certified.exact_capacity_scale_max << ",\n";
  fout << "  \"exact_solver_time_sec\": " << result.certified.exact_solver_sec << ",\n";
  fout << "  \"exact_avg_solver_time_sec\": "
       << AverageExactSolverTime(result.certified) << ",\n";
  fout << "  \"final_exact_trace_used\": "
       << (result.certified.final_exact_trace.used ? "true" : "false") << ",\n";
  fout << "  \"eval_time_sec\": " << result.eval_time_sec << ",\n";
  fout << "  \"memory_stats_available\": "
       << (memory.available ? "true" : "false") << ",\n";
  fout << "  \"memory_rss_bytes\": " << memory.rss_bytes << ",\n";
  fout << "  \"memory_rss_mib\": " << BytesToMiB(memory.rss_bytes) << ",\n";
  fout << "  \"memory_peak_rss_bytes\": " << memory.peak_rss_bytes << ",\n";
  fout << "  \"memory_peak_rss_mib\": " << BytesToMiB(memory.peak_rss_bytes) << ",\n";
  fout << "  \"trial_aggregate\": {\n";
  fout << "    \"early_stop_count\": " << result.aggregate.early_stop_count << ",\n";
  fout << "    \"early_stop_rate\": " << result.aggregate.early_stop_rate << ",\n";
  auto write_metric_json = [&](const char* name,
                               const TrialAggregateMetric& metric,
                               bool trailing_comma) {
    fout << "    \"" << name << "\": {\"mean\": " << metric.mean
         << ", \"min\": " << metric.min
         << ", \"max\": " << metric.max << "}"
         << (trailing_comma ? "," : "") << "\n";
  };
  write_metric_json("rho_final", result.aggregate.rho_final, true);
  write_metric_json("rho_eval", result.aggregate.rho_eval, true);
  write_metric_json("u_hat_size", result.aggregate.u_hat_size, true);
  write_metric_json("tracked_nodes", result.aggregate.tracked_nodes, true);
  write_metric_json("e_hat_tau_size", result.aggregate.e_hat_tau_size, true);
  write_metric_json("sample_cap_used", result.aggregate.sample_cap_used, true);
  write_metric_json("eval_cap_used", result.aggregate.eval_cap_used, true);
  write_metric_json("t0", result.aggregate.t0, true);
  write_metric_json("t_stop", result.aggregate.t_stop, true);
  write_metric_json("t_max", result.aggregate.t_max, true);
  write_metric_json("checkpoints", result.aggregate.checkpoints, true);
  write_metric_json("pilot_time_sec", result.aggregate.pilot_time_sec, true);
  write_metric_json("sampling_sec", result.aggregate.sampling_sec, true);
  write_metric_json("checkpoint_opt_sec", result.aggregate.checkpoint_opt_sec, true);
  write_metric_json("exact_solve_calls", result.aggregate.exact_solve_calls, true);
  write_metric_json("exact_flow_calls", result.aggregate.exact_flow_calls, true);
  write_metric_json("exact_avg_flow_calls", result.aggregate.exact_avg_flow_calls,
                    true);
  write_metric_json("exact_binary_search_iters",
                    result.aggregate.exact_binary_search_iters, true);
  write_metric_json("exact_avg_binary_search_iters",
                    result.aggregate.exact_avg_binary_search_iters, true);
  write_metric_json("exact_topology_rebuilds",
                    result.aggregate.exact_topology_rebuilds, true);
  write_metric_json("exact_solver_sec", result.aggregate.exact_solver_sec, true);
  write_metric_json("exact_avg_solver_time_sec",
                    result.aggregate.exact_avg_solver_time_sec, true);
  write_metric_json("eval_time_sec", result.aggregate.eval_time_sec, true);
  write_metric_json("total_time_sec", result.aggregate.total_time_sec, false);
  fout << "  },\n";
  fout << "  \"trial_summaries\": [\n";
  for (std::size_t i = 0; i < result.trial_summaries.size(); ++i) {
    const TrialSummary& summary = result.trial_summaries[i];
    fout << "    {\n";
    fout << "      \"trial_index\": " << summary.trial_index << ",\n";
    fout << "      \"trial_seed\": " << summary.trial_seed << ",\n";
    fout << "      \"sample_cap_used\": " << summary.sample_cap_used << ",\n";
    fout << "      \"eval_cap_used\": " << summary.eval_cap_used << ",\n";
    fout << "      \"u_hat_size\": " << summary.u_hat_size << ",\n";
    fout << "      \"tracked_nodes\": " << summary.tracked_nodes << ",\n";
    fout << "      \"e_hat_tau_size\": " << summary.e_hat_tau_size << ",\n";
    fout << "      \"t0\": " << summary.t0 << ",\n";
    fout << "      \"t_stop\": " << summary.t_stop << ",\n";
    fout << "      \"t_max\": " << summary.t_max << ",\n";
    fout << "      \"checkpoints\": " << summary.checkpoints << ",\n";
    fout << "      \"rho_final\": " << summary.rho_final << ",\n";
    fout << "      \"rho_eval\": " << summary.rho_eval << ",\n";
    fout << "      \"early_stop\": " << (summary.early_stop ? "true" : "false") << ",\n";
    fout << "      \"pilot_time_sec\": " << summary.pilot_time_sec << ",\n";
    fout << "      \"sampling_sec\": " << summary.sampling_sec << ",\n";
    fout << "      \"checkpoint_opt_sec\": " << summary.checkpoint_opt_sec << ",\n";
    fout << "      \"exact_solve_calls\": " << summary.exact_solve_calls << ",\n";
    fout << "      \"exact_flow_calls\": " << summary.exact_flow_calls << ",\n";
    fout << "      \"exact_avg_flow_calls\": " << summary.exact_avg_flow_calls
         << ",\n";
    fout << "      \"exact_binary_search_iters\": "
         << summary.exact_binary_search_iters << ",\n";
    fout << "      \"exact_avg_binary_search_iters\": "
         << summary.exact_avg_binary_search_iters << ",\n";
    fout << "      \"exact_topology_rebuilds\": "
         << summary.exact_topology_rebuilds << ",\n";
    fout << "      \"exact_capacity_scale_last\": "
         << summary.exact_capacity_scale_last << ",\n";
    fout << "      \"exact_capacity_scale_max\": "
         << summary.exact_capacity_scale_max << ",\n";
    fout << "      \"exact_solver_sec\": " << summary.exact_solver_sec << ",\n";
    fout << "      \"exact_avg_solver_time_sec\": "
         << summary.exact_avg_solver_time_sec << ",\n";
    fout << "      \"eval_time_sec\": " << summary.eval_time_sec << ",\n";
    fout << "      \"total_time_sec\": " << summary.total_time_sec << "\n";
    fout << "    }" << ((i + 1 == result.trial_summaries.size()) ? "\n" : ",\n");
  }
  fout << "  ],\n";
  fout << "  \"final_exact_trace\": ";
  if (!result.certified.final_exact_trace.used) {
    fout << "null,\n";
  } else {
    const CertifiedFinalExactTrace& trace = result.certified.final_exact_trace;
    fout << "{\n";
    fout << "    \"t\": " << trace.t << ",\n";
    fout << "    \"u_hat_size\": " << trace.u_hat_size << ",\n";
    fout << "    \"rho\": " << trace.rho << ",\n";
    fout << "    \"opt_sec\": " << trace.opt_sec << ",\n";
    fout << "    \"exact_solve_calls\": " << trace.exact_solve_calls << ",\n";
    fout << "    \"exact_flow_calls\": " << trace.exact_flow_calls << ",\n";
    fout << "    \"exact_avg_flow_calls\": " << trace.exact_avg_flow_calls << ",\n";
    fout << "    \"exact_binary_search_iters\": " << trace.exact_binary_search_iters << ",\n";
    fout << "    \"exact_avg_binary_search_iters\": "
         << trace.exact_avg_binary_search_iters << ",\n";
    fout << "    \"exact_topology_rebuilds\": " << trace.exact_topology_rebuilds << ",\n";
    fout << "    \"exact_capacity_scale_last\": " << trace.exact_capacity_scale_last << ",\n";
    fout << "    \"exact_capacity_scale_max\": " << trace.exact_capacity_scale_max << ",\n";
    fout << "    \"exact_solver_sec\": " << trace.exact_solver_sec << ",\n";
    fout << "    \"exact_avg_solver_time_sec\": " << trace.exact_avg_solver_time_sec << "\n";
    fout << "  },\n";
  }
  fout << "  \"checkpoint_trace\": [\n";
  for (std::size_t i = 0; i < result.certified.checkpoint_trace.size(); ++i) {
    const CertifiedCheckpointTrace& trace = result.certified.checkpoint_trace[i];
    fout << "    {\n";
    fout << "      \"checkpoint_index\": " << i << ",\n";
    fout << "      \"t\": " << trace.t << ",\n";
    fout << "      \"radius\": " << trace.radius << ",\n";
    fout << "      \"rho_tilde\": " << trace.rho_tilde << ",\n";
    fout << "      \"rho_lower\": " << trace.rho_lower << ",\n";
    fout << "      \"rho_upper\": " << trace.rho_upper << ",\n";
    fout << "      \"u_hat_size\": " << trace.u_hat_size << ",\n";
    fout << "      \"u_plus_size\": " << trace.u_plus_size << ",\n";
    fout << "      \"opt_sec\": " << trace.opt_sec << ",\n";
    fout << "      \"exact_solve_calls\": " << trace.exact_solve_calls << ",\n";
    fout << "      \"exact_flow_calls\": " << trace.exact_flow_calls << ",\n";
    fout << "      \"exact_avg_flow_calls\": " << trace.exact_avg_flow_calls << ",\n";
    fout << "      \"exact_binary_search_iters\": "
         << trace.exact_binary_search_iters << ",\n";
    fout << "      \"exact_avg_binary_search_iters\": "
         << trace.exact_avg_binary_search_iters << ",\n";
    fout << "      \"exact_topology_rebuilds\": "
         << trace.exact_topology_rebuilds << ",\n";
    fout << "      \"exact_capacity_scale_last\": "
         << trace.exact_capacity_scale_last << ",\n";
    fout << "      \"exact_capacity_scale_max\": "
         << trace.exact_capacity_scale_max << ",\n";
    fout << "      \"exact_solver_sec\": " << trace.exact_solver_sec << ",\n";
    fout << "      \"exact_avg_solver_time_sec\": "
         << trace.exact_avg_solver_time_sec << ",\n";
    fout << "      \"early_stop\": " << (trace.early_stop ? "true" : "false") << "\n";
    fout << "    }" << ((i + 1 == result.certified.checkpoint_trace.size()) ? "\n" : ",\n");
  }
  fout << "  ]\n";
  fout << "}\n";
  return true;
}

void PrintSummary(const RuntimeOptions& opts, const DcgsResult& result) {
  // Stdout mirrors the text log so quick command-line runs show the same
  // headline metrics without opening output files.
  WriteSummary(std::cout, opts, result);
  WriteAggregateText(std::cout, result);
  WriteCheckpointTraceText(std::cout, result);
  WriteFinalExactTraceText(std::cout, result);
}

}  // namespace dcgs
