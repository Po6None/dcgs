#ifndef RHO_ESTIMATOR_H_
#define RHO_ESTIMATOR_H_

#include "common.h"
#include "diffusion_simulator.h"
#include "graph.h"

#include <cstdint>
#include <random>
#include <vector>

namespace dcgs {

// CountedRhoRun stores one counted run of pair occurrences for rho evaluation.
struct CountedRhoRun {
  std::vector<std::uint64_t> keys;
  std::vector<SampleCount> counts;
};

// RhoEstimationResult summarizes the sampled estimate of rho(U).
struct RhoEstimationResult {
  Real rho_hat = 0.0;
  SampleCount eval_cap_used = 0;
  Real elapsed_sec = 0.0;
};

// RhoEstimationWorkspace owns reusable buffers for repeated rho(U) evaluation
// runs across trials and algorithms.
struct RhoEstimationWorkspace {
  DiffusionWorkspace diffusion_workspace;

  std::vector<std::uint32_t> candidate_mark;
  std::uint32_t candidate_epoch = 1;
  std::vector<NodeId> filtered_active_nodes;

  std::vector<std::uint64_t> pair_buffer;
  std::vector<std::uint64_t> chunk_buffer;

  std::vector<std::uint64_t> global_keys;
  std::vector<SampleCount> global_counts;

  std::vector<CountedRhoRun> run_levels;
  std::vector<std::uint64_t> carry_keys;
  std::vector<SampleCount> carry_counts;
  std::vector<std::uint64_t> temp_keys;
  std::vector<SampleCount> temp_counts;
};

// EstimateRhoBySampling estimates rho(U) by sampling eval_cap realizations and
// counting thresholded co-activated pairs inside nodes.
RhoEstimationResult EstimateRhoBySampling(const Graph& graph,
                                          const std::vector<NodeId>& seeds,
                                          const std::vector<NodeId>& nodes,
                                          SampleCount eval_cap,
                                          const RuntimeOptions& opts,
                                          RhoEstimationWorkspace* workspace,
                                          std::mt19937_64* rng);

// Convenience overload that uses a temporary RhoEstimationWorkspace.
RhoEstimationResult EstimateRhoBySampling(const Graph& graph,
                                          const std::vector<NodeId>& seeds,
                                          const std::vector<NodeId>& nodes,
                                          SampleCount eval_cap,
                                          const RuntimeOptions& opts,
                                          std::mt19937_64* rng);

}  // namespace dcgs

#endif  // RHO_ESTIMATOR_H_
