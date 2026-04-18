# DCGS

This repository provides a clean and runnable C++17 implementation of DCGS and the main baseline methods used in our experiments, including the `pilot_peeling_cap` ablation baseline.

## 1) What is included

This package includes the complete C++17 source code for the `dcgs` executable,
the build configuration in `Makefile`, and two utility tools:
`graph_convert` (text edge list to `.dcgsbin`) and `seed_generate`
(seed-set generation). A runnable example dataset is provided under
`dataset/hamsterster/`, including `hamsterster.txt`, `graph.dcgsbin`,
and `seeds_top_outdeg_10.txt`, so the project can be compiled and executed
directly after cloning.

## 2) Code layout

| Path(s) | Purpose |
|---|---|
| `main.cpp` | CLI entry and algorithm dispatch |
| `dcgs_pipeline.*` | End-to-end DCGS pipeline orchestration |
| `graph.*`, `graph_binary.*`, `tracked_graph.*` | Graph loading and tracked subgraph |
| `diffusion_simulator.*` | IC / LT realization simulation |
| `pilot.*`, `certified.*` | Pilot stage and certified stage |
| `peeling.*`, `exact_solver.*` | Peeling solver and min-cut-based exact solver |
| `baseline.*`, `rho_estimator.*` | Baseline methods and final objective estimation |
| `io.*`, `output.*`, `progress.*`, `resource_usage.*` | Input, output, progress, memory reporting |
| `graph_convert.cpp` | Text edge list to binary `.dcgsbin` tool |
| `seed_generate.cpp`, `seed_generator.*` | Seed set generation tool |

## 3) Build

| Command | Output binary |
|---|---|
| `make` or `make dcgs` | `./dcgs` |
| `make graph_convert` | `./graph_convert` |
| `make seed_generate` | `./seed_generate` |
| `make clean` | Remove built binaries |

## 4) Quick start (included hamsterster example)

Build the main executable:

```bash
make dcgs
```

Run DCGS on the included hamsterster example (LT + peeling):

```bash
./dcgs --graph dataset/hamsterster/graph.dcgsbin --seeds dataset/hamsterster/seeds_top_outdeg_10.txt --algorithm dcgs --subproblem_solver peeling --model lt --lt_w_rule c_over_indeg --c 0.5 --tau 0.2 --eps 0.5 --delta 0.001 --gamma 0.0001 --rng_seed 1 --trials 1 --eval_cap 10000 --out result/output_u_hat.txt --log result/output_log.txt --trace_csv result/output_trace.csv --trace_json result/output_trace.json --progress
```

Run the `pilot_peeling_cap` ablation baseline on the same graph:

```bash
./dcgs --graph dataset/hamsterster/graph.dcgsbin --seeds dataset/hamsterster/seeds_top_outdeg_10.txt --algorithm pilot_peeling_cap --model lt --lt_w_rule c_over_indeg --c 0.5 --tau 0.2 --eps 0.5 --delta 0.001 --gamma 0.0001 --sample_cap 10000 --eval_cap 10000 --rng_seed 1 --trials 1 --out result/output_u_hat.txt --log result/output_log.txt --trace_csv result/output_trace.csv --trace_json result/output_trace.json --progress
```

Show all available CLI options:

```bash
./dcgs --help
```

## 5) Main CLI options

| Option | Type | Default | Description |
|---|---|---|---|
| `--graph` | `string` (path) | Required | Input graph file (`.txt` or `.dcgsbin`) |
| `--seeds` | `string` (path) | Required | Seed node file (one node id per line) |
| `--undirected` | flag (`bool`) | `false` | Treat text input as undirected (insert both directions) |
| `--algorithm` | enum (`string`) | `dcgs` | `dcgs`, `best_pair`, `topk_activation_prob`, `topk_coactivation_degree`, `greedy_add`, `local_search`, `structure_peeling`, `edge_coact_peeling`, `certified_cap_no_early_stop`, `pilot_peeling_cap` |
| `--subproblem_solver` | enum (`string`) | `peeling` | `peeling` or `exact` (used by `dcgs` and `certified_cap_no_early_stop`) |
| `--model` | enum (`string`) | `ic` | `ic` or `lt` |
| `--ic_p_rule` | enum (`string`) | `const` | `const`, `c_over_indeg`, `file` |
| `--p` | `float` | `0.01` | Constant IC edge probability when `--ic_p_rule const` |
| `--c` | `float` | `1.0` | Coefficient for `c_over_indeg` rule |
| `--lt_w_rule` | enum (`string`) | `uniform_in` | `uniform_in`, `c_over_indeg`, `file` |
| `--lt_normalize_in` | flag (`bool`) | `false` | Normalize incoming LT weights after loading |
| `--tau` | `float` | `0.05` | Threshold parameter, must be in `(0, 1)` |
| `--eps` | `float` | `0.1` | Approximation parameter, must be `> 0` |
| `--delta` | `float` | `0.1` | Failure probability parameter, must be in `(0, 1)` |
| `--gamma` | `float` | `1e-4` | Accuracy parameter, must be `> 0` |
| `--rng_seed` | `uint64` | `1` | Base random seed |
| `--trials` | `uint32` | `1` | Number of full runs, must be `>= 1` |
| `--sample_cap` | `uint64` | `0` | Sampling cap for sample-budget methods |
| `--eval_cap` | `uint64` | `0` | Number of realizations for final `rho(U)` estimation |
| `--pilot_chunk_bytes` | `uint64` (bytes) | `268435456` | Pilot buffer budget (default 256 MiB, min 8) |
| `--baseline_chunk_bytes` | `uint64` (bytes) | `268435456` | Baseline pair-buffer budget (default 256 MiB, min 8) |
| `--local_search_max_iters` | `uint32` | `1000` | Max improvement rounds, must be `>= 1` |
| `--local_search_swap_candidates` | `uint32` | `32` | Swap candidate count (`0` disables swap) |
| `--out` | `string` (path) | `result/output_u_hat.txt` | Output node list (`U_hat`) |
| `--log` | `string` (path) | `result/output_log.txt` | Text summary/log output |
| `--trace_csv` | `string` (path) | `result/output_trace.csv` | Checkpoint trace in CSV |
| `--trace_json` | `string` (path) | `result/output_trace.json` | Checkpoint trace in JSON |
| `--progress` | flag (`bool`) | `false` | Print runtime progress to stderr |

## 6) Utilities

`graph_convert` converts a text edge list into a binary `.dcgsbin` graph file.

```bash
./graph_convert --in dataset/hamsterster/hamsterster.txt --out dataset/hamsterster/graph.dcgsbin --undirected
```

`seed_generate` generates seed sets (for example, top-out-degree seeds).

```bash
./seed_generate --graph dataset/hamsterster/hamsterster.txt --strategy top_outdeg --k 10 --out dataset/hamsterster/seeds_top_outdeg_10.txt --undirected
```

## 7) Notes

This project is built with `g++` using `-std=c++17 -O3`. By default, output
files are written under `result/`, and missing parent directories are created
automatically. Node IDs are assumed to be 0-based contiguous integers after
preprocessing.
