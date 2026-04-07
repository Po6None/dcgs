CXX := g++
CXXFLAGS := -std=c++17 -O3 -Wall -Wextra -pedantic
SOURCES := main.cpp graph.cpp graph_binary.cpp progress.cpp resource_usage.cpp tracked_graph.cpp diffusion_simulator.cpp pilot.cpp peeling.cpp certified.cpp baseline.cpp rho_estimator.cpp io.cpp output.cpp dcgs_pipeline.cpp exact_solver.cpp
TEST_SOURCES := tiny_tests.cpp graph.cpp graph_binary.cpp progress.cpp resource_usage.cpp seed_generator.cpp tracked_graph.cpp diffusion_simulator.cpp pilot.cpp peeling.cpp certified.cpp baseline.cpp rho_estimator.cpp io.cpp output.cpp dcgs_pipeline.cpp exact_solver.cpp
HEADERS := common.h dcgs.hpp graph.h graph_binary.h progress.h resource_usage.h seed_generator.h tracked_graph.h diffusion_simulator.h pilot.h peeling.h certified.h baseline.h rho_estimator.h io.h output.h dcgs_pipeline.h exact_solver.h

.PHONY: all test clean run-tiny run-facebook prep-facebook-bin run-facebook-bin

all: dcgs

dcgs: $(SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(SOURCES) -o dcgs

graph_convert: graph_convert.cpp graph_binary.cpp graph_binary.h common.h
	$(CXX) $(CXXFLAGS) graph_convert.cpp graph_binary.cpp -o graph_convert

seed_generate: seed_generate.cpp seed_generator.cpp graph_binary.cpp seed_generator.h graph_binary.h common.h
	$(CXX) $(CXXFLAGS) seed_generate.cpp seed_generator.cpp graph_binary.cpp -o seed_generate

test: tiny_tests
	./tiny_tests

tiny_tests: $(TEST_SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(TEST_SOURCES) -o tiny_tests

run-tiny: dcgs
	./dcgs --graph examples/tiny/graph.txt --seeds examples/tiny/seeds.txt --model ic --ic_p_rule const --p 1.0 --tau 0.1 --eps 0.2 --delta 0.2 --gamma 1.0 --rng_seed 1 --out tiny_output_u_hat.txt --log tiny_output_log.txt --trace_csv tiny_output_trace.csv --trace_json tiny_output_trace.json

run-facebook: dcgs
	./scripts/run_facebook.sh --text

prep-facebook-bin: graph_convert
	./graph_convert --in dataset/facebook/facebook.txt --out dataset/facebook/graph.dcgsbin --undirected

run-facebook-bin: dcgs
	./scripts/run_facebook.sh --binary

clean:
	rm -f dcgs graph_convert seed_generate tiny_tests
