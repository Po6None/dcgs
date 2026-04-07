#ifndef IO_H_
#define IO_H_

#include "common.h"
#include "graph.h"

#include <vector>

namespace dcgs {

// ParseCommandLine fills RuntimeOptions from CLI flags without performing the
// full semantic validation step.
bool ParseCommandLine(int argc, char** argv, RuntimeOptions* opts, std::string* err);

// ValidateOptions checks cross-flag constraints after parsing.
bool ValidateOptions(const RuntimeOptions& opts, std::string* err);

// LoadSeeds reads a text seed list, validates node ids, and removes duplicates.
bool LoadSeeds(const std::string& seeds_path, NodeCount n, std::vector<NodeId>* seeds, std::string* err);

}  // namespace dcgs

#endif  // IO_H_
