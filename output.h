#ifndef OUTPUT_H_
#define OUTPUT_H_

#include "common.h"

#include <string>
#include <vector>

namespace dcgs {

struct DcgsResult;

// SaveNodeList writes the selected node set to a plain-text file.
bool SaveNodeList(const std::string& path,
                  const std::vector<NodeId>& nodes,
                  std::string* err);

// SaveLog writes the textual experiment summary plus traces.
bool SaveLog(const std::string& path,
             const RuntimeOptions& opts,
             const DcgsResult& result,
             std::string* err);

// SaveCheckpointTraceCsv writes checkpoint-level trace data in CSV format.
bool SaveCheckpointTraceCsv(const std::string& path,
                            const DcgsResult& result,
                            std::string* err);

// SaveCheckpointTraceJson writes the full structured result in JSON format.
bool SaveCheckpointTraceJson(const std::string& path,
                             const RuntimeOptions& opts,
                             const DcgsResult& result,
                             std::string* err);

// PrintSummary writes the human-readable result summary to stdout.
void PrintSummary(const RuntimeOptions& opts, const DcgsResult& result);

}  // namespace dcgs

#endif  // OUTPUT_H_
