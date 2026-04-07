#ifndef PROGRESS_H_
#define PROGRESS_H_

#include "common.h"

#include <string>

namespace dcgs {

// PrintProgressLine emits one human-readable runtime progress message when the
// user has enabled progress reporting.
void PrintProgressLine(const RuntimeOptions& opts,
                       const std::string& stage,
                       const std::string& message);

// ProgressReportInterval chooses a bounded reporting cadence so large runs stay
// informative without flooding the terminal.
SampleCount ProgressReportInterval(SampleCount total);

// ShouldReportProgress advances the caller-owned next_report threshold and
// returns true only when the current counter should be printed.
bool ShouldReportProgress(SampleCount current,
                          SampleCount total,
                          SampleCount* next_report);

// FormatProgressFraction renders one "current/total (percent%)" fragment used
// by pilot and certified progress messages.
std::string FormatProgressFraction(SampleCount current, SampleCount total);

}  // namespace dcgs

#endif  // PROGRESS_H_
