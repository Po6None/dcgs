#include "progress.h"

#include "resource_usage.h"

#include <algorithm>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace dcgs {

namespace {

std::string CurrentLocalTimestamp() {
  const auto now = std::chrono::system_clock::now();
  const std::time_t now_time = std::chrono::system_clock::to_time_t(now);
  std::tm local_tm{};
#if defined(_WIN32)
  localtime_s(&local_tm, &now_time);
#else
  localtime_r(&now_time, &local_tm);
#endif

  std::ostringstream out;
  out << std::put_time(&local_tm, "%Y-%m-%d %H:%M:%S");
  return out.str();
}

}  // namespace

void PrintProgressLine(const RuntimeOptions& opts,
                       const std::string& stage,
                       const std::string& message) {
  // Progress is written to stderr so final stdout summaries remain machine
  // readable and shell redirection can separate runtime logs from results.
  if (!opts.progress) return;
  const ProcessMemorySnapshot snapshot = ReadProcessMemorySnapshot();
  std::cerr << "[" << stage << "]"
            << "[" << CurrentLocalTimestamp() << "] "
            << message << FormatMemorySuffix(snapshot) << "\n";
}

SampleCount ProgressReportInterval(SampleCount total) {
  // Large stages report at most about twenty periodic updates. Small stages can
  // still report each step so debugging tiny runs stays straightforward.
  if (total <= 20) return 1;
  return std::max<SampleCount>(1, total / 20);
}

bool ShouldReportProgress(SampleCount current,
                          SampleCount total,
                          SampleCount* next_report) {
  if (!next_report || total == 0 || current == 0) return false;
  if (current >= total) return true;
  if (current < *next_report) return false;

  const SampleCount step = ProgressReportInterval(total);
  while (*next_report <= current) {
    *next_report = std::min(total, *next_report + step);
    if (*next_report == total) break;
  }
  return true;
}

std::string FormatProgressFraction(SampleCount current, SampleCount total) {
  std::ostringstream out;
  out << current << "/" << total;
  if (total > 0) {
    const Real percent =
        100.0 * static_cast<Real>(current) / static_cast<Real>(total);
    out << " (" << std::fixed << std::setprecision(1) << percent << "%)";
  }
  return out.str();
}

}  // namespace dcgs
