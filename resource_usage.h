#ifndef RESOURCE_USAGE_H_
#define RESOURCE_USAGE_H_

#include "common.h"

#include <cstdint>
#include <string>

namespace dcgs {

// ProcessMemorySnapshot is a lightweight runtime memory reading intended for
// progress logs and final summaries on Linux hosts.
struct ProcessMemorySnapshot {
  bool available = false;
  std::uint64_t rss_bytes = 0;
  std::uint64_t peak_rss_bytes = 0;
};

// ReadProcessMemorySnapshot reads current RSS and peak RSS from the current
// process when the host platform exposes the required information.
ProcessMemorySnapshot ReadProcessMemorySnapshot();

// BytesToMiB is the shared formatter helper for human-readable memory output.
Real BytesToMiB(std::uint64_t bytes);

// FormatMemorySuffix returns one compact "rss/peak" fragment appended to
// runtime progress messages.
std::string FormatMemorySuffix(const ProcessMemorySnapshot& snapshot);

}  // namespace dcgs

#endif  // RESOURCE_USAGE_H_
