#include "resource_usage.h"

#include <fstream>
#include <sstream>
#include <string>

namespace dcgs {

namespace {

// ParseStatusKilobytes extracts one "VmRSS: <n> kB" style field from
// /proc/self/status.
bool ParseStatusKilobytes(const std::string& line,
                          const std::string& key,
                          std::uint64_t* bytes) {
  if (!bytes) return false;
  if (line.compare(0, key.size(), key) != 0) return false;

  std::istringstream iss(line.substr(key.size()));
  std::uint64_t kib = 0;
  std::string unit;
  if (!(iss >> kib >> unit)) return false;
  if (unit != "kB") return false;
  *bytes = kib * 1024ull;
  return true;
}

}  // namespace

ProcessMemorySnapshot ReadProcessMemorySnapshot() {
  ProcessMemorySnapshot snapshot;
#if defined(__linux__)
  std::ifstream fin("/proc/self/status");
  if (!fin) return snapshot;

  std::string line;
  while (std::getline(fin, line)) {
    ParseStatusKilobytes(line, "VmRSS:", &snapshot.rss_bytes);
    ParseStatusKilobytes(line, "VmHWM:", &snapshot.peak_rss_bytes);
  }
  snapshot.available = (snapshot.rss_bytes > 0 || snapshot.peak_rss_bytes > 0);
#endif
  return snapshot;
}

Real BytesToMiB(std::uint64_t bytes) {
  return static_cast<Real>(bytes) / (1024.0 * 1024.0);
}

std::string FormatMemorySuffix(const ProcessMemorySnapshot& snapshot) {
  if (!snapshot.available) return "";
  std::ostringstream out;
  out << " rss_mib=" << BytesToMiB(snapshot.rss_bytes)
      << " peak_rss_mib=" << BytesToMiB(snapshot.peak_rss_bytes);
  return out.str();
}

}  // namespace dcgs
