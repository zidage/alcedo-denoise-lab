// Measure UtNet2 in ONNX Runtime on macOS through the C++ API: precision against the DN0
// expected outputs and the time of a full photo with the DN1 tile planner.
//
// One process measures one execution provider configuration, so that the peak memory
// footprint of the process belongs to that configuration.

#include <mach/mach.h>
#import <Foundation/Foundation.h>

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {

constexpr int kAlignment = 16;
constexpr double kTolerance = 1e-5;

using Clock = std::chrono::steady_clock;

double Seconds(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double>(b - a).count();
}

struct Footprint {
  uint64_t current = 0;
  uint64_t peak = 0;
};

Footprint ReadFootprint() {
  task_vm_info_data_t info{};
  mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
  if (task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&info), &count) !=
      KERN_SUCCESS) {
    return {};
  }
  return {info.phys_footprint, static_cast<uint64_t>(info.ledger_phys_footprint_peak)};
}

// 0 nominal, 1 fair, 2 serious, 3 critical (NSProcessInfoThermalState).
int ThermalState() { return static_cast<int>([[NSProcessInfo processInfo] thermalState]); }

// --- Arrays -----------------------------------------------------------------------------------

struct Array {
  int c = 0, h = 0, w = 0;
  std::vector<float> data;
  size_t size() const { return static_cast<size_t>(c) * h * w; }
};

Array ReadArray(const std::string& path, int c, int h, int w) {
  Array a{c, h, w, {}};
  a.data.resize(a.size());
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open " + path);
  in.read(reinterpret_cast<char*>(a.data.data()), static_cast<std::streamsize>(a.size() * 4));
  if (static_cast<size_t>(in.gcount()) != a.size() * 4) {
    throw std::runtime_error("short read " + path);
  }
  return a;
}

struct ManifestEntry {
  std::string name, kind, file;
  int c, h, w;
};

std::vector<ManifestEntry> ReadManifest(const std::string& dir) {
  std::ifstream in(dir + "/manifest.txt");
  if (!in) throw std::runtime_error("cannot open manifest in " + dir);
  std::vector<ManifestEntry> entries;
  ManifestEntry e;
  while (in >> e.name >> e.kind >> e.c >> e.h >> e.w >> e.file) entries.push_back(e);
  return entries;
}

const ManifestEntry& Find(const std::vector<ManifestEntry>& m, const std::string& name,
                          const std::string& kind) {
  for (const auto& e : m) {
    if (e.name == name && e.kind == kind) return e;
  }
  throw std::runtime_error("manifest has no " + name + " " + kind);
}

struct Stats {
  double relative_difference = 0;
  uint64_t non_finite = 0;
  double max_abs = 0;
};

Stats Compare(const float* result, const Array& expected) {
  double largest = 0, diff = 0;
  uint64_t non_finite = 0;
  for (size_t i = 0; i < expected.size(); ++i) {
    largest = std::max(largest, std::fabs(static_cast<double>(expected.data[i])));
    if (!std::isfinite(result[i])) {
      ++non_finite;
      continue;
    }
    diff = std::max(diff, std::fabs(static_cast<double>(result[i]) - expected.data[i]));
  }
  Stats s;
  s.relative_difference = non_finite ? INFINITY : diff / largest;
  s.non_finite = non_finite;
  s.max_abs = largest;
  return s;
}

// --- Planner (lab measurement_common.plan_axis, last tile shrunk) ----------------------------

struct AxisTile {
  int input_start, input_size, keep_start, keep_end;
};

int AlignUp(int v) { return (v + kAlignment - 1) / kAlignment * kAlignment; }

std::vector<AxisTile> PlanAxis(int length, int tile, int overlap) {
  if (tile % kAlignment || overlap < 0 || 2 * overlap >= tile) {
    throw std::runtime_error("tile must be a multiple of 16 and larger than 2 x overlap");
  }
  if (length <= tile) return {{0, AlignUp(length), 0, length}};
  const int kept = tile - 2 * overlap;
  const int count = std::max(2, 1 + (length - tile + kept - 1) / kept);
  std::vector<AxisTile> tiles{{0, tile, 0, tile - overlap}};
  for (int i = 1; i < count - 1; ++i) {
    const int start = tile - overlap + (i - 1) * kept;
    tiles.push_back({start - overlap, tile, start, start + kept});
  }
  const int keep_start = tiles.back().keep_end;
  const int last_input = keep_start - overlap;
  tiles.push_back({last_input, AlignUp(length - last_input), keep_start, length});
  return tiles;
}

// np.pad mode "reflect": mirror without a repeat of the edge sample.
int Reflect(int i, int n) {
  if (n == 1) return 0;
  const int period = 2 * (n - 1);
  i %= period;
  if (i < 0) i += period;
  return i < n ? i : period - i;
}

void TileInput(const Array& packed, const AxisTile& ty, const AxisTile& tx, float* out) {
  const int h = ty.input_size, w = tx.input_size;
  for (int c = 0; c < 4; ++c) {
    const float* plane = packed.data.data() + static_cast<size_t>(c) * packed.h * packed.w;
    for (int y = 0; y < h; ++y) {
      const int sy = Reflect(ty.input_start + y, packed.h);
      const float* row = plane + static_cast<size_t>(sy) * packed.w;
      float* dst = out + (static_cast<size_t>(c) * h + y) * w;
      const int direct = std::min(w, packed.w - tx.input_start);
      std::memcpy(dst, row + tx.input_start, sizeof(float) * direct);
      for (int x = direct; x < w; ++x) dst[x] = row[Reflect(tx.input_start + x, packed.w)];
    }
  }
}

// --- Sessions ---------------------------------------------------------------------------------

struct Config {
  std::string ep = "coreml";  // cpu, coreml, webgpu
  std::string coreml_format = "MLProgram";
  std::string coreml_units = "CPUAndGPU";
  std::string coreml_low_precision = "0";
  std::string coreml_static = "0";
  std::string coreml_specialization = "Default";
  std::string coreml_cache;
  std::string webgpu_lib;
  std::string strategy = "free";  // free, per_shape
  int threads = 4;
  bool disable_cpu_fallback = true;
};

std::vector<std::string> g_log;
bool g_capture_coreml = false;
// The graph has no DepthToSpace node and gives `output_packed` (1, 12, h, w).
bool g_packed_output = false;

void ORT_API_CALL LogCallback(void*, OrtLoggingLevel severity, const char* category,
                              const char*, const char* location, const char* message) {
  std::string m = message ? message : "";
  const std::string where = location ? location : "";
  // Keep the messages that say where the nodes run, and every warning or error. During the
  // probe session, also every message of the Core ML execution provider (its compute plan).
  const bool placement = m.find("supported by") != std::string::npos ||
                         m.find("partition") != std::string::npos ||
                         m.find("placed on") != std::string::npos ||
                         m.find("Node placements") != std::string::npos ||
                         m.find("not assigned") != std::string::npos ||
                         m.find("not supported") != std::string::npos;
  const bool coreml = g_capture_coreml && (where.find("coreml") != std::string::npos ||
                                           where.find("model.mm") != std::string::npos ||
                                           m.find("Device") != std::string::npos ||
                                           m.find("compute") != std::string::npos);
  if (placement || coreml || severity >= ORT_LOGGING_LEVEL_WARNING) {
    if (g_log.size() < 600) {
      g_log.push_back(std::string(category ? category : "") + " " + (location ? location : "") +
                      ": " + m.substr(0, 600));
    }
  }
}

class SessionSet {
 public:
  SessionSet(Ort::Env& env, const std::string& graph, const Config& cfg)
      : env_(env), graph_(graph), cfg_(cfg) {}

  struct Created {
    int h, w;
    bool free;
    double create_seconds, first_run_seconds;
  };

  Ort::Session& Get(int h, int w, int log_level = ORT_LOGGING_LEVEL_WARNING) {
    const auto key = cfg_.strategy == "free" ? std::make_pair(0, 0) : std::make_pair(h, w);
    auto it = sessions_.find(key);
    if (it != sessions_.end()) return *it->second;
    const auto t0 = Clock::now();
    auto session = std::make_unique<Ort::Session>(env_, graph_.c_str(),
                                                  Options(key.first, key.second, log_level));
    const auto t1 = Clock::now();
    std::vector<float> zeros(static_cast<size_t>(4) * h * w, 0.0f);
    std::vector<float> out(static_cast<size_t>(3) * 4 * h * w);
    Run(*session, zeros.data(), out.data(), h, w);
    const auto t2 = Clock::now();
    created_.push_back({h, w, key.first == 0, Seconds(t0, t1), Seconds(t1, t2)});
    auto& ref = *session;
    sessions_.emplace(key, std::move(session));
    return ref;
  }

  static void Run(Ort::Session& s, float* input, float* output, int h, int w) {
    static const Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    const int64_t in_shape[] = {1, 4, h, w};
    const int64_t out_shape[] = {1, 3, 2 * h, 2 * w};
    const int64_t packed_shape[] = {1, 12, h, w};
    const size_t count = static_cast<size_t>(12) * h * w;
    Ort::Value in = Ort::Value::CreateTensor<float>(mem, input, static_cast<size_t>(4) * h * w,
                                                    in_shape, 4);
    const char* in_names[] = {"input"};
    if (!g_packed_output) {
      Ort::Value out = Ort::Value::CreateTensor<float>(mem, output, count, out_shape, 4);
      const char* out_names[] = {"output"};
      s.Run(Ort::RunOptions{nullptr}, in_names, &in, 1, out_names, &out, 1);
      return;
    }
    // Graph without its DepthToSpace node: PixelShuffle(2), mode CRD, on the host.
    thread_local std::vector<float> packed;
    packed.resize(count);
    Ort::Value out = Ort::Value::CreateTensor<float>(mem, packed.data(), count, packed_shape, 4);
    const char* out_names[] = {"output_packed"};
    s.Run(Ort::RunOptions{nullptr}, in_names, &in, 1, out_names, &out, 1);
    const size_t plane = static_cast<size_t>(h) * w;
    for (int c = 0; c < 3; ++c) {
      for (int y = 0; y < h; ++y) {
        for (int i = 0; i < 2; ++i) {
          const float* a = packed.data() + (4 * c + 2 * i) * plane + static_cast<size_t>(y) * w;
          const float* b = a + plane;
          float* dst = output + (static_cast<size_t>(c) * 2 * h + 2 * y + i) * 2 * w;
          for (int x = 0; x < w; ++x) {
            dst[2 * x] = a[x];
            dst[2 * x + 1] = b[x];
          }
        }
      }
    }
  }

  const std::vector<Created>& created() const { return created_; }
  void Clear() { sessions_.clear(); }

 private:
  Ort::SessionOptions Options(int h, int w, int log_level) {
    Ort::SessionOptions o;
    o.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    o.SetIntraOpNumThreads(cfg_.threads);
    o.SetInterOpNumThreads(1);
    o.AddConfigEntry("session.intra_op.allow_spinning", "0");
    o.DisableCpuMemArena();
    o.SetLogSeverityLevel(log_level);
    if (h > 0) {
      o.AddFreeDimensionOverrideByName("height", h);
      o.AddFreeDimensionOverrideByName("width", w);
    }
    if (cfg_.ep == "coreml") {
      std::unordered_map<std::string, std::string> p{
          {"ModelFormat", cfg_.coreml_format},
          {"MLComputeUnits", cfg_.coreml_units},
          {"RequireStaticInputShapes", cfg_.coreml_static},
          {"AllowLowPrecisionAccumulationOnGPU", cfg_.coreml_low_precision},
          {"SpecializationStrategy", cfg_.coreml_specialization}};
      if (!cfg_.coreml_cache.empty()) p["ModelCacheDirectory"] = cfg_.coreml_cache;
      // The probe session logs the Core ML compute plan: the device of each operation.
      if (log_level == ORT_LOGGING_LEVEL_VERBOSE) p["ProfileComputePlan"] = "1";
      o.AppendExecutionProvider("CoreML", p);
      if (cfg_.disable_cpu_fallback) o.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
    } else if (cfg_.ep == "webgpu_builtin") {
      // The WebGPU execution provider that the macOS package of ONNX Runtime contains.
      o.AppendExecutionProvider("WebGPU", std::unordered_map<std::string, std::string>{});
      if (cfg_.disable_cpu_fallback) o.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
    } else if (cfg_.ep == "webgpu") {
      // The package of ONNX Runtime has its own WebGPU execution provider with the same name.
      // The device of the registered plugin library comes after it in the device list.
      std::vector<Ort::ConstEpDevice> devices;
      for (const auto& d : env_.GetEpDevices()) {
        if (std::string(d.EpName()) == "WebGpuExecutionProvider") devices = {d};
      }
      if (devices.empty()) throw std::runtime_error("no WebGpuExecutionProvider device");
      o.AppendExecutionProvider_V2(env_, devices, std::unordered_map<std::string, std::string>{});
      if (cfg_.disable_cpu_fallback) o.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
    }
    return o;
  }

  Ort::Env& env_;
  std::string graph_;
  Config cfg_;
  std::map<std::pair<int, int>, std::unique_ptr<Ort::Session>> sessions_;
  std::vector<Created> created_;
};

// --- Tile loop --------------------------------------------------------------------------------

struct TiledRun {
  int tile_count = 0, tile_shapes = 0;
  uint64_t computed_packed_pixels = 0;
  double seconds_total = 0, seconds_run = 0;
};

TiledRun RunTiled(SessionSet& sessions, const Array& packed, int tile, int overlap,
                  std::vector<float>& output) {
  const auto ys = PlanAxis(packed.h, tile, overlap);
  const auto xs = PlanAxis(packed.w, tile, overlap);
  std::map<std::pair<int, int>, int> counts;
  for (const auto& ty : ys) {
    for (const auto& tx : xs) ++counts[{ty.input_size, tx.input_size}];
  }
  // Sessions of new shapes are made before the time starts, the most frequent shape first.
  std::vector<std::pair<int, std::pair<int, int>>> order;
  for (const auto& [shape, n] : counts) order.push_back({n, shape});
  std::sort(order.rbegin(), order.rend());
  for (const auto& [n, shape] : order) sessions.Get(shape.first, shape.second);

  const int oh = 2 * packed.h, ow = 2 * packed.w;
  output.resize(static_cast<size_t>(3) * oh * ow);
  std::vector<float> in(static_cast<size_t>(4) * tile * tile);
  std::vector<float> out(static_cast<size_t>(12) * tile * tile);
  TiledRun r;
  r.tile_shapes = static_cast<int>(counts.size());
  const auto t0 = Clock::now();
  for (const auto& ty : ys) {
    for (const auto& tx : xs) {
      const int h = ty.input_size, w = tx.input_size;
      TileInput(packed, ty, tx, in.data());
      Ort::Session& s = sessions.Get(h, w);
      const auto b = Clock::now();
      SessionSet::Run(s, in.data(), out.data(), h, w);
      r.seconds_run += Seconds(b, Clock::now());
      const int y0 = 2 * (ty.keep_start - ty.input_start);
      const int x0 = 2 * (tx.keep_start - tx.input_start);
      const int kh = 2 * (ty.keep_end - ty.keep_start), kw = 2 * (tx.keep_end - tx.keep_start);
      for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < kh; ++y) {
          const float* src = out.data() + (static_cast<size_t>(c) * 2 * h + y0 + y) * 2 * w + x0;
          float* dst = output.data() +
                       (static_cast<size_t>(c) * oh + 2 * ty.keep_start + y) * ow + 2 * tx.keep_start;
          std::memcpy(dst, src, sizeof(float) * kw);
        }
      }
      ++r.tile_count;
      r.computed_packed_pixels += static_cast<uint64_t>(h) * w;
    }
  }
  r.seconds_total = Seconds(t0, Clock::now());
  return r;
}

// --- JSON -------------------------------------------------------------------------------------

std::string Quote(const std::string& s) {
  std::string o = "\"";
  for (char ch : s) {
    if (ch == '"' || ch == '\\') {
      o += '\\';
      o += ch;
    } else if (ch == '\n') {
      o += "\\n";
    } else if (static_cast<unsigned char>(ch) < 0x20) {
      o += ' ';
    } else {
      o += ch;
    }
  }
  return o + "\"";
}

std::string Num(double v) {
  if (!std::isfinite(v)) return "null";
  char b[64];
  std::snprintf(b, sizeof b, "%.6g", v);
  return b;
}

std::vector<std::string> Split(const std::string& s) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (!item.empty()) out.push_back(item);
  }
  return out;
}

double Median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v.empty() ? 0 : (v.size() % 2 ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]));
}

}  // namespace

int main(int argc, char** argv) {
  std::map<std::string, std::string> a;
  for (int i = 1; i + 1 < argc; i += 2) a[argv[i]] = argv[i + 1];
  auto get = [&](const char* k, const char* d) { return a.count(k) ? a[k] : std::string(d); };
  Config cfg;
  cfg.ep = get("--ep", "coreml");
  cfg.coreml_format = get("--coreml-format", "MLProgram");
  cfg.coreml_units = get("--coreml-units", "CPUAndGPU");
  cfg.coreml_low_precision = get("--coreml-low-precision", "0");
  cfg.coreml_static = get("--coreml-static", "0");
  cfg.coreml_specialization = get("--coreml-specialization", "Default");
  cfg.coreml_cache = get("--coreml-cache", "");
  cfg.webgpu_lib = get("--webgpu-lib", "");
  cfg.strategy = get("--strategy", "free");
  cfg.disable_cpu_fallback = get("--cpu-fallback", "0") == "0";
  const std::string graph = get("--graph", "utnet2_bayer_dark_loss_10k.onnx");
  g_packed_output = get("--packed-output", "0") == "1";
  const std::string data = get("--data", "data");
  const std::string out_path = get("--out", "result.json");
  const auto photos = Split(get("--photos", ""));
  const auto tiles = Split(get("--tiles", "1536"));
  const int overlap = std::stoi(get("--overlap", "96"));
  const int repeats = std::stoi(get("--repeats", "3"));
  const bool precision = get("--precision", "1") == "1";

  std::ostringstream j;
  j << "{\n \"onnxruntime\": " << Quote(Ort::GetVersionString()) << ",\n";
  j << " \"config\": {\"ep\": " << Quote(cfg.ep) << ", \"strategy\": " << Quote(cfg.strategy);
  if (cfg.ep == "coreml") {
    j << ", \"ModelFormat\": " << Quote(cfg.coreml_format)
      << ", \"MLComputeUnits\": " << Quote(cfg.coreml_units)
      << ", \"RequireStaticInputShapes\": " << Quote(cfg.coreml_static)
      << ", \"AllowLowPrecisionAccumulationOnGPU\": " << Quote(cfg.coreml_low_precision)
      << ", \"SpecializationStrategy\": " << Quote(cfg.coreml_specialization)
      << ", \"ModelCacheDirectory\": " << Quote(cfg.coreml_cache);
  }
  j << ", \"cpu_ep_fallback_disabled\": " << (cfg.disable_cpu_fallback ? "true" : "false")
    << ", \"intra_op_threads\": " << cfg.threads << ", \"graph\": " << Quote(graph)
    << ", \"packed_output\": " << (g_packed_output ? "true" : "false") << "},\n";
  j << " \"available_providers\": [";
  {
    const auto providers = Ort::GetAvailableProviders();
    for (size_t i = 0; i < providers.size(); ++i) j << (i ? ", " : "") << Quote(providers[i]);
  }
  j << "],\n";
  const Footprint start = ReadFootprint();
  j << " \"footprint_start_bytes\": " << start.current << ",\n";

  int status = 0;
  std::ostringstream photo_json;
  try {
    Ort::Env env(ORT_LOGGING_LEVEL_VERBOSE, "measure", LogCallback, nullptr);
    if (cfg.ep == "webgpu") {
      env.RegisterExecutionProviderLibrary("webgpu_ep", cfg.webgpu_lib);
      j << " \"ep_devices\": [";
      bool first = true;
      for (const auto& d : env.GetEpDevices()) {
        j << (first ? "" : ", ") << "{\"ep\": " << Quote(d.EpName())
          << ", \"vendor\": " << Quote(d.EpVendor()) << "}";
        first = false;
      }
      j << "],\n";
    }
    const auto manifest = ReadManifest(data);

    // 1. Node placement: one session with verbose logging.
    {
      SessionSet probe(env, graph, cfg);
      const auto t0 = Clock::now();
      g_capture_coreml = true;
      probe.Get(256, 256, ORT_LOGGING_LEVEL_VERBOSE);
      g_capture_coreml = false;
      j << " \"probe_session_seconds\": " << Num(Seconds(t0, Clock::now())) << ",\n";
    }

    // 2. Precision against the expected outputs, and the tiled output against one pass.
    if (precision) {
      SessionSet sessions(env, graph, cfg);
      std::ostringstream pj;
      pj << " \"precision\": [";
      bool first = true;
      Array crop_out;
      for (const char* name : {"random_64", "random_256", "rawnind_crop_512"}) {
        const auto& ei = Find(manifest, name, "expected_input");
        const auto& eo = Find(manifest, name, "expected_output");
        Array x = ReadArray(data + "/" + ei.file, ei.c, ei.h, ei.w);
        Array y = ReadArray(data + "/" + eo.file, eo.c, eo.h, eo.w);
        Array r{3, eo.h, eo.w, std::vector<float>(y.size())};
        Ort::Session& s = sessions.Get(x.h, x.w);
        SessionSet::Run(s, x.data.data(), r.data.data(), x.h, x.w);
        const Stats st = Compare(r.data.data(), y);
        pj << (first ? "\n  " : ",\n  ") << "{\"input\": " << Quote(name)
          << ", \"relative_difference\": " << Num(st.relative_difference)
          << ", \"non_finite\": " << st.non_finite
          << ", \"within_tolerance\": " << (st.relative_difference <= kTolerance ? "true" : "false")
          << "}";
        first = false;
        std::cout << "precision " << name << ": " << st.relative_difference << " non-finite "
                  << st.non_finite << std::endl;
        if (std::string(name) == "rawnind_crop_512") {
          // Tiles of 256 with overlap 96 against the one-pass output of the same configuration.
          std::vector<float> tiled;
          SessionSet tile_sessions(env, graph, cfg);
          RunTiled(tile_sessions, x, 256, overlap, tiled);
          const Stats ts = Compare(tiled.data(), r);
          const Stats te = Compare(tiled.data(), y);
          pj << ",\n  {\"input\": \"rawnind_crop_512 tiles 256 overlap 96\", "
            << "\"relative_difference_to_one_pass\": " << Num(ts.relative_difference)
            << ", \"relative_difference\": " << Num(te.relative_difference)
            << ", \"within_tolerance\": " << (te.relative_difference <= kTolerance ? "true" : "false")
            << "}";
          std::cout << "tiled crop: to one pass " << ts.relative_difference << ", to expected "
                    << te.relative_difference << std::endl;
        }
      }
      pj << "\n ],\n";
      pj << " \"precision_sessions\": [";
      first = true;
      for (const auto& c : sessions.created()) {
        pj << (first ? "" : ", ") << "{\"shape\": [" << c.h << ", " << c.w
          << "], \"create_seconds\": " << Num(c.create_seconds)
          << ", \"first_run_seconds\": " << Num(c.first_run_seconds) << "}";
        first = false;
      }
      pj << "],\n";
      j << pj.str();
    }

    // 3. Time of the full photos.
    bool first_photo = true;
    for (const auto& name : photos) {
      const auto& e = Find(manifest, name, "photo");
      Array packed = ReadArray(data + "/" + e.file, e.c, e.h, e.w);
      for (const auto& tile_text : tiles) {
        const int tile = std::stoi(tile_text);
        SessionSet sessions(env, graph, cfg);
        std::vector<float> output;
        RunTiled(sessions, packed, tile, overlap, output);  // not measured
        std::vector<double> totals, runs;
        std::vector<int> thermal{ThermalState()};
        TiledRun last;
        for (int i = 0; i < repeats; ++i) {
          std::this_thread::sleep_for(std::chrono::seconds(1));
          last = RunTiled(sessions, packed, tile, overlap, output);
          totals.push_back(last.seconds_total);
          runs.push_back(last.seconds_run);
          thermal.push_back(ThermalState());
        }
        double sum = 0, largest = 0;
        uint64_t non_finite = 0;
        for (float v : output) {
          if (!std::isfinite(v)) {
            ++non_finite;
            continue;
          }
          sum += v;
          largest = std::max(largest, std::fabs(static_cast<double>(v)));
        }
        const Footprint fp = ReadFootprint();
        std::ostringstream e1;
        e1 << "{\"photo\": " << Quote(name)
          << ", \"mosaic_megapixels\": " << Num(4.0 * packed.h * packed.w / 1e6)
          << ", \"tile\": " << tile << ", \"overlap\": " << overlap
          << ", \"tile_count\": " << last.tile_count << ", \"tile_shapes\": " << last.tile_shapes
          << ", \"computed_packed_pixels\": " << last.computed_packed_pixels
          << ", \"median_seconds_total\": " << Num(Median(totals))
          << ", \"min_seconds_total\": " << Num(*std::min_element(totals.begin(), totals.end()))
          << ", \"median_seconds_run\": " << Num(Median(runs)) << ", \"seconds_total\": [";
        for (size_t i = 0; i < totals.size(); ++i) e1 << (i ? ", " : "") << Num(totals[i]);
        e1 << "], \"thermal_state\": [";
        for (size_t i = 0; i < thermal.size(); ++i) e1 << (i ? ", " : "") << thermal[i];
        e1 << "], \"output_sum\": " << Num(sum) << ", \"output_max_abs\": " << Num(largest)
          << ", \"output_non_finite\": " << non_finite
          << ", \"footprint_after_runs_bytes\": " << fp.current
          << ", \"footprint_peak_bytes\": " << fp.peak << ", \"sessions\": [";
        bool first = true;
        for (const auto& c : sessions.created()) {
          e1 << (first ? "" : ", ") << "{\"shape\": [" << c.h << ", " << c.w
            << "], \"free\": " << (c.free ? "true" : "false")
            << ", \"create_seconds\": " << Num(c.create_seconds)
            << ", \"first_run_seconds\": " << Num(c.first_run_seconds) << "}";
          first = false;
        }
        e1 << "]}";
        photo_json << (first_photo ? "\n  " : ",\n  ") << e1.str();
        first_photo = false;
        sessions.Clear();
        const Footprint released = ReadFootprint();
        std::cout << name << " tile " << tile << ": " << last.tile_count << " tiles, "
                  << last.tile_shapes << " shapes, " << Median(totals) << " s (run "
                  << Median(runs) << " s), footprint " << fp.current / double(1 << 30)
                  << " GiB, peak " << fp.peak / double(1 << 30) << " GiB, after release "
                  << released.current / double(1 << 30) << " GiB, non-finite " << non_finite
                  << std::endl;
      }
    }
  } catch (const std::exception& error) {
    j << " \"error\": " << Quote(error.what()) << ",\n";
    std::cerr << "error: " << error.what() << std::endl;
    status = 1;
  }
  j << " \"photos\": [" << photo_json.str() << "\n ],\n";
  const Footprint end = ReadFootprint();
  j << " \"footprint_peak_bytes\": " << end.peak << ",\n \"log\": [";
  for (size_t i = 0; i < g_log.size(); ++i) j << (i ? ",\n  " : "\n  ") << Quote(g_log[i]);
  j << "\n ]\n}\n";
  std::ofstream(out_path) << j.str();
  return status;
}
