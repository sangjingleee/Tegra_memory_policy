// Spatter-style gather: device-memory path vs zero-copy path, co-run on a
// Green Context SM split, to test whether their memory bandwidths AGGREGATE
// (L2 + SLC independent paths) or CONTEND (shared DRAM).
//
// MEASUREMENT (effective bandwidth, GB/s):
//   bytes_read = m_elements * 4B * reuse_iters
//   gbps       = bytes_read / kernel_time     (kernel_time via CUDA events
//                bracketing the gather kernel on its own stream)
//   solo : each path measured alone.
//   co   : both kernels launched concurrently on DISJOINT SM partitions
//          (two Green Contexts) -> only the memory hierarchy is shared.
//          zc reuse is scaled (zc_it = IT*zc_solo/dev_solo) so the two kernels
//          have ~equal runtime and actually overlap.
//   Reported value = MEDIAN over `trials` (robust; max/best is biased optimistic).
//   aggregate = dev_co+zc_co ; solo_sum = dev_solo+zc_solo ;
//   ratio = aggregate/solo_sum (~100% => the two paths are independent).
//
// **REQUIRES locked clocks**: `sudo jetson_clocks`. Otherwise DVFS lets the
// heavier co-run ramp the GPU/EMC clock above the solo run, producing
// co > solo (ratio > 100%) which is a clock artifact, not real aggregation.
#include <cuda.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static const char* cn(CUresult r) {
  const char* n = nullptr;
  cuGetErrorName(r, &n);
  return n ? n : "UNKNOWN";
}

#define CK(x)                                                                    \
  do {                                                                           \
    CUresult _r = (x);                                                           \
    if (_r != CUDA_SUCCESS) {                                                    \
      std::fprintf(stderr, "%s:%s\n", #x, cn(_r));                              \
      std::exit(2);                                                              \
    }                                                                            \
  } while (0)

static int ia(int c, char** v, const char* k, int d) {
  std::string p = std::string("--") + k + "=";
  for (int i = 1; i < c; ++i) {
    std::string a = v[i];
    if (a.rfind(p, 0) == 0) return std::atoi(a.substr(p.size()).c_str());
  }
  return d;
}

static std::string sa(int c, char** v, const char* k, const char* d) {
  std::string p = std::string("--") + k + "=";
  for (int i = 1; i < c; ++i) {
    std::string a = v[i];
    if (a.rfind(p, 0) == 0) return a.substr(p.size());
  }
  return d;
}

struct Summary {
  double p05;
  double p50;
  double p95;
};

static double quantile(std::vector<double> v, double q) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  const double pos = q * static_cast<double>(v.size() - 1);
  const size_t lo = static_cast<size_t>(pos);
  const size_t hi = std::min(lo + 1, v.size() - 1);
  const double frac = pos - static_cast<double>(lo);
  return v[lo] * (1.0 - frac) + v[hi] * frac;
}

static Summary summarize(const std::vector<double>& v) {
  return {quantile(v, 0.05), quantile(v, 0.50), quantile(v, 0.95)};
}

int main(int argc, char** argv) {
  int dev_sms_req = ia(argc, argv, "dev-sms", 8);
  int zc_sms_req = ia(argc, argv, "zc-sms", 8);
  int MB = ia(argc, argv, "mb", 1);
  int IT = ia(argc, argv, "iters", 32);
  int TR = ia(argc, argv, "trials", 31);
  int BATCH = ia(argc, argv, "batches", 1);
  int stride = ia(argc, argv, "stride", 17);
  std::string pat = sa(argc, argv, "pat", "uniform");

  if (dev_sms_req <= 0 || zc_sms_req <= 0 || MB <= 0 || IT <= 0 || TR < 3 || BATCH <= 0) {
    std::fprintf(stderr, "SM counts, mb, iters, batches must be positive; trials must be >= 3.\n");
    return 2;
  }

  int m = (MB * 1024 * 1024) / 4;
  size_t bytes = static_cast<size_t>(m) * 4;
  size_t index_bytes = static_cast<size_t>(m) * 4;

  CK(cuInit(0));
  CUdevice dev;
  CK(cuDeviceGet(&dev, 0));
  int smtot = 0;
  CK(cuDeviceGetAttribute(&smtot, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev));
  if (dev_sms_req + zc_sms_req > smtot) {
    std::fprintf(stderr, "requested split exceeds total SMs: %d+%d > %d\n",
                 dev_sms_req, zc_sms_req, smtot);
    return 2;
  }

  CUcontext pctx;
  CK(cuDevicePrimaryCtxRetain(&pctx, dev));
  CK(cuCtxSetCurrent(pctx));

  CUmodule mod;
  CK(cuModuleLoad(&mod, "spatter_gather.ptx"));
  CUfunction fn;
  CK(cuModuleGetFunction(&fn, mod, "gather_reuse"));

  std::vector<int> idx(m);
  if (pat == "uniform") {
    for (int i = 0; i < m; ++i) idx[i] = i;
  } else if (pat == "strided") {
    for (int i = 0; i < m; ++i) idx[i] = static_cast<int>((static_cast<long long>(i) * stride) % m);
  } else {
    for (int i = 0; i < m; ++i) idx[i] = static_cast<int>((static_cast<unsigned long long>(i) * 2654435761ull) % m);
  }

  CUdeviceptr d_in, d_idx, d_out;
  CK(cuMemAlloc(&d_in, bytes));
  CK(cuMemAlloc(&d_idx, index_bytes));
  CK(cuMemAlloc(&d_out, bytes));
  {
    std::vector<float> h(m, 1.0f);
    CK(cuMemcpyHtoD(d_in, h.data(), bytes));
    CK(cuMemcpyHtoD(d_idx, idx.data(), index_bytes));
  }

  void* z_in_h = nullptr;
  void* z_idx_h = nullptr;
  CUdeviceptr z_in, z_idx, z_out;
  CK(cuMemHostAlloc(&z_in_h, bytes, CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP));
  CK(cuMemHostAlloc(&z_idx_h, index_bytes, CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP));
  {
    float* p = static_cast<float*>(z_in_h);
    for (int i = 0; i < m; ++i) p[i] = 1.0f;
    int* q = static_cast<int*>(z_idx_h);
    for (int i = 0; i < m; ++i) q[i] = idx[i];
  }
  CK(cuMemHostGetDevicePointer(&z_in, z_in_h, 0));
  CK(cuMemHostGetDevicePointer(&z_idx, z_idx_h, 0));
  CK(cuMemAlloc(&z_out, bytes));

  CUdevResource all, dev_grp, rem0, zc_grp, rem1;
  std::memset(&all, 0, sizeof(all));
  std::memset(&dev_grp, 0, sizeof(dev_grp));
  std::memset(&rem0, 0, sizeof(rem0));
  std::memset(&zc_grp, 0, sizeof(zc_grp));
  std::memset(&rem1, 0, sizeof(rem1));
  CK(cuDeviceGetDevResource(dev, &all, CU_DEV_RESOURCE_TYPE_SM));
  unsigned dev_n = 1;
  CK(cuDevSmResourceSplitByCount(&dev_grp, &dev_n, &all, &rem0, 0,
                                 static_cast<unsigned>(dev_sms_req)));
  if (dev_sms_req + zc_sms_req == smtot) {
    zc_grp = rem0;
  } else {
    unsigned zc_n = 1;
    CK(cuDevSmResourceSplitByCount(&zc_grp, &zc_n, &rem0, &rem1, 0,
                                   static_cast<unsigned>(zc_sms_req)));
  }

  CUdevResourceDesc desc0, desc1;
  CK(cuDevResourceGenerateDesc(&desc0, &dev_grp, 1));
  CK(cuDevResourceGenerateDesc(&desc1, &zc_grp, 1));
  CUgreenCtx gc0, gc1;
  CK(cuGreenCtxCreate(&gc0, desc0, dev, CU_GREEN_CTX_DEFAULT_STREAM));
  CK(cuGreenCtxCreate(&gc1, desc1, dev, CU_GREEN_CTX_DEFAULT_STREAM));

  CUdevResource actual0, actual1;
  std::memset(&actual0, 0, sizeof(actual0));
  std::memset(&actual1, 0, sizeof(actual1));
  CK(cuGreenCtxGetDevResource(gc0, &actual0, CU_DEV_RESOURCE_TYPE_SM));
  CK(cuGreenCtxGetDevResource(gc1, &actual1, CU_DEV_RESOURCE_TYPE_SM));
  int dev_sms = actual0.sm.smCount;
  int zc_sms = actual1.sm.smCount;

  CUcontext c0, c1;
  CK(cuCtxFromGreenCtx(&c0, gc0));
  CK(cuCtxFromGreenCtx(&c1, gc1));
  CUstream s0, s1;
  CK(cuGreenCtxStreamCreate(&s0, gc0, CU_STREAM_NON_BLOCKING, 0));
  CK(cuGreenCtxStreamCreate(&s1, gc1, CU_STREAM_NON_BLOCKING, 0));

  int block = 256;
  int grid = (m + block - 1) / block;
  auto launch = [&](CUstream s, CUdeviceptr in, CUdeviceptr id, CUdeviceptr out, int it) {
    void* args[] = {&in, &id, &out, &m, &it};
    return cuLaunchKernel(fn, grid, 1, 1, block, 1, 1, 0, s, args, nullptr);
  };

  CK(cuCtxSetCurrent(c0));
  CUevent e0a, e0b;
  CK(cuEventCreate(&e0a, 0));
  CK(cuEventCreate(&e0b, 0));
  CK(cuCtxSetCurrent(c1));
  CUevent e1a, e1b;
  CK(cuEventCreate(&e1a, 0));
  CK(cuEventCreate(&e1b, 0));

  auto measure = [&](CUcontext ctx, CUstream st, CUevent a, CUevent b,
                     CUdeviceptr in, CUdeviceptr id, CUdeviceptr out, int it) -> Summary {
    double rb = static_cast<double>(m) * 4 * it * BATCH;
    cuCtxSetCurrent(ctx);
    for (int w = 0; w < 3; ++w) CK(launch(st, in, id, out, it));
    cuStreamSynchronize(st);
    std::vector<double> samples;
    for (int t = 0; t < TR; ++t) {
      cuEventRecord(a, st);
      for (int bidx = 0; bidx < BATCH; ++bidx) CK(launch(st, in, id, out, it));
      cuEventRecord(b, st);
      cuEventSynchronize(b);
      float ms = 0.0f;
      cuEventElapsedTime(&ms, a, b);
      samples.push_back(rb / (ms / 1e3) / 1e9);
    }
    return summarize(samples);
  };

  Summary dev_solo = measure(c0, s0, e0a, e0b, d_in, d_idx, d_out, IT);
  Summary zc_solo = measure(c1, s1, e1a, e1b, z_in, z_idx, z_out, IT);
  // Preserve the requested reuse in both kernels.  Instead of making one
  // kernel use fewer reuse iterations, balance their total duration by
  // submitting more batches to the faster path.
  const int dev_it = IT;
  const int zc_it = IT;
  int dev_batches = BATCH;
  int zc_batches = BATCH;
  if (dev_solo.p50 >= zc_solo.p50) {
    dev_batches = std::max(1, static_cast<int>(BATCH * dev_solo.p50 / zc_solo.p50 + 0.5));
  } else {
    zc_batches = std::max(1, static_cast<int>(BATCH * zc_solo.p50 / dev_solo.p50 + 0.5));
  }
  double dev_rb = static_cast<double>(m) * 4 * dev_it * dev_batches;
  double zc_rb = static_cast<double>(m) * 4 * zc_it * zc_batches;

  std::vector<double> dev_s, zc_s, aggregate_s, overlap_s, wall_s;
  for (int t = 0; t < TR + 3; ++t) {  // first 3 iterations are warmup (discarded)
    const auto wall_begin = std::chrono::steady_clock::now();
    CK(cuCtxSetCurrent(c0));
    CK(cuEventRecord(e0a, s0));
    CK(cuCtxSetCurrent(c1));
    CK(cuEventRecord(e1a, s1));
    const int max_batches = std::max(dev_batches, zc_batches);
    for (int bidx = 0; bidx < max_batches; ++bidx) {
      // Alternate submission order to avoid a systematic first-stream advantage.
      if ((bidx + t) % 2 == 0) {
        if (bidx < dev_batches) {
          CK(cuCtxSetCurrent(c0));
          CK(launch(s0, d_in, d_idx, d_out, dev_it));
        }
        if (bidx < zc_batches) {
          CK(cuCtxSetCurrent(c1));
          CK(launch(s1, z_in, z_idx, z_out, zc_it));
        }
      } else {
        if (bidx < zc_batches) {
          CK(cuCtxSetCurrent(c1));
          CK(launch(s1, z_in, z_idx, z_out, zc_it));
        }
        if (bidx < dev_batches) {
          CK(cuCtxSetCurrent(c0));
          CK(launch(s0, d_in, d_idx, d_out, dev_it));
        }
      }
    }
    CK(cuCtxSetCurrent(c0));
    CK(cuEventRecord(e0b, s0));
    CK(cuCtxSetCurrent(c1));
    CK(cuEventRecord(e1b, s1));
    CK(cuCtxSetCurrent(c0));
    CK(cuEventSynchronize(e0b));
    CK(cuCtxSetCurrent(c1));
    CK(cuEventSynchronize(e1b));
    const auto wall_end = std::chrono::steady_clock::now();
    float md = 0.0f, mz = 0.0f;
    CK(cuCtxSetCurrent(c0));
    CK(cuEventElapsedTime(&md, e0a, e0b));
    CK(cuCtxSetCurrent(c1));
    CK(cuEventElapsedTime(&mz, e1a, e1b));
    if (t < 3) continue;  // warmup
    const double wall_ms = std::chrono::duration<double, std::milli>(wall_end - wall_begin).count();
    dev_s.push_back(dev_rb / (md / 1e3) / 1e9);
    zc_s.push_back(zc_rb / (mz / 1e3) / 1e9);
    // This is the primary co-run aggregate: both paths' bytes over one common interval.
    aggregate_s.push_back((dev_rb + zc_rb) / (wall_ms / 1e3) / 1e9);
    overlap_s.push_back(std::max(md, mz) / wall_ms);
    wall_s.push_back(wall_ms);
  }
  Summary dev_co = summarize(dev_s);
  Summary zc_co = summarize(zc_s);
  Summary aggregate = summarize(aggregate_s);
  Summary overlap = summarize(overlap_s);
  Summary wall = summarize(wall_s);

  const double solo_sum = dev_solo.p50 + zc_solo.p50;
  std::printf("pattern,size_mb,iters,base_batches,dev_batches,zc_batches,dev_sms,zc_sms,dev_solo_p50_gbps,zc_solo_p50_gbps,dev_co_p50_gbps,zc_co_p50_gbps,aggregate_common_p05_gbps,aggregate_common_p50_gbps,aggregate_common_p95_gbps,solo_sum_p50_gbps,aggregate_ratio_p05_pct,aggregate_ratio_p50_pct,aggregate_ratio_p95_pct,device_preserve_p50_pct,zc_preserve_p50_pct,co_wall_p50_ms,overlap_p50_ratio\n");
  std::printf("%s,%d,%d,%d,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
              pat.c_str(), MB, IT, BATCH, dev_batches, zc_batches, dev_sms, zc_sms, dev_solo.p50, zc_solo.p50,
              dev_co.p50, zc_co.p50, aggregate.p05, aggregate.p50, aggregate.p95, solo_sum,
              aggregate.p05 / solo_sum * 100.0, aggregate.p50 / solo_sum * 100.0,
              aggregate.p95 / solo_sum * 100.0, dev_co.p50 / dev_solo.p50 * 100.0,
              zc_co.p50 / zc_solo.p50 * 100.0, wall.p50, overlap.p50);
  return 0;
}
