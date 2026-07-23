// Isolates what Green Context contributes, independent of the memory path.
//
// Total work is held constant across three ways of issuing it, all on the
// device memory path:
//   case1_1stream_2x  : one stream, one kernel over a 2N working set
//   case2_2stream_nogc: two plain streams (default context), N each
//   case3_2stream_gc  : two Green Context streams on disjoint SMs, N each
//
// Logical bytes read are identical in all three (2N * reuse), so the wall
// time differences are purely about how the work is issued. If plain streams
// serialize, case2 ~ case1 and only case3 overlaps.
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

#define CK(x)                                                                  \
  do {                                                                         \
    CUresult _r = (x);                                                         \
    if (_r != CUDA_SUCCESS) {                                                  \
      std::fprintf(stderr, "%s:%d %s:%s\n", __FILE__, __LINE__, #x, cn(_r));  \
      std::exit(2);                                                            \
    }                                                                          \
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

static double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  size_t n = v.size();
  return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

static void make_idx(std::vector<int>& idx, int m, const std::string& pat, int stride) {
  idx.resize(m);
  if (pat == "uniform") {
    for (int i = 0; i < m; ++i) idx[i] = i;
  } else if (pat == "strided") {
    for (int i = 0; i < m; ++i)
      idx[i] = static_cast<int>((static_cast<long long>(i) * stride) % m);
  } else {
    for (int i = 0; i < m; ++i)
      idx[i] = static_cast<int>((static_cast<unsigned long long>(i) * 2654435761ull) % m);
  }
}

struct Buf {
  CUdeviceptr in, idx, out;
  int m;
};

static Buf alloc_device(int m, const std::vector<int>& idx) {
  Buf b{};
  b.m = m;
  size_t bytes = static_cast<size_t>(m) * 4;
  CK(cuMemAlloc(&b.in, bytes));
  CK(cuMemAlloc(&b.idx, bytes));
  CK(cuMemAlloc(&b.out, bytes));
  std::vector<float> h(m, 1.0f);
  CK(cuMemcpyHtoD(b.in, h.data(), bytes));
  CK(cuMemcpyHtoD(b.idx, idx.data(), bytes));
  return b;
}

int main(int argc, char** argv) {
  int MB = ia(argc, argv, "mb", 1);          // per-partition size for the 2-stream cases
  int IT = ia(argc, argv, "iters", 1);
  int TR = ia(argc, argv, "trials", 31);
  int stride = ia(argc, argv, "stride", 17);
  int sms = ia(argc, argv, "sms", 8);        // per Green Context
  // Cases 2 and 3 read the very same buffers a and b, so running them back to
  // back leaves case3 with everything already warm from case2 -- a bias in
  // Green Context's favour. --only=1|2|3 runs a single case so each can be
  // timed from a cold start in its own process; 0 (default) runs all three.
  int ONLY = ia(argc, argv, "only", 0);
  std::string pat = sa(argc, argv, "pat", "uniform");

  const int m1 = (MB * 1024 * 1024) / 4;     // one partition
  const int m2 = m1 * 2;                     // single-stream case covers both

  CK(cuInit(0));
  CUdevice dev;
  CK(cuDeviceGet(&dev, 0));
  CUcontext pctx;
  CK(cuDevicePrimaryCtxRetain(&pctx, dev));
  CK(cuCtxSetCurrent(pctx));

  CUmodule mod;
  CK(cuModuleLoad(&mod, "spatter_gather.ptx"));
  CUfunction fn;
  CK(cuModuleGetFunction(&fn, mod, "gather_reuse"));

  std::vector<int> idx1, idx2;
  make_idx(idx1, m1, pat, stride);
  make_idx(idx2, m2, pat, stride);

  // Two N-sized buffers for the split cases, one 2N buffer for the single case.
  Buf a = alloc_device(m1, idx1);
  Buf b = alloc_device(m1, idx1);
  Buf big = alloc_device(m2, idx2);

  const int block = 256;
  auto launch = [&](CUstream s, const Buf& buf, int it) {
    int m = buf.m;
    void* args[] = {(void*)&buf.in, (void*)&buf.idx, (void*)&buf.out, &m, &it};
    int grid = (m + block - 1) / block;
    return cuLaunchKernel(fn, grid, 1, 1, block, 1, 1, 0, s, args, nullptr);
  };

  // case1 / case2 run on the default (full-GPU) context: no Green Context.
  CUstream p0, p1;
  CK(cuStreamCreate(&p0, CU_STREAM_NON_BLOCKING));
  CK(cuStreamCreate(&p1, CU_STREAM_NON_BLOCKING));

  // case3: two Green Contexts over disjoint SM partitions.
  CUdevResource all, g0res, rem0, g1res, rem1;
  std::memset(&all, 0, sizeof(all));
  std::memset(&g0res, 0, sizeof(g0res));
  std::memset(&rem0, 0, sizeof(rem0));
  std::memset(&g1res, 0, sizeof(g1res));
  std::memset(&rem1, 0, sizeof(rem1));
  CK(cuDeviceGetDevResource(dev, &all, CU_DEV_RESOURCE_TYPE_SM));
  unsigned n0 = 1;
  CK(cuDevSmResourceSplitByCount(&g0res, &n0, &all, &rem0, 0, static_cast<unsigned>(sms)));
  int smtot = 0;
  CK(cuDeviceGetAttribute(&smtot, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev));
  if (sms * 2 == smtot) {
    g1res = rem0;
  } else {
    unsigned n1 = 1;
    CK(cuDevSmResourceSplitByCount(&g1res, &n1, &rem0, &rem1, 0, static_cast<unsigned>(sms)));
  }
  CUdevResourceDesc d0, d1;
  CK(cuDevResourceGenerateDesc(&d0, &g0res, 1));
  CK(cuDevResourceGenerateDesc(&d1, &g1res, 1));
  CUgreenCtx gc0, gc1;
  CK(cuGreenCtxCreate(&gc0, d0, dev, CU_GREEN_CTX_DEFAULT_STREAM));
  CK(cuGreenCtxCreate(&gc1, d1, dev, CU_GREEN_CTX_DEFAULT_STREAM));
  CUstream s0, s1;
  CK(cuGreenCtxStreamCreate(&s0, gc0, CU_STREAM_NON_BLOCKING, 0));
  CK(cuGreenCtxStreamCreate(&s1, gc1, CU_STREAM_NON_BLOCKING, 0));

  auto time_ms = [](auto&& body) {
    auto t0 = std::chrono::steady_clock::now();
    body();
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
  };

  auto run_case1 = [&]() {
    CK(launch(p0, big, IT));
    CK(cuStreamSynchronize(p0));
  };
  auto run_case2 = [&]() {
    CK(launch(p0, a, IT));
    CK(launch(p1, b, IT));
    CK(cuStreamSynchronize(p0));
    CK(cuStreamSynchronize(p1));
  };
  auto run_case3 = [&]() {
    CK(launch(s0, a, IT));
    CK(launch(s1, b, IT));
    CK(cuStreamSynchronize(s0));
    CK(cuStreamSynchronize(s1));
  };

  auto want = [&](int c) { return ONLY == 0 || ONLY == c; };
  for (int w = 0; w < 5; ++w) {
    if (want(1)) run_case1();
    if (want(2)) run_case2();
    if (want(3)) run_case3();
  }

  std::vector<double> t1v, t2v, t3v;
  for (int t = 0; t < TR; ++t) {
    if (want(1)) t1v.push_back(time_ms(run_case1));
    if (want(2)) t2v.push_back(time_ms(run_case2));
    if (want(3)) t3v.push_back(time_ms(run_case3));
  }
  if (t1v.empty()) t1v.push_back(0.0);
  if (t2v.empty()) t2v.push_back(0.0);
  if (t3v.empty()) t3v.push_back(0.0);

  const double bytes = static_cast<double>(m2) * 4.0 * IT;  // identical for all cases
  const double c1 = median(t1v), c2 = median(t2v), c3 = median(t3v);
  auto gbps = [&](double ms) { return bytes / (ms / 1e3) / 1e9; };

  std::printf("pattern,mb_per_stream,total_mb,reuse,threads_per_stream,total_bytes,"
              "case1_1stream_2x_ms,case2_2stream_nogc_ms,case3_2stream_gc_ms,"
              "case1_gbps,case2_gbps,case3_gbps,"
              "speedup_case2_vs_case1,speedup_case3_vs_case1,speedup_case3_vs_case2\n");
  std::printf("%s,%d,%d,%d,%d,%.0f,%.4f,%.4f,%.4f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
              pat.c_str(), MB, MB * 2, IT, m1, bytes,
              c1, c2, c3, gbps(c1), gbps(c2), gbps(c3),
              c1 / c2, c1 / c3, c2 / c3);
  return 0;
}
