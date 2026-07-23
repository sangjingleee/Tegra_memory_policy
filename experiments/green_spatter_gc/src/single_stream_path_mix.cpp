// Does mixing memory paths raise bandwidth on ONE stream?
//
// No Green Context, no second stream, no MPS -- the only thing that changes is
// where the two halves of the working set live. Total logical bytes are equal
// in all three cases:
//   case1_dev2mb      : a single 2N device buffer
//   case2_dev1_dev1   : two separate N device buffers
//   case3_dev1_zc1    : one N device buffer + one N zero-copy buffer
//
// Cases 2 and 3 are structurally identical (same launch count, same chunking),
// so their difference isolates the memory path. Chunked launches alternate
// A,B,A,B,... on the one stream with no sync in between, so the two buffers'
// work is in flight together rather than serialised.
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

// kind: "device" -> cuMemAlloc, "zc" -> host-mapped (no copy, GPU reads host).
static Buf alloc_buf(const std::string& kind, int m, const std::vector<int>& idx) {
  Buf b{};
  b.m = m;
  size_t bytes = static_cast<size_t>(m) * 4;
  if (kind == "zc") {
    void* in_h = nullptr;
    void* idx_h = nullptr;
    CK(cuMemHostAlloc(&in_h, bytes, CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP));
    CK(cuMemHostAlloc(&idx_h, bytes, CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP));
    float* p = static_cast<float*>(in_h);
    for (int i = 0; i < m; ++i) p[i] = 1.0f;
    int* q = static_cast<int*>(idx_h);
    for (int i = 0; i < m; ++i) q[i] = idx[i];
    CK(cuMemHostGetDevicePointer(&b.in, in_h, 0));
    CK(cuMemHostGetDevicePointer(&b.idx, idx_h, 0));
  } else {
    CK(cuMemAlloc(&b.in, bytes));
    CK(cuMemAlloc(&b.idx, bytes));
    std::vector<float> h(m, 1.0f);
    CK(cuMemcpyHtoD(b.in, h.data(), bytes));
    CK(cuMemcpyHtoD(b.idx, idx.data(), bytes));
  }
  CK(cuMemAlloc(&b.out, bytes));
  return b;
}

int main(int argc, char** argv) {
  int MB = ia(argc, argv, "mb", 1);        // per half; case1 uses 2*MB
  int IT = ia(argc, argv, "iters", 1);
  int TR = ia(argc, argv, "trials", 51);
  int CH = ia(argc, argv, "chunks", 5);    // launches per buffer, interleaved
  int stride = ia(argc, argv, "stride", 17);
  std::string pat = sa(argc, argv, "pat", "uniform");

  const int m1 = (MB * 1024 * 1024) / 4;
  const int m2 = m1 * 2;

  CK(cuInit(0));
  CUdevice dev;
  CK(cuDeviceGet(&dev, 0));
  CUcontext ctx;
  CK(cuDevicePrimaryCtxRetain(&ctx, dev));
  CK(cuCtxSetCurrent(ctx));

  CUmodule mod;
  CK(cuModuleLoad(&mod, "spatter_gather.ptx"));
  CUfunction fn;
  CK(cuModuleGetFunction(&fn, mod, "gather_reuse"));

  std::vector<int> idx1, idx2;
  make_idx(idx1, m1, pat, stride);
  make_idx(idx2, m2, pat, stride);

  Buf big = alloc_buf("device", m2, idx2);            // case1
  Buf d_a = alloc_buf("device", m1, idx1);            // case2/3 first half
  Buf d_b = alloc_buf("device", m1, idx1);            // case2 second half
  Buf z_b = alloc_buf("zc", m1, idx1);                // case3 second half

  // One stream for everything: this experiment is explicitly not about streams.
  CUstream s;
  CK(cuStreamCreate(&s, CU_STREAM_NON_BLOCKING));

  const int block = 256;
  auto launch = [&](const Buf& buf, int it) {
    int m = buf.m;
    void* args[] = {(void*)&buf.in, (void*)&buf.idx, (void*)&buf.out, &m, &it};
    int grid = (m + block - 1) / block;
    return cuLaunchKernel(fn, grid, 1, 1, block, 1, 1, 0, s, args, nullptr);
  };

  auto time_ms = [](auto&& body) {
    auto t0 = std::chrono::steady_clock::now();
    body();
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
  };

  // case1: the single 2N buffer, CH launches.
  auto run1 = [&]() {
    for (int c = 0; c < CH; ++c) CK(launch(big, IT));
    CK(cuStreamSynchronize(s));
  };
  // case2/3: alternate the two halves, no sync between launches.
  auto run_pair = [&](const Buf& A, const Buf& B) {
    return [&, A, B]() {
      for (int c = 0; c < CH; ++c) {
        CK(launch(A, IT));
        CK(launch(B, IT));
      }
      CK(cuStreamSynchronize(s));
    };
  };
  auto run2 = run_pair(d_a, d_b);
  auto run3 = run_pair(d_a, z_b);

  for (int w = 0; w < 5; ++w) { run1(); run2(); run3(); }

  std::vector<double> v1, v2, v3;
  for (int t = 0; t < TR; ++t) {
    v1.push_back(time_ms(run1));
    v2.push_back(time_ms(run2));
    v3.push_back(time_ms(run3));
  }

  const double bytes = static_cast<double>(m2) * 4.0 * IT * CH;  // same for all
  const double t1 = median(v1), t2 = median(v2), t3 = median(v3);
  auto gbps = [&](double ms) { return bytes / (ms / 1e3) / 1e9; };

  std::printf("pattern,mb_per_half,total_mb,reuse,chunks,total_bytes,"
              "case1_dev2mb_ms,case2_dev1_dev1_ms,case3_dev1_zc1_ms,"
              "case1_gbps,case2_gbps,case3_gbps,"
              "speedup_case3_vs_case2,speedup_case3_vs_case1\n");
  std::printf("%s,%d,%d,%d,%d,%.0f,%.4f,%.4f,%.4f,%.3f,%.3f,%.3f,%.4f,%.4f\n",
              pat.c_str(), MB, MB * 2, IT, CH, bytes,
              t1, t2, t3, gbps(t1), gbps(t2), gbps(t3), t2 / t3, t1 / t3);
  return 0;
}
