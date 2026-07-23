// Measures the H2D copy cost that the device path pays but the steady-state
// read benchmark never times, so the device(copy+read) vs zero-copy(read-only)
// crossover can be computed.
//
// For each working-set size we time cuMemcpyHtoD from pinned host memory into
// device memory (the copy a cudaMalloc-based pipeline must perform before the
// kernel can read). Zero-copy performs no such copy.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <cuda.h>

#define CK(x)                                                                  \
  do {                                                                         \
    CUresult r = (x);                                                          \
    if (r != CUDA_SUCCESS) {                                                   \
      const char* s = nullptr;                                                 \
      cuGetErrorString(r, &s);                                                 \
      std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, s ? s : "?");     \
      std::exit(1);                                                            \
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

static double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  size_t n = v.size();
  return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

int main(int argc, char** argv) {
  int trials = ia(argc, argv, "trials", 101);

  CK(cuInit(0));
  CUdevice dev;
  CK(cuDeviceGet(&dev, 0));
  CUcontext ctx;
  CK(cuDevicePrimaryCtxRetain(&ctx, dev));
  CK(cuCtxSetCurrent(ctx));

  CUstream st;
  CK(cuStreamCreate(&st, CU_STREAM_NON_BLOCKING));
  CUevent a, b;
  CK(cuEventCreate(&a, CU_EVENT_DEFAULT));
  CK(cuEventCreate(&b, CU_EVENT_DEFAULT));

  const int sizes[] = {1, 2, 4, 8, 16, 32, 64};

  std::printf("size_mb,copy_ms_p50,copy_gbps_p50\n");
  for (int mb : sizes) {
    size_t bytes = static_cast<size_t>(mb) * 1024 * 1024;

    // Pinned host source: the best case for the device path (pageable would be
    // slower still), so the copy cost reported here is a lower bound.
    void* h = nullptr;
    CK(cuMemHostAlloc(&h, bytes, CU_MEMHOSTALLOC_PORTABLE));
    std::memset(h, 1, bytes);
    CUdeviceptr d;
    CK(cuMemAlloc(&d, bytes));

    for (int w = 0; w < 3; ++w) CK(cuMemcpyHtoDAsync(d, h, bytes, st));
    CK(cuStreamSynchronize(st));

    std::vector<double> ms_samples;
    for (int t = 0; t < trials; ++t) {
      CK(cuEventRecord(a, st));
      CK(cuMemcpyHtoDAsync(d, h, bytes, st));
      CK(cuEventRecord(b, st));
      CK(cuEventSynchronize(b));
      float ms = 0.0f;
      CK(cuEventElapsedTime(&ms, a, b));
      ms_samples.push_back(ms);
    }
    double ms = median(ms_samples);
    std::printf("%d,%.6f,%.3f\n", mb, ms, bytes / (ms / 1e3) / 1e9);
    std::fflush(stdout);

    CK(cuMemFree(d));
    CK(cuMemFreeHost(h));
  }
  return 0;
}
