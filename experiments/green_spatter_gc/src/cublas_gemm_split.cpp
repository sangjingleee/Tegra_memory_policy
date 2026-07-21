// cuBLAS GEMM sanity check for Green Context device vs mapped zero-copy paths.
//
// This is not the mechanism benchmark. It is a dense AI primitive check that
// complements the Spatter-style controlled access-pattern experiment.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
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

static const char* cbn(cublasStatus_t s) {
  switch (s) {
    case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
  }
  return "CUBLAS_STATUS_UNKNOWN";
}

#define CK(x)                                                                    \
  do {                                                                           \
    CUresult _r = (x);                                                           \
    if (_r != CUDA_SUCCESS) {                                                    \
      std::fprintf(stderr, "%s:%s\n", #x, cn(_r));                              \
      std::exit(2);                                                              \
    }                                                                            \
  } while (0)

#define CB(x)                                                                    \
  do {                                                                           \
    cublasStatus_t _s = (x);                                                     \
    if (_s != CUBLAS_STATUS_SUCCESS) {                                           \
      std::fprintf(stderr, "%s:%s\n", #x, cbn(_s));                             \
      std::exit(3);                                                              \
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

static double quantile(std::vector<double> v, double q) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  const double pos = q * static_cast<double>(v.size() - 1);
  const size_t lo = static_cast<size_t>(pos);
  const size_t hi = std::min(lo + 1, v.size() - 1);
  const double frac = pos - static_cast<double>(lo);
  return v[lo] * (1.0 - frac) + v[hi] * frac;
}

struct Summary {
  double p05;
  double p50;
  double p95;
};

static Summary summarize(const std::vector<double>& v) {
  return {quantile(v, 0.05), quantile(v, 0.50), quantile(v, 0.95)};
}

static void fill_host(float* p, size_t n, float value) {
  for (size_t i = 0; i < n; ++i) p[i] = value;
}

int main(int argc, char** argv) {
  int N = ia(argc, argv, "n", 1024);
  int TR = ia(argc, argv, "trials", 51);
  int dev_sms_req = ia(argc, argv, "dev-sms", -1);
  int zc_sms_req = ia(argc, argv, "zc-sms", -1);
  if (N <= 0 || TR < 3) {
    std::fprintf(stderr, "n must be positive and trials must be >= 3\n");
    return 2;
  }

  CK(cuInit(0));
  CUdevice dev;
  CK(cuDeviceGet(&dev, 0));
  int smtot = 0;
  CK(cuDeviceGetAttribute(&smtot, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev));
  if (dev_sms_req < 0 || zc_sms_req < 0) {
    dev_sms_req = smtot / 2;
    zc_sms_req = smtot - dev_sms_req;
  }
  if (dev_sms_req <= 0 || zc_sms_req <= 0 || dev_sms_req + zc_sms_req > smtot) {
    std::fprintf(stderr, "invalid SM split %d/%d for total %d\n", dev_sms_req, zc_sms_req, smtot);
    return 2;
  }

  CUcontext pctx;
  CK(cuDevicePrimaryCtxRetain(&pctx, dev));
  CK(cuCtxSetCurrent(pctx));

  const size_t elems = static_cast<size_t>(N) * static_cast<size_t>(N);
  const size_t bytes = elems * sizeof(float);
  std::vector<float> hA(elems, 1.0f), hB(elems, 1.0f);

  CUdeviceptr dA, dB, dC;
  CK(cuMemAlloc(&dA, bytes));
  CK(cuMemAlloc(&dB, bytes));
  CK(cuMemAlloc(&dC, bytes));
  CK(cuMemcpyHtoD(dA, hA.data(), bytes));
  CK(cuMemcpyHtoD(dB, hB.data(), bytes));

  void* zA_h = nullptr;
  void* zB_h = nullptr;
  CUdeviceptr zA, zB, zC;
  CK(cuMemHostAlloc(&zA_h, bytes, CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP));
  CK(cuMemHostAlloc(&zB_h, bytes, CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP));
  fill_host(static_cast<float*>(zA_h), elems, 1.0f);
  fill_host(static_cast<float*>(zB_h), elems, 1.0f);
  CK(cuMemHostGetDevicePointer(&zA, zA_h, 0));
  CK(cuMemHostGetDevicePointer(&zB, zB_h, 0));
  CK(cuMemAlloc(&zC, bytes));

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

  cublasHandle_t h0, h1;
  CK(cuCtxSetCurrent(c0));
  CB(cublasCreate(&h0));
  CB(cublasSetStream(h0, reinterpret_cast<cudaStream_t>(s0)));
  CK(cuCtxSetCurrent(c1));
  CB(cublasCreate(&h1));
  CB(cublasSetStream(h1, reinterpret_cast<cudaStream_t>(s1)));

  CK(cuCtxSetCurrent(c0));
  CUevent e0a, e0b;
  CK(cuEventCreate(&e0a, 0));
  CK(cuEventCreate(&e0b, 0));
  CK(cuCtxSetCurrent(c1));
  CUevent e1a, e1b;
  CK(cuEventCreate(&e1a, 0));
  CK(cuEventCreate(&e1b, 0));

  const float alpha = 1.0f;
  const float beta = 0.0f;
  const double flops = 2.0 * static_cast<double>(N) * N * N;

  auto gemm = [&](cublasHandle_t h, CUdeviceptr A, CUdeviceptr B, CUdeviceptr C) {
    return cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha,
                       reinterpret_cast<const float*>(A), N,
                       reinterpret_cast<const float*>(B), N,
                       &beta, reinterpret_cast<float*>(C), N);
  };

  auto measure = [&](CUcontext ctx, CUstream st, CUevent a, CUevent b,
                     cublasHandle_t h, CUdeviceptr A, CUdeviceptr B, CUdeviceptr C) -> Summary {
    CK(cuCtxSetCurrent(ctx));
    for (int w = 0; w < 3; ++w) CB(gemm(h, A, B, C));
    CK(cuStreamSynchronize(st));
    std::vector<double> samples;
    for (int t = 0; t < TR; ++t) {
      CK(cuEventRecord(a, st));
      CB(gemm(h, A, B, C));
      CK(cuEventRecord(b, st));
      CK(cuEventSynchronize(b));
      float ms = 0.0f;
      CK(cuEventElapsedTime(&ms, a, b));
      samples.push_back(flops / (ms / 1e3) / 1e9);
    }
    return summarize(samples);
  };

  Summary dev_solo = measure(c0, s0, e0a, e0b, h0, dA, dB, dC);
  Summary zc_solo = measure(c1, s1, e1a, e1b, h1, zA, zB, zC);

  std::vector<double> dev_co_v, zc_co_v, agg_v;
  for (int t = 0; t < TR + 3; ++t) {
    CK(cuCtxSetCurrent(c0));
    CK(cuEventRecord(e0a, s0));
    CB(gemm(h0, dA, dB, dC));
    CK(cuEventRecord(e0b, s0));
    CK(cuCtxSetCurrent(c1));
    CK(cuEventRecord(e1a, s1));
    CB(gemm(h1, zA, zB, zC));
    CK(cuEventRecord(e1b, s1));

    CK(cuCtxSetCurrent(c0));
    CK(cuEventSynchronize(e0b));
    CK(cuCtxSetCurrent(c1));
    CK(cuEventSynchronize(e1b));
    if (t < 3) continue;

    float md = 0.0f, mz = 0.0f;
    CK(cuCtxSetCurrent(c0));
    CK(cuEventElapsedTime(&md, e0a, e0b));
    CK(cuCtxSetCurrent(c1));
    CK(cuEventElapsedTime(&mz, e1a, e1b));
    double dev_g = flops / (md / 1e3) / 1e9;
    double zc_g = flops / (mz / 1e3) / 1e9;
    double common_ms = std::max(md, mz);
    dev_co_v.push_back(dev_g);
    zc_co_v.push_back(zc_g);
    agg_v.push_back((2.0 * flops) / (common_ms / 1e3) / 1e9);
  }

  Summary dev_co = summarize(dev_co_v);
  Summary zc_co = summarize(zc_co_v);
  Summary agg = summarize(agg_v);
  double solo_sum = dev_solo.p50 + zc_solo.p50;

  std::printf("n,trials,dev_sms,zc_sms,dev_solo_p50_gflops,zc_solo_p50_gflops,dev_co_p50_gflops,zc_co_p50_gflops,aggregate_common_p50_gflops,solo_sum_p50_gflops,aggregate_ratio_p50_pct,device_preserve_p50_pct,zc_preserve_p50_pct\n");
  std::printf("%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
              N, TR, dev_sms, zc_sms, dev_solo.p50, zc_solo.p50,
              dev_co.p50, zc_co.p50, agg.p50, solo_sum,
              agg.p50 / solo_sum * 100.0,
              dev_co.p50 / dev_solo.p50 * 100.0,
              zc_co.p50 / zc_solo.p50 * 100.0);
  return 0;
}

