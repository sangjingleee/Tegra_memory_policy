// Measured peak DRAM bandwidth on this board (anchor for the aggregation claim).
// Large buffer (>> caches) streamed -> DRAM-bound. Read & copy, median of 30.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

__global__ void streamRead(const float4* in, long n, float* out) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  long stride = (long)blockDim.x * gridDim.x;
  float4 acc = make_float4(0, 0, 0, 0);
  for (long k = i; k < n; k += stride) {
    float4 v = in[k]; acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
  }
  if (i == 0) out[0] = acc.x + acc.y + acc.z + acc.w;
}
__global__ void streamCopy(const float4* in, float4* out, long n) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  long stride = (long)blockDim.x * gridDim.x;
  for (long k = i; k < n; k += stride) out[k] = in[k];
}
static double med(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  size_t n = v.size();
  return n ? (n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2])) : 0;
}
int main(int argc, char** argv) {
  long MB = argc > 1 ? atol(argv[1]) : 512;
  long bytes = MB * 1024 * 1024;
  long n = bytes / sizeof(float4);
  float4 *in, *out; float* res;
  cudaMalloc(&in, bytes); cudaMalloc(&out, bytes); cudaMalloc(&res, sizeof(float));
  cudaMemset(in, 1, bytes);
  cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
  int block = 256, grid = p.multiProcessorCount * 32;
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  for (int w = 0; w < 5; w++) streamRead<<<grid, block>>>(in, n, res);
  cudaDeviceSynchronize();
  std::vector<double> rd, cp;
  for (int t = 0; t < 30; t++) {
    cudaEventRecord(a); streamRead<<<grid, block>>>(in, n, res); cudaEventRecord(b);
    cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms, a, b);
    rd.push_back(bytes / (ms / 1e3) / 1e9);
  }
  for (int t = 0; t < 30; t++) {
    cudaEventRecord(a); streamCopy<<<grid, block>>>(in, out, n); cudaEventRecord(b);
    cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms, a, b);
    cp.push_back(2.0 * bytes / (ms / 1e3) / 1e9);
  }
  printf("buffer=%ld MB  (DRAM-bound)\n", MB);
  printf("stream READ peak : %.1f GB/s\n", med(rd));
  printf("stream COPY peak : %.1f GB/s (rd+wr)\n", med(cp));
  printf("spec (AGX Orin)  : 204.8 GB/s\n");
  return 0;
}
