// l2_bench_v2.cu
// Compare GPU L2 behavior for device memory vs mapped zero-copy input.
//
// Usage:
//   ./l2_bench_v2 <device|zerocopy> <mb>
//
// The kernel is intentionally identical across modes. Only the input memory
// path changes:
//   device   : malloc host -> cudaMalloc device -> cudaMemcpy H2D -> kernel read
//   zerocopy : cudaHostAllocMapped -> cudaHostGetDevicePointer -> kernel read
//
// The access pattern repeats reads over the working set to expose cache reuse.

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); return 1; }} while(0)

__global__ void read_kernel(const float* in, float* out, int n, int iters) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;
  float acc = 0.0f;
  for (int r = 0; r < iters; r++) {
    int idx = (tid + r * 131) % n;
    acc += in[idx];
  }
  out[tid] = acc;
}

int main(int argc, char** argv) {
  const char* mode = argc > 1 ? argv[1] : "device";
  int mb           = argc > 2 ? atoi(argv[2]) : 1;
  int iters        = 64;

  int n = (mb * 1024 * 1024) / sizeof(float);
  size_t bytes = (size_t)n * sizeof(float);

  CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));

  float *h = NULL, *in = NULL, *out = NULL;
  CHECK(cudaMalloc(&out, bytes));

  if (!strcmp(mode, "device")) {
    h = (float*)malloc(bytes);
    for (int i=0; i<n; i++) h[i] = 1.0f;
    CHECK(cudaMalloc(&in, bytes));
    CHECK(cudaMemcpy(in, h, bytes, cudaMemcpyHostToDevice));
  } else if (!strcmp(mode, "zerocopy")) {
    float *host = NULL, *dev = NULL;
    CHECK(cudaHostAlloc((void**)&host, bytes, cudaHostAllocMapped | cudaHostAllocPortable));
    for (int i=0; i<n; i++) host[i] = 1.0f;
    CHECK(cudaHostGetDevicePointer((void**)&dev, host, 0));
    h = host;
    in = dev;
  } else {
    fprintf(stderr, "usage: %s device|zerocopy <mb>\n", argv[0]);
    return 2;
  }

  dim3 block(256);
  dim3 grid((n + block.x - 1) / block.x);

  // Warm up before the measured kernel.
  for (int w=0; w<10; w++) {
    read_kernel<<<grid, block>>>(in, out, n, iters);
  }
  CHECK(cudaDeviceSynchronize());

  read_kernel<<<grid, block>>>(in, out, n, iters);
  CHECK(cudaDeviceSynchronize());

  printf("mode=%s mb=%d n=%d\n", mode, mb, n);

  if (!strcmp(mode, "device")) {
    CHECK(cudaFree(in));
    free(h);
  } else {
    CHECK(cudaFreeHost(h));
  }
  CHECK(cudaFree(out));
  return 0;
}
