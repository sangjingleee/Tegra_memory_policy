#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

__global__ void scale_kernel(const float* in, float* out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = in[idx] * 1.0001f + 0.5f;
}

static void check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(err));
        exit(1);
    }
}

static float elapsed_ms(cudaEvent_t a, cudaEvent_t b) {
    float ms = 0.0f;
    check(cudaEventElapsedTime(&ms, a, b), "cudaEventElapsedTime");
    return ms;
}

static void fill_host(float* p, size_t n) {
    for (size_t i = 0; i < n; ++i) p[i] = (float)(i % 1024) * 0.001f;
}

static void bench_copy_mode(const char* mode, size_t n, int repeats, int pinned) {
    const size_t bytes = n * sizeof(float);
    float *h_in = NULL, *h_out = NULL, *d_in = NULL, *d_out = NULL;
    if (pinned) {
        check(cudaHostAlloc(&h_in, bytes, cudaHostAllocDefault), "cudaHostAlloc h_in");
        check(cudaHostAlloc(&h_out, bytes, cudaHostAllocDefault), "cudaHostAlloc h_out");
    } else {
        h_in = (float*)malloc(bytes);
        h_out = (float*)malloc(bytes);
        if (!h_in || !h_out) {
            fprintf(stderr, "malloc failed\n");
            exit(1);
        }
    }
    fill_host(h_in, n);
    check(cudaMalloc(&d_in, bytes), "cudaMalloc d_in");
    check(cudaMalloc(&d_out, bytes), "cudaMalloc d_out");

    cudaEvent_t s, after_h2d, after_kernel, e;
    check(cudaEventCreate(&s), "event s");
    check(cudaEventCreate(&after_h2d), "event after_h2d");
    check(cudaEventCreate(&after_kernel), "event after_kernel");
    check(cudaEventCreate(&e), "event e");
    int block = 256;
    int grid = (int)((n + block - 1) / block);
    double h2d = 0.0, kernel = 0.0, d2h = 0.0, total = 0.0;
    for (int i = 0; i < repeats; ++i) {
        check(cudaEventRecord(s), "record s");
        check(cudaMemcpyAsync(d_in, h_in, bytes, cudaMemcpyHostToDevice), "H2D");
        check(cudaEventRecord(after_h2d), "record after_h2d");
        scale_kernel<<<grid, block>>>(d_in, d_out, n);
        check(cudaGetLastError(), "scale_kernel");
        check(cudaEventRecord(after_kernel), "record after_kernel");
        check(cudaMemcpyAsync(h_out, d_out, bytes, cudaMemcpyDeviceToHost), "D2H");
        check(cudaEventRecord(e), "record e");
        check(cudaEventSynchronize(e), "sync e");
        h2d += elapsed_ms(s, after_h2d);
        kernel += elapsed_ms(after_h2d, after_kernel);
        d2h += elapsed_ms(after_kernel, e);
        total += elapsed_ms(s, e);
    }
    printf("RESULT mode=%s bytes=%zu repeats=%d h2d_ms=%.6f kernel_ms=%.6f d2h_ms=%.6f total_ms=%.6f memcpy_count_per_iter=2\n",
           mode, bytes, repeats, h2d / repeats, kernel / repeats, d2h / repeats, total / repeats);

    cudaFree(d_in);
    cudaFree(d_out);
    if (pinned) {
        cudaFreeHost(h_in);
        cudaFreeHost(h_out);
    } else {
        free(h_in);
        free(h_out);
    }
}

static void bench_mapped_zc(size_t n, int repeats) {
    const size_t bytes = n * sizeof(float);
    float *h_in = NULL, *h_out = NULL, *d_in = NULL, *d_out = NULL;
    check(cudaHostAlloc(&h_in, bytes, cudaHostAllocMapped), "cudaHostAllocMapped h_in");
    check(cudaHostAlloc(&h_out, bytes, cudaHostAllocMapped), "cudaHostAllocMapped h_out");
    check(cudaHostGetDevicePointer(&d_in, h_in, 0), "cudaHostGetDevicePointer d_in");
    check(cudaHostGetDevicePointer(&d_out, h_out, 0), "cudaHostGetDevicePointer d_out");
    fill_host(h_in, n);
    int block = 256;
    int grid = (int)((n + block - 1) / block);
    cudaEvent_t s, e;
    check(cudaEventCreate(&s), "event s");
    check(cudaEventCreate(&e), "event e");
    double total = 0.0;
    for (int i = 0; i < repeats; ++i) {
        check(cudaEventRecord(s), "record s");
        scale_kernel<<<grid, block>>>(d_in, d_out, n);
        check(cudaGetLastError(), "scale_kernel");
        check(cudaEventRecord(e), "record e");
        check(cudaEventSynchronize(e), "sync e");
        total += elapsed_ms(s, e);
    }
    printf("RESULT mode=mapped_zc bytes=%zu repeats=%d h2d_ms=0.000000 kernel_ms=%.6f d2h_ms=0.000000 total_ms=%.6f memcpy_count_per_iter=0\n",
           bytes, repeats, total / repeats, total / repeats);
    cudaFreeHost(h_in);
    cudaFreeHost(h_out);
}

static void bench_managed(size_t n, int repeats) {
    const size_t bytes = n * sizeof(float);
    float *in = NULL, *out = NULL;
    check(cudaMallocManaged(&in, bytes), "cudaMallocManaged in");
    check(cudaMallocManaged(&out, bytes), "cudaMallocManaged out");
    fill_host(in, n);
    check(cudaDeviceSynchronize(), "sync after host fill");
    int block = 256;
    int grid = (int)((n + block - 1) / block);
    cudaEvent_t s, e;
    check(cudaEventCreate(&s), "event s");
    check(cudaEventCreate(&e), "event e");
    double total = 0.0;
    for (int i = 0; i < repeats; ++i) {
        check(cudaEventRecord(s), "record s");
        scale_kernel<<<grid, block>>>(in, out, n);
        check(cudaGetLastError(), "scale_kernel");
        check(cudaEventRecord(e), "record e");
        check(cudaEventSynchronize(e), "sync e");
        total += elapsed_ms(s, e);
    }
    printf("RESULT mode=managed bytes=%zu repeats=%d h2d_ms=0.000000 kernel_ms=%.6f d2h_ms=0.000000 total_ms=%.6f memcpy_count_per_iter=0\n",
           bytes, repeats, total / repeats, total / repeats);
    cudaFree(in);
    cudaFree(out);
}

static void bench_device_preloaded(size_t n, int repeats) {
    const size_t bytes = n * sizeof(float);
    float *h = (float*)malloc(bytes);
    float *d_in = NULL, *d_out = NULL;
    if (!h) {
        fprintf(stderr, "malloc h failed\n");
        exit(1);
    }
    fill_host(h, n);
    check(cudaMalloc(&d_in, bytes), "cudaMalloc d_in");
    check(cudaMalloc(&d_out, bytes), "cudaMalloc d_out");
    check(cudaMemcpy(d_in, h, bytes, cudaMemcpyHostToDevice), "preload H2D");
    int block = 256;
    int grid = (int)((n + block - 1) / block);
    cudaEvent_t s, e;
    check(cudaEventCreate(&s), "event s");
    check(cudaEventCreate(&e), "event e");
    double total = 0.0;
    for (int i = 0; i < repeats; ++i) {
        check(cudaEventRecord(s), "record s");
        scale_kernel<<<grid, block>>>(d_in, d_out, n);
        check(cudaGetLastError(), "scale_kernel");
        check(cudaEventRecord(e), "record e");
        check(cudaEventSynchronize(e), "sync e");
        total += elapsed_ms(s, e);
    }
    printf("RESULT mode=device_preloaded bytes=%zu repeats=%d h2d_ms=0.000000 kernel_ms=%.6f d2h_ms=0.000000 total_ms=%.6f memcpy_count_per_iter=0\n",
           bytes, repeats, total / repeats, total / repeats);
    cudaFree(d_in);
    cudaFree(d_out);
    free(h);
}

int main(int argc, char** argv) {
    size_t bytes = 4ULL * 1024 * 1024;
    int repeats = 200;
    if (argc > 1) bytes = (size_t)atoll(argv[1]);
    if (argc > 2) repeats = atoi(argv[2]);
    size_t n = bytes / sizeof(float);
    if (n == 0) n = 1;
    bytes = n * sizeof(float);
    bench_copy_mode("pageable_copy", n, repeats, 0);
    bench_copy_mode("pinned_copy", n, repeats, 1);
    bench_mapped_zc(n, repeats);
    bench_managed(n, repeats);
    bench_device_preloaded(n, repeats);
    return 0;
}
