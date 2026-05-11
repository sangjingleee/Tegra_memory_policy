# 0511 GPU L2 Cache Probe

## Goal

This microbenchmark checks whether the CUDA device-memory path and mapped zero-copy path use GPU L2 cache differently on Orin-A.

The experiment intentionally avoids PyTorch, TensorRT, and model kernels. It uses one CUDA kernel and changes only the input memory path.

## Memory Paths

`device`:

```text
malloc host -> cudaMalloc device -> cudaMemcpy H2D -> read_kernel
```

`zerocopy`:

```text
cudaHostAllocMapped -> cudaHostGetDevicePointer -> read_kernel
```

## Kernel

`read_kernel` repeatedly reads the input tensor:

```cpp
int idx = (tid + r * 131) % n;
acc += in[idx];
```

This access pattern creates reuse pressure so the device-memory path can expose L2 cache reuse.

## NCU Results

| condition | L2 hit % | L2 sectors | L2 read sectors | duration |
| --- | ---: | ---: | ---: | ---: |
| device 1MB | 94.17 | 562,205 | 529,104 | 547 us |
| zerocopy 1MB | 5.94 | 560,733 | 527,632 | 383 us |
| device 64MB | 94.12 | 35,656,901 | 33,559,416 | 34.0 ms |
| zerocopy 64MB | 5.88 | 35,655,173 | 33,557,688 | 34.2 ms |

## Interpretation

The device path has a very high L2 hit rate, around 94%, for both 1MB and 64MB working sets. This means that repeated reads from `cudaMalloc` device memory reuse GPU L2 cache effectively in this kernel.

The mapped zero-copy path has a very low L2 hit rate, around 6%, under the same kernel and similar L2 sector counts. This means mapped zero-copy input is not cached/reused like device memory.

Important nuance:

```text
Zero-copy does not reduce L2 sector count in this result.
It changes cache reuse behavior: sectors are similar, but hits collapse.
```

So the safe claim is:

> Device memory and mapped zero-copy show clearly different GPU L2 cache reuse behavior. Device memory is L2-hit-heavy, while mapped zero-copy produces very low L2 hit rate under the same reuse-heavy read kernel.

Do not claim yet:

> Zero-copy reduces L2 traffic.

The next step is to test whether this different cache-reuse behavior reduces cache pollution for a co-running victim workload.

## Reproduction

Compile on Orin-A:

```bash
cd /home2/sangjin_work/slc_bench
/usr/local/cuda-11.4/bin/nvcc -O3 -lineinfo l2_bench_v2.cu -o l2_bench_v2
```

Run NCU:

```bash
sudo /opt/nvidia/nsight-compute/2024.3.0/ncu \
  --replay-mode application \
  --metrics lts__t_sector_hit_rate.pct,lts__t_sectors.sum,lts__t_sectors_op_read.sum,dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum \
  --kernel-name regex:"read_kernel" \
  --launch-count 1 \
  --csv \
  ./l2_bench_v2 device 1
```

Repeat for:

```text
device 1
zerocopy 1
device 64
zerocopy 64
```
