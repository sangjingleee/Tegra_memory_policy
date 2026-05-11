# Next Experiment Plan After L2 Cache Probe

## Current Confirmed Result

The 0511 microbenchmark confirmed that device memory and mapped zero-copy have very different GPU L2 cache reuse behavior:

```text
device path   : L2 hit rate about 94%
zero-copy path: L2 hit rate about 6%
```

The next experiments should test whether this difference matters under co-running contention.

## Phase 1. Co-running Microbenchmark

Goal:

```text
Does a zero-copy co-run workload protect a device-memory victim better than a device-memory co-run workload?
```

Setup:

```text
victim: device path, latency measured
co-run workload A: device path infinite loop
co-run workload B: zero-copy path infinite loop
```

Measure:

```text
victim latency
victim L2 hit rate if NCU can isolate it
victim duration under no co-run / device co-run / zero-copy co-run
```

Expected useful result:

```text
device co-run    -> victim latency increases more
zero-copy co-run -> victim latency increases less
```

This would support the claim that mapped zero-copy changes cache reuse/pollution behavior.

## Phase 2. Access Pattern Sweep

The current kernel is reuse-heavy. Add at least three access patterns:

1. `reuse_read`: current repeated read pattern.
2. `stream_read`: linear one-pass read with little reuse.
3. `random_stride_read`: larger stride to reduce locality.
4. `write_heavy`: output-dominant kernel.

For each pattern:

```text
device vs zero-copy
working set: 1MB, 4MB, 16MB, 64MB
metrics: L2 hit rate, L2 sectors, duration
```

Question:

```text
Is zero-copy's low L2 hit behavior universal, or only visible for reuse-heavy reads?
```

## Phase 3. Mixed Policy Microbenchmark

Goal:

```text
Find how much co-run workload can be moved to zero-copy while preserving its own latency and protecting victim latency.
```

Policies:

```text
all_device
all_zerocopy
mixed_25_zc
mixed_50_zc
mixed_75_zc
size_based_mixed
```

Measure both sides:

```text
victim latency improvement
co-run workload slowdown
```

Accept rule candidate:

```text
Use zero-copy for co-run tensors only if victim protection gain is meaningful and co-run slowdown is acceptable.
```

## Phase 4. Model-level Validation

Only after microbenchmark phases are clear, validate on real models.

Recommended first pair:

```text
victim: MobileNetV2 or GPT2 in device/default path
co-run: CNN workload with device vs zero-copy/mixed allocator
```

Measure:

```text
victim latency
co-run latency
policy cost/gain tradeoff
```

Do not start with many models. First prove the cache-pollution mechanism with one stable pair.

## Phase 5. General Policy

After microbenchmark and one model pair:

```text
features:
  working-set size
  reuse pattern
  read/write ratio
  victim priority
  co-run slowdown tolerance

decision:
  device vs zero-copy vs mixed
```

The policy should not be "zero-copy is faster." The safer target is:

```text
Use zero-copy selectively for lower-priority co-running work when its low L2 hit/reuse behavior reduces victim interference more than it hurts the co-run workload.
```
