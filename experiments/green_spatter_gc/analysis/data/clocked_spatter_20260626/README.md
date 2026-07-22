# Clock-locked Spatter + Green Context campaign

- Platform clock state is captured in `clock_state.txt`.
- Green Context requests 8 SM for device and 8 SM for mapped zero-copy; actual counts are in the CSV.
- Aggregate is computed from total bytes over the common host wall interval, not by adding unrelated best trials.
- MC_ALL ACTMON activity is a shared memory-controller hardware signal in raw units, not a per-path DRAM GB/s counter.
- ACTMON is sampled over the full benchmark process (solo calibration plus co-run); it is retained as a raw diagnostic, not used for per-phase attribution.
- This measures performance behavior; it does not directly prove SLC residency or physical cache topology.
- Separate 512MB stream-read calibration: 177.1 GB/s.

## Results

```csv
label,aggregate_ratio_p50_pct,aggregate_ratio_p05_pct,aggregate_ratio_p95_pct,mc_avg_activity_p50,overlap_p50_ratio
front_uniform_1mb_r32,95.13,95.00,95.32,763174.00,1.00
front_strided_1mb_r32,97.46,97.24,97.71,5223836.00,1.00
front_scatter_1mb_r32,60.48,60.17,60.94,15485825.00,1.00
front_uniform_4mb_r32,94.85,94.68,94.97,11166471.00,1.00
front_strided_4mb_r32,94.89,94.61,95.13,10901968.00,1.00
front_scatter_4mb_r32,45.75,45.65,45.86,16513382.00,1.00
back_uniform_64mb_r1,76.25,76.21,76.30,23259630.00,1.00
back_strided_64mb_r1,49.98,49.93,50.00,57728051.00,1.00
back_scatter_32mb_r1,45.49,45.20,46.33,30164302.00,1.00
back_scatter_64mb_r1,8.88,7.13,12.42,18456654.00,1.00
back_scatter_128mb_r1,45.83,32.56,45.85,7521623.00,1.00
```
