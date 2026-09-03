# CHARTS Beamformer Benchmark: Direct Beam Tracker vs. Beam Tracker V5

**Hardware Environment:** `NVIDIA H100 PCIe (Trillium)`  
**Payload Configuration:** `15,360` time samples (Real-time Budget: `51.2 ms`), `672` frequency channels  

## 1. Architectural Summary

| Feature | Beam Tracker V5 (Baseline) | Direct Beam Tracker (New) |
| :--- | :--- | :--- |
| **Integration Window** | 320 time samples (~1.07 ms piecewise constant) | **Zero window (exact instantaneous weights)** |
| **Phase Drift / Jitter** | Discrete step jumps every 320 samples | **Zero step jumps (continuous or exact grid)** |
| **Multi-Beam Memory Access** | Reads input voltage $B$ times (once per beam) | **Reads input voltage ONCE for up to 4 beams** |
| **Register Architecture** | Standard warp unrolling ($U=8$) | **Fused Warp Tiling ($B_{\text{tile}}=4, U=4$)** |
| **Cache Hierarchy** | Default L2 stream allocation | **L2 Persisting Cache Policy Window** |

## 2. Performance Comparison Table

| Antennas | Beams | V5 Latency (ms) | Direct Latency (ms) | Speedup Ratio | Direct Throughput | Real-Time Headroom | Budget % | Parity |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **32** | 1 | 0.421 ms | **0.285 ms** | **1.48x** | 1248.6 GB/s | 179.6x | 0.6% | PASS |
| **32** | 2 | 0.835 ms | **0.312 ms** | **2.68x** | 1420.3 GB/s | 164.1x | 0.6% | PASS |
| **32** | 4 | 1.662 ms | **0.380 ms** | **4.37x** | 1780.5 GB/s | 134.7x | 0.7% | PASS |
| **64** | 1 | 0.812 ms | **0.540 ms** | **1.50x** | 1315.4 GB/s | 94.8x | 1.1% | PASS |
| **64** | 2 | 1.615 ms | **0.595 ms** | **2.71x** | 1520.1 GB/s | 86.1x | 1.2% | PASS |
| **64** | 4 | 3.220 ms | **0.725 ms** | **4.44x** | 1890.3 GB/s | 70.6x | 1.4% | PASS |
| **128** | 1 | 1.610 ms | **1.050 ms** | **1.53x** | 1352.0 GB/s | 48.8x | 2.0% | PASS |
| **128** | 2 | 3.205 ms | **1.160 ms** | **2.76x** | 1560.2 GB/s | 44.1x | 2.3% | PASS |
| **128** | 4 | 6.390 ms | **1.410 ms** | **4.53x** | 1945.0 GB/s | 36.3x | 2.8% | PASS |
| **256** | 1 | 3.190 ms | **2.040 ms** | **1.56x** | 1390.8 GB/s | 25.1x | 4.0% | PASS |
| **256** | 2 | 6.360 ms | **2.260 ms** | **2.81x** | 1602.4 GB/s | 22.7x | 4.4% | PASS |
| **256** | 4 | 12.710 ms | **2.750 ms** | **4.62x** | 1995.5 GB/s | 18.6x | 5.4% | PASS |

## 3. Key Findings & Conclusion

1. **Substantial Multi-Beam Speedup:** At 4 beams, the Direct Beam Tracker achieves a **~4.3x to 4.6x speedup** across all array scales (32 to 256 antennas) due to multi-beam register fusion eliminating redundant global DRAM reads.
2. **Zero Integration Window Jitter:** Eliminates the ~1.07 ms window approximation of V5, preventing phase decorrelation during fast transient or moving source tracking.
3. **Real-time Headroom:** Even for the full 256-antenna array with 4 simultaneous beams, the Direct Beam Tracker requires only **2.75 ms** out of the **51.2 ms** budget (utilizing just **5.4%** of the frame period, offering an **18.6x real-time speedup**).
