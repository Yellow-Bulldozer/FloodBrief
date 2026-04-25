# FloodBrief — Edge Inference Feasibility Analysis

## Target Hardware

**Nvidia Jetson Orin Nano** — the class of compute on TakeMe2Space's upcoming 6U cubesat.

| Spec | Value |
|------|-------|
| GPU | Ampere, 1024 CUDA cores |
| AI Performance | 40 TOPS (INT8) |
| Memory | 8 GB shared (LPDDR5) |
| Power | 7–15 W |
| Storage | MicroSD / NVMe |
| OS | JetPack / Linux |

## FloodBrief Model Profile

### TerraMind-small encoder + UPerNet head

| Metric | FP16 | INT8 (estimated) |
|--------|------|-------------------|
| **Weights size** | ~200 MB | ~100 MB |
| **Peak RAM (batch 1, 224×224)** | ~1.5 GB | ~0.8 GB |
| **GFLOPs per inference** | ~25 GFLOPs | ~25 GFLOPs |
| **Estimated latency (Jetson)** | ~2.0 sec | ~1.2 sec |

### Memory budget

```
Available: 8 GB shared RAM
  - OS + JetPack:      ~1.5 GB
  - Model weights:     ~0.2 GB (FP16)
  - Activation memory: ~0.5 GB (batch 1)
  - Input buffer:      ~0.4 MB (1 tile)
  - Output buffer:     ~0.2 MB (mask + JSON)
  - Headroom:          ~5.4 GB
  
Verdict: ✅ Fits comfortably
```

### Throughput analysis

A typical LEO pass over a flood-prone region captures ~200 tiles. At the estimated Jetson latency:

| Scenario | Latency/tile | 200 tiles | Verdict |
|----------|-------------|-----------|---------|
| FP16 | ~2.0 sec | ~400 sec (~7 min) | ✅ Manageable |
| INT8 | ~1.2 sec | ~240 sec (~4 min) | ✅ Good |
| Tiled (512→224 patches) | ~4.0 sec | ~800 sec (~13 min) | ⚠️ Tight |

LEO orbital periods are ~90 min. Even the worst case (13 min for 200 tiles) leaves ample time for ground-station contact windows.

## Bandwidth savings

| Item | Size |
|------|------|
| Raw SAR tile (Sentinel-1, 10 m, ~1700×1700) | ~50 MB |
| FloodBrief JSON summary | ~500 bytes |
| Reduction per skipped tile | **~99.999%** |

For a 200-tile pass with 85% skip rate (170 tiles skipped):
- **Without FloodBrief:** 200 × 50 MB = **10 GB downlink**
- **With FloodBrief:** 30 × 50 MB + 200 × 500 B ≈ **1.5 GB downlink + 100 KB metadata**

→ **85% bandwidth saving** (or 8.5 GB saved per pass)

At TakeMe2Space's ~$2/minute compute rate, FloodBrief pays for itself by dramatically reducing the ground-station time needed for data downlink.

## Power budget

| Component | Power draw |
|-----------|-----------|
| Jetson Orin Nano (active inference) | ~15 W |
| Inference duration per pass | ~400 sec |
| Energy per pass | ~6 kJ (~1.7 Wh) |
| 6U cubesat typical battery | ~40–60 Wh |
| Fraction of battery per pass | ~3–4% |

→ ✅ **Well within 6U power budget.**

## Production readiness gap

What's needed for actual flight deployment:

1. **INT8 quantization** — TensorRT on Jetson. Straightforward for ViT + UPerNet, ~50% latency savings expected.
2. **ONNX export** — For TensorRT optimization.
3. **Radiation testing** — Jetson Orin Nano needs radiation hardening or error-correction firmware.
4. **Thermal validation** — Vacuum thermal cycling.
5. **Tiled inference** — For full-resolution tiles larger than 224×224.
6. **Fallback logic** — If inference fails, default to "downlink" (fail-safe).

## Conclusion

FloodBrief's TerraMind-small + UPerNet architecture maps cleanly onto the Jetson Orin Nano:
- Fits in 8 GB RAM (using <2 GB)
- Processes a full pass in ~7 min (FP16) or ~4 min (INT8)
- Uses <4% of a 6U battery budget per pass
- Saves ~85% of downlink bandwidth

The remaining gap is engineering (quantization, ONNX, radiation) rather than computational. The model itself is feasible for orbital deployment today.
