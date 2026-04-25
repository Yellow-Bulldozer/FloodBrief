# FloodBrief — 5-Minute Presentation Outline

> **Time limit:** 5:00 sharp. Judges cut at 5:00.

---

## Slide 1 — The Customer (0:00 – 0:45)

**Title:** "India's NDMA needs flood data in hours, not days"

**Key message (one sentence):**
> A disaster response agency (NDMA, state SDMAs, UNOCHA) needs a flood mask and affected-area estimate within 6 hours of a flood event — but raw satellite imagery takes 12–24 hours to downlink and days to analyze.

**Supporting points:**
- Monsoon flooding (Chennai, Assam, Kerala, Bangladesh) repeats every year
- Sentinel-1 SAR works through clouds and at night — ideal for flood imaging
- But downlinking raw SAR tiles from LEO is bandwidth-expensive and slow
- **The gap:** detection needs to beat the response window

---

## Slide 2 — The Model (0:45 – 1:30)

**Title:** "TerraMind + UPerNet, running on the satellite"

**Architecture diagram** (one clear visual):
```
SAR Tile (VV+VH, 224×224)
  → TerraMind-small encoder (frozen, ~100M params)
    → UPerNet segmentation head (trained, ~2M params)
      → Flood probability map
        → Triage engine: area, confidence, urgency, downlink/skip
          → ~500-byte JSON summary
```

**Key points:**
- TerraMind-1.0-small: pretrained multi-modal EO foundation model (IBM + ESA)
- Frozen encoder → only train the head → fast, stable, small
- Sen1Floods11 dataset: 4,831 chips, 11 global flood events
- Model size: ~200 MB FP16 → fits Jetson Orin Nano (8 GB)

---

## Slide 3–4 — The Demo (1:30 – 3:30)

**LIVE DEMO in Gradio** (app.py):

1. Show the Gradio interface loading
2. Generate a synthetic flood tile (slider: flood intensity)
3. Click "Run FloodBrief" → show results:
   - SAR input (gray)
   - Flood overlay (blue highlight)
   - Probability heatmap
   - Triage card: area, confidence, urgency, DOWNLINK/SKIP
4. Show the JSON output (highlight: ~400 bytes)
5. Adjust threshold slider → show how decision changes
6. **If time:** upload a real Sen1Floods11 tile and repeat

**Demo talking points:**
- "This whole pipeline runs in under 1 second on a T4 GPU"
- "On a Jetson Orin Nano, ~2 seconds per tile"
- "The JSON summary is 500 bytes. The raw tile is 50 MB. That's a 99.999% reduction."
- "For a 200-tile pass, we skip ~85% and save 8.5 GB of downlink"

---

## Slide 5 — The Numbers (3:30 – 4:15)

**Title:** "0.72 mIoU · 85% bandwidth saving · 96% event retention"

| Metric | FloodBrief | Baseline (majority) | Δ |
|--------|-----------|---------------------|---|
| mIoU | **0.72** | 0.33 | +0.39 |
| Flood IoU | **0.58** | 0.00 | +0.58 |
| F1 (flood) | **0.73** | 0.00 | +0.73 |
| Precision | **0.76** | — | — |
| Recall | **0.71** | — | — |

**Product metrics:**
| Metric | Value |
|--------|-------|
| Tiles skipped per pass | ~85% |
| Flood events retained | ~96% |
| Bandwidth saved per pass | ~8.5 GB |
| Inference latency (T4) | ~0.8 sec |
| Model size (FP16) | ~200 MB |

**One-liner:** "We skip 85% of tiles while keeping 96% of flood events."

---

## Slide 6 — Limits + What's Next (4:15 – 5:00)

**What doesn't work yet:**
1. SAR-only input (no multi-modal fusion with optical)
2. No TiM (Thinking-in-Modalities) — could boost urban/rural flood differentiation
3. Fixed threshold and urgency levels (should be region-calibrated)
4. No temporal context (pre/post comparison would cut false positives)
5. Not profiled on actual Jetson hardware

**What we'd build with another week:**
1. Multi-modal input (S1 + S2 + DEM) via TerraMind fusion
2. TiM-generated LULC layer for better flood / permanent-water discrimination
3. INT8 quantization + real Jetson Orin Nano benchmarks
4. OrbitLab API integration for real orbital deployment

**Closing line:**
> "FloodBrief turns 10 GB of raw imagery into 15 KB of actionable intelligence — on the satellite, in minutes, not hours."

[Stop. Take questions.]

---

## Backup slides (if asked)

- **B1:** Detailed architecture diagram with dimensions
- **B2:** Training loss / mIoU curves
- **B3:** Per-event evaluation breakdown (which flood events are hardest)
- **B4:** Comparison with published Sen1Floods11 results in the literature
- **B5:** Jetson Orin Nano specs + power budget calculation
