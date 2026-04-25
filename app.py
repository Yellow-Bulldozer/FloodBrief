"""
FloodBrief - Gradio demo app.

Interactive web interface for flood detection on Sentinel-1 SAR imagery.
Upload a tile or use a synthetic demo tile to see:
  - Flood probability map
  - Binary flood mask overlay
  - Triage summary (area, confidence, urgency, downlink decision)
  - Compact JSON output

Usage:
    python app.py
    # Opens at http://localhost:7860
"""

import os
import sys
import io
import json
import time
import tempfile
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.model import load_model
from src.triage import triage_tile
from src.metrics import InferenceTimer
from src.visualization import plot_flood_overlay

# Try importing gradio
try:
    import gradio as gr
except ImportError:
    print("ERROR: gradio not installed. Install with: pip install gradio>=4.0.0")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = None


def get_model():
    """Lazy-load the model (so startup is fast if just checking the UI)."""
    global MODEL
    if MODEL is None:
        # Try to load from checkpoint, fall back to fresh model
        checkpoint = None
        if os.path.exists("./checkpoints/best_model.pt"):
            checkpoint = "./checkpoints/best_model.pt"
        MODEL = load_model(checkpoint_path=checkpoint, device=DEVICE)
    return MODEL




# ---------------------------------------------------------------------------
# Main inference function (called by Gradio)
# ---------------------------------------------------------------------------

def run_inference(
    upload_file,
    threshold: float,
    tile_id: str,
):
    """
    Run FloodBrief inference on an uploaded tile.

    Returns:
       (visualization_fig, json_summary, triage_text, mask_image)
    """
    # Fix memory leak from Gradio creating too many pyplot figures
    plt.close('all')

    img_size = 224
    model = get_model()

    # --- Load tile ---
    if upload_file is None:
        return None, "{}", "## ❌ Error\nPlease upload a Sentinel-1 SAR GeoTIFF file.", None
    else:
        # Load uploaded file
        try:
            from src.data_loader import Sen1Floods11Dataset
            tile = Sen1Floods11Dataset._load_tif(upload_file)
            if tile.shape[0] > 2:
                tile = tile[:2]
            elif tile.shape[0] < 2:
                tile = np.repeat(tile, 2, axis=0)[:2]

            # Resize
            _, H, W = tile.shape
            min_dim = min(H, W)
            top = (H - min_dim) // 2
            left = (W - min_dim) // 2
            tile = tile[:, top:top + min_dim, left:left + min_dim]

            if min_dim != img_size:
                t = torch.from_numpy(tile).unsqueeze(0)
                t = torch.nn.functional.interpolate(
                    t, size=(img_size, img_size),
                    mode="bilinear", align_corners=False
                )
                tile = t.squeeze(0).numpy()

            # Normalize SAR
            tile = Sen1Floods11Dataset._normalize_s1(tile, method="terramind")
            tile = np.nan_to_num(tile, nan=0.0, posinf=0.0, neginf=0.0)

            if not tile_id:
                tile_id = Path(upload_file).stem

        except Exception as e:
            return None, f"Error loading file: {e}", str(e), None

    # --- Inference ---
    input_tensor = torch.from_numpy(tile).unsqueeze(0).float().to(DEVICE)

    with InferenceTimer() as timer:
        output = model.predict(input_tensor)

    flood_prob = output["flood_probability"][0].cpu().numpy()
    binary_mask = output["binary_mask"][0].cpu().numpy()

    # --- Triage ---
    triage_result = triage_tile(
        flood_probability=flood_prob,
        tile_id=tile_id,
        threshold=threshold,
        inference_latency_ms=timer.elapsed_ms,
    )

    # --- Visualization ---
    fig = plot_flood_overlay(
        sar_image=tile,
        flood_mask=binary_mask,
        flood_probability=flood_prob,
        triage_result=triage_result,
        title=f"FloodBrief - {tile_id}",
    )

    # --- JSON summary ---
    json_str = triage_result.to_json(indent=2)

    # --- Triage text ---
    decision_emoji = "🔴 DOWNLINK" if triage_result.downlink_decision == "downlink" else "🟢 SKIP"
    urgency_emoji = {
        "CRITICAL": "🔴", "HIGH": "🟠", "MODERATE": "🟡", "LOW": "🟢", "NONE": "⚪"
    }.get(triage_result.urgency, "⚪")

    # Fetch latest legit accuracy from validation history
    acc_str = ""
    try:
        with open("checkpoints/training_history.json", "r") as f:
            hist = json.load(f)
            vals = hist.get("val", [])
            if vals:
                # We show the accuracy of the best mIoU run
                best_val = max(vals, key=lambda x: x.get("mIoU", 0.0))
                best_acc = best_val.get("accuracy", 0.0)
                acc_str = f"| **Validation Accuracy:** `{best_acc:.1%}`"
    except Exception:
        pass

    triage_text = (
        f"## Decision: {decision_emoji}  {acc_str}\n\n"
        f"| Metric | Value |\n"
        f"|--------|-------|\n"
        f"| Flood Detected | {'✅ Yes' if triage_result.flood_detected else '❌ No'} |\n"
        f"| Flooded Area | {triage_result.flooded_area_km2:.3f} km2 |\n"
        f"| Flood Fraction | {triage_result.flood_fraction:.1%} |\n"
        f"| Confidence | {triage_result.confidence:.1%} |\n"
        f"| Urgency | {urgency_emoji} {triage_result.urgency} |\n"
        f"| Inference Latency | {triage_result.inference_latency_ms:.0f} ms |\n"
        f"| Summary Size | ~{len(json_str)} bytes |\n"
    )

    # --- Mask image ---
    mask_fig, mask_ax = plt.subplots(figsize=(4, 4))
    mask_ax.imshow(binary_mask, cmap="Blues", vmin=0, vmax=1)
    mask_ax.set_title("Flood Mask")
    mask_ax.axis("off")
    plt.tight_layout()

    return fig, json_str, triage_text, mask_fig


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_app():
    """Build the Gradio application."""

    with gr.Blocks(
        title="FloodBrief - Orbital Flood Intelligence",
    ) as app:
        gr.Markdown("""
        # 🛰️ FloodBrief - Orbital-Compute Flood Intelligence

        **Downlink the answer, not the data.**

        Upload a Sentinel-1 SAR tile or generate a synthetic demo tile.
        FloodBrief runs TerraMind-based flood segmentation and produces a compact
        triage summary - area, confidence, urgency, and a downlink/skip decision
        - in under 2 seconds.

        *Built for the AI/ML in Space Track * TakeMe2Space x IBM TerraMind*
        """)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📡 Input")

                upload_file = gr.File(
                    label="Upload Sentinel-1 SAR GeoTIFF",
                    file_types=[".tif", ".tiff", ".png", ".jpg"],
                    visible=True,
                )

                threshold = gr.Slider(
                    minimum=0.1, maximum=0.9, value=0.5, step=0.05,
                    label="Classification threshold",
                    info="Higher = more conservative (fewer false positives)"
                )

                tile_id = gr.Textbox(
                    label="Tile ID (optional)",
                    value="",
                    placeholder="e.g., India_103757"
                )

                run_btn = gr.Button("🚀 Run FloodBrief", variant="primary", size="lg")

            with gr.Column(scale=2):
                gr.Markdown("### 📊 Results")

                triage_output = gr.Markdown(label="Triage Summary")

                with gr.Tab("Analysis"):
                    analysis_plot = gr.Plot(label="FloodBrief Analysis")

                with gr.Tab("Flood Mask"):
                    mask_plot = gr.Plot(label="Flood Mask")

                with gr.Tab("JSON Summary"):
                    json_output = gr.Code(
                        label="Downlink JSON (~500 bytes)",
                        language="json",
                    )

        # Wire up the button
        run_btn.click(
            fn=run_inference,
            inputs=[upload_file, threshold, tile_id],
            outputs=[analysis_plot, json_output, triage_output, mask_plot],
        )

        gr.Markdown("""
        ---
        ### How it works

        1. **Input:** Sentinel-1 SAR tile (VV + VH channels, 224x224 pixels)
        2. **Encoder:** TerraMind-1.0-small extracts 196 patch embeddings
        3. **Head:** UPerNet decoder produces per-pixel flood probabilities
        4. **Triage:** Area (km2), confidence, urgency, downlink decision
        5. **Output:** ~500-byte JSON summary (vs ~50 MB raw tile)

        **Bandwidth saving per skipped tile: ~99.999%**

        | Component | Detail |
        |-----------|--------|
        | Model | TerraMind-1.0-small (~100M params, ~200 MB FP16) |
        | Dataset | Sen1Floods11 (4,831 chips, 11 flood events) |
        | Input | Sentinel-1 SAR (VV, VH), 224x224 |
        | Target hardware | Nvidia Jetson Orin Nano (8 GB, 40 TOPS INT8) |
        | Latency | ~2 sec/tile on Jetson (est.) |
        """)

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="cyan"),
    )
