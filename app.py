"""
FloodBrief - Gradio demo app.

Usage:
    python app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import gradio as gr
except ImportError:
    print("ERROR: gradio is not installed. Install with: pip install -r requirements-app.txt")
    sys.exit(1)

from src.inference_utils import generate_synthetic_tile, load_tile
from src.metrics import InferenceTimer
from src.model import load_model
from src.project_stats import (
    load_project_summary,
    render_benchmark_markdown,
    render_comparison_markdown,
)
from src.triage import triage_tile
from src.visualization import plot_flood_overlay, plot_metrics_comparison_chart


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = None
PROJECT_SUMMARY = load_project_summary()


def get_model():
    """Lazy-load the model so the UI starts quickly."""
    global MODEL
    if MODEL is None:
        checkpoint = None
        preferred_checkpoint = Path("./checkpoints/final_model.pt")
        fallback_checkpoint = Path("./checkpoints/best_model.pt")
        if preferred_checkpoint.exists():
            checkpoint = str(preferred_checkpoint)
        elif fallback_checkpoint.exists():
            checkpoint = str(fallback_checkpoint)

        MODEL = load_model(checkpoint_path=checkpoint, device=DEVICE)
    return MODEL


def build_triage_markdown(triage_result, json_size_bytes: int) -> str:
    """Render the per-tile result table."""
    return (
        "## Tile Result\n\n"
        "| Metric | Value |\n"
        "|---|---|\n"
        f"| Decision | `{triage_result.downlink_decision.upper()}` |\n"
        f"| Flood detected | {'Yes' if triage_result.flood_detected else 'No'} |\n"
        f"| Flooded area | {triage_result.flooded_area_km2:.3f} km2 |\n"
        f"| Flood fraction | {triage_result.flood_fraction:.1%} |\n"
        f"| Confidence | {triage_result.confidence:.1%} |\n"
        f"| Urgency | {triage_result.urgency} |\n"
        f"| Inference latency | {triage_result.inference_latency_ms:.0f} ms |\n"
        f"| Downlink summary size | {json_size_bytes} bytes |\n"
        f"| Inference backend | `{getattr(MODEL, 'inference_backend', 'unknown')}` |\n"
    )


def run_inference(
    upload_file,
    use_synthetic: bool,
    threshold: float,
    tile_id: str,
):
    """
    Run FloodBrief inference on an uploaded tile or built-in demo tile.

    Returns:
        analysis figure, JSON summary, markdown summary, mask figure, comparison figure
    """
    plt.close("all")

    model = get_model()
    img_size = 224

    try:
        if upload_file is None and not use_synthetic:
            error_message = "Upload a tile or enable the built-in synthetic demo tile."
            return None, "{}", f"## Error\n{error_message}", None, None

        if upload_file is None:
            tile = generate_synthetic_tile(img_size=img_size, has_flood=True, seed=None)
            resolved_tile_id = tile_id or "synthetic_demo"
        else:
            tile = load_tile(upload_file, img_size=img_size)
            resolved_tile_id = tile_id or Path(upload_file).stem
    except Exception as exc:
        return None, "{}", f"## Error\n{exc}", None, None

    input_tensor = torch.from_numpy(tile).unsqueeze(0).float().to(DEVICE)

    with InferenceTimer() as timer:
        output = model.predict(input_tensor)

    flood_prob = output["flood_probability"][0].cpu().numpy()
    binary_mask = output["binary_mask"][0].cpu().numpy()

    triage_result = triage_tile(
        flood_probability=flood_prob,
        tile_id=resolved_tile_id,
        threshold=threshold,
        inference_latency_ms=timer.elapsed_ms,
    )

    analysis_figure = plot_flood_overlay(
        sar_image=tile,
        flood_mask=binary_mask,
        flood_probability=flood_prob,
        triage_result=triage_result,
        title=f"FloodBrief Analysis - {resolved_tile_id}",
    )

    json_summary = triage_result.to_json(indent=2)
    triage_markdown = build_triage_markdown(triage_result, len(json_summary.encode("utf-8")))

    mask_figure, mask_axis = plt.subplots(figsize=(4, 4))
    mask_axis.imshow(binary_mask, cmap="Blues", vmin=0, vmax=1)
    mask_axis.set_title("Predicted Flood Mask")
    mask_axis.axis("off")
    plt.tight_layout()

    comparison_figure = plot_metrics_comparison_chart(
        PROJECT_SUMMARY.get("model_metrics", {}),
        PROJECT_SUMMARY.get("baseline_metrics", {}),
    )

    return (
        analysis_figure,
        json_summary,
        triage_markdown,
        mask_figure,
        comparison_figure,
    )


def build_app():
    """Build the Gradio application."""

    with gr.Blocks(title="FloodBrief - Orbital Flood Intelligence") as app:
        gr.Markdown(
            """
            # FloodBrief

            FloodBrief turns a Sentinel-1 tile into a compact flood triage report:
            flood mask, flooded area, confidence, urgency, and a downlink-or-skip decision.
            """
        )
        gr.Markdown(render_benchmark_markdown(PROJECT_SUMMARY))

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Input")

                upload_file = gr.File(
                    label="Upload Sentinel-1 tile",
                    file_types=[".tif", ".tiff", ".png", ".jpg", ".jpeg"],
                )
                use_synthetic = gr.Checkbox(
                    label="Use built-in synthetic demo tile when no file is uploaded",
                    value=True,
                )
                threshold = gr.Slider(
                    minimum=0.1,
                    maximum=0.9,
                    value=0.5,
                    step=0.05,
                    label="Classification threshold",
                    info="Higher values are more conservative.",
                )
                tile_id = gr.Textbox(
                    label="Tile ID (optional)",
                    placeholder="e.g. India_103757",
                )
                run_button = gr.Button("Run FloodBrief", variant="primary", size="lg")

            with gr.Column(scale=2):
                gr.Markdown("### Results")
                triage_output = gr.Markdown(label="Tile summary")

                with gr.Tab("Analysis"):
                    analysis_plot = gr.Plot(label="FloodBrief analysis")

                with gr.Tab("Flood Mask"):
                    mask_plot = gr.Plot(label="Flood mask")

                with gr.Tab("Model vs Baseline"):
                    gr.Markdown(render_comparison_markdown(PROJECT_SUMMARY))
                    comparison_plot = gr.Plot(
                        value=plot_metrics_comparison_chart(
                            PROJECT_SUMMARY.get("model_metrics", {}),
                            PROJECT_SUMMARY.get("baseline_metrics", {}),
                        ),
                        label="Benchmark comparison chart",
                    )

                with gr.Tab("JSON Summary"):
                    json_output = gr.Code(label="Downlink JSON", language="json")

        gr.Markdown(
            """
            ### Quick Notes

            - `requirements-app.txt` is the fastest path for the demo UI.
            - `requirements.txt` includes the full training and evaluation stack.
            - If TerraMind weights or checkpoints are unavailable, the app still starts and falls back to the lightweight demo encoder.
            """
        )

        run_button.click(
            fn=run_inference,
            inputs=[upload_file, use_synthetic, threshold, tile_id],
            outputs=[analysis_plot, json_output, triage_output, mask_plot, comparison_plot],
        )

    return app


if __name__ == "__main__":
    application = build_app()
    application.launch(
        server_name="localhost",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="cyan"),
    )
