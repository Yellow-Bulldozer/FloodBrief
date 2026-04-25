"""
FloodBrief - Single-tile inference entry point.

Loads one EO tile, runs TerraMind-based flood inference, produces
flood mask, triage summary, and visual output.

Usage:
    python infer.py --input ./sample_input/example_s1_tile.tif --output-dir ./sample_output

With synthetic data (no real tile needed):
    python infer.py --synthetic --output-dir ./sample_output

With a trained checkpoint:
    python infer.py --input tile.tif --checkpoint ./checkpoints/best_model.pt
"""

import os
import sys
import argparse
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.model import load_model
from src.triage import triage_tile
from src.metrics import InferenceTimer
from src.visualization import plot_flood_overlay
from src.inference_utils import generate_synthetic_tile, load_tile


def parse_args():
    parser = argparse.ArgumentParser(description="FloodBrief single-tile inference")
    parser.add_argument("--input", type=str, default=None, help="Path to Sentinel-1 SAR GeoTIFF tile")
    parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint path")
    parser.add_argument("--output-dir", type=str, default="./sample_output", help="Output directory")
    parser.add_argument("--img-size", type=int, default=224, help="Input image size")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")
    parser.add_argument("--device", type=str, default="auto", help="Device")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic tile for demo")
    parser.add_argument("--tile-id", type=str, default=None, help="Override tile ID")
    return parser.parse_args()


def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_str)


def main():
    args = parse_args()
    device = get_device(args.device)

    print(f"\n{'='*60}")
    print(f"  FloodBrief - Single-Tile Inference")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load input tile ---
    if args.synthetic or args.input is None:
        print("[FloodBrief] Using synthetic SAR tile for demo.")
        tile = generate_synthetic_tile(args.img_size, has_flood=True)
        tile_id = args.tile_id or "synthetic_demo"
    else:
        print(f"[FloodBrief] Loading tile: {args.input}")
        tile = load_tile(args.input, args.img_size)
        tile_id = args.tile_id or Path(args.input).stem

    print(f"[FloodBrief] Tile shape: {tile.shape}, ID: {tile_id}")

    # --- Load model ---
    model = load_model(
        checkpoint_path=args.checkpoint,
        device=str(device),
        img_size=args.img_size,
    )

    # --- Run inference ---
    print("[FloodBrief] Running flood inference...")

    input_tensor = torch.from_numpy(tile).unsqueeze(0).float().to(device)

    with InferenceTimer() as timer:
        output = model.predict(input_tensor)

    flood_prob = output["flood_probability"][0].cpu().numpy()
    binary_mask = output["binary_mask"][0].cpu().numpy()

    print(f"[FloodBrief] Inference complete in {timer.elapsed_ms:.1f} ms")

    # --- Triage ---
    triage_result = triage_tile(
        flood_probability=flood_prob,
        tile_id=tile_id,
        threshold=args.threshold,
        inference_latency_ms=timer.elapsed_ms,
    )

    # Print triage summary
    print(f"\n{'-'*40}")
    print(f"  FLOODBRIEF TRIAGE RESULT")
    print(f"{'-'*40}")
    print(f"  Tile ID:        {triage_result.tile_id}")
    print(f"  Flood Detected: {'YES' if triage_result.flood_detected else 'NO'}")
    print(f"  Flooded Area:   {triage_result.flooded_area_km2:.3f} km2")
    print(f"  Flood Fraction: {triage_result.flood_fraction:.1%}")
    print(f"  Confidence:     {triage_result.confidence:.1%}")
    print(f"  Urgency:        {triage_result.urgency}")
    print(f"  Decision:       {triage_result.downlink_decision.upper()}")
    print(f"  Latency:        {triage_result.inference_latency_ms:.0f} ms")
    print(f"{'-'*40}\n")

    # --- Save JSON summary ---
    json_path = os.path.join(args.output_dir, f"{tile_id}_triage.json")
    json_str = triage_result.to_json(indent=2)
    with open(json_path, "w") as f:
        f.write(json_str)
    print(f"[FloodBrief] JSON summary saved: {json_path} ({len(json_str)} bytes)")

    # --- Save visualization ---
    vis_path = os.path.join(args.output_dir, f"{tile_id}_analysis.png")
    plot_flood_overlay(
        sar_image=tile,
        flood_mask=binary_mask,
        flood_probability=flood_prob,
        triage_result=triage_result,
        title=f"FloodBrief Analysis - {tile_id}",
        save_path=vis_path,
    )

    # --- Save binary mask ---
    import matplotlib.pyplot as plt
    mask_path = os.path.join(args.output_dir, f"{tile_id}_flood_mask.png")
    plt.imsave(mask_path, binary_mask, cmap="Blues")
    print(f"[FloodBrief] Flood mask saved: {mask_path}")

    print(f"\n[FloodBrief] All outputs saved to {args.output_dir}")
    print(f"[FloodBrief] Done.")


if __name__ == "__main__":
    main()
