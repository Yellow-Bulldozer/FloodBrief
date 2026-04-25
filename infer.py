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
import json
import argparse
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.model import load_model
from src.triage import TriageEngine, triage_tile
from src.metrics import InferenceTimer
from src.visualization import plot_flood_overlay


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


def load_tile(path: str, img_size: int = 224) -> np.ndarray:
    """Load a Sentinel-1 SAR GeoTIFF tile."""
    try:
        import rasterio
        with rasterio.open(path) as src:
            data = src.read().astype(np.float32)

            # Ensure 2 channels (VV, VH)
            if data.shape[0] > 2:
                data = data[:2]
            elif data.shape[0] < 2:
                data = np.repeat(data, 2, axis=0)[:2]

            # Crop / resize to img_size
            _, H, W = data.shape
            min_dim = min(H, W)
            top = (H - min_dim) // 2
            left = (W - min_dim) // 2
            data = data[:, top:top + min_dim, left:left + min_dim]

            if min_dim != img_size:
                t = torch.from_numpy(data).unsqueeze(0)
                t = torch.nn.functional.interpolate(t, size=(img_size, img_size), mode="bilinear", align_corners=False)
                data = t.squeeze(0).numpy()

            # Normalize SAR dB using official function
            from src.data_loader import Sen1Floods11Dataset
            data = Sen1Floods11Dataset._normalize_s1(data, method="terramind")
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

            return data

    except ImportError:
        print("[FloodBrief] rasterio not available. Trying PIL fallback...")
        from PIL import Image
        img = np.array(Image.open(path), dtype=np.float32)
        if img.ndim == 2:
            img = np.stack([img, img], axis=0)
        elif img.ndim == 3:
            img = img.transpose(2, 0, 1)[:2]
        # Resize
        t = torch.from_numpy(img).unsqueeze(0)
        t = torch.nn.functional.interpolate(t, size=(img_size, img_size), mode="bilinear", align_corners=False)
        return t.squeeze(0).numpy()


def generate_synthetic_tile(img_size: int = 224, has_flood: bool = True) -> np.ndarray:
    """Generate a synthetic SAR tile with a flood region for demo."""
    np.random.seed(int(time.time()) % 1000)

    # Base SAR texture
    img = np.random.uniform(0.3, 0.7, (2, img_size, img_size)).astype(np.float32)

    # Add some spatial structure (blobs)
    from scipy.ndimage import gaussian_filter
    for c in range(2):
        img[c] = gaussian_filter(img[c], sigma=5)
        # Re-normalize
        img[c] = (img[c] - img[c].min()) / (img[c].max() - img[c].min() + 1e-8)
        img[c] = img[c] * 0.5 + 0.2

    if has_flood:
        # Create a irregular flood region
        cx = np.random.randint(60, img_size - 60)
        cy = np.random.randint(60, img_size - 60)
        rx = np.random.randint(30, 70)
        ry = np.random.randint(30, 70)

        yy, xx = np.ogrid[:img_size, :img_size]
        mask = ((xx - cx) ** 2 / rx ** 2 + (yy - cy) ** 2 / ry ** 2) <= 1.0

        # Water appears dark in SAR (low backscatter)
        img[0][mask] *= 0.15
        img[1][mask] *= 0.15

        # Add noise to edges
        edge = gaussian_filter(mask.astype(np.float32), sigma=3) > 0.3
        noise_mask = edge & ~mask
        img[0][noise_mask] *= 0.5
        img[1][noise_mask] *= 0.5

    return img


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
