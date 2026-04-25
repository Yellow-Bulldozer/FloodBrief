"""
FloodBrief - Evaluation script.

Evaluates a trained FloodBrief model on the Sen1Floods11 test set.
Computes segmentation metrics, baseline comparisons, product metrics,
and inference latency.

Usage:
    python evaluate.py --data-dir ./data/sen1floods11 --checkpoint ./checkpoints/best_model.pt

With synthetic data:
    python evaluate.py --synthetic --checkpoint ./checkpoints/best_model.pt
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.model import FloodBriefModel, load_model
from src.data_loader import get_dataloaders
from src.triage import TriageEngine
from src.metrics import (
    SegmentationMetrics,
    majority_class_baseline,
    random_baseline,
    compute_product_metrics,
    InferenceTimer,
)
from src.visualization import plot_comparison, plot_metrics_table
from src.visualization import plot_metrics_comparison_chart


def parse_args():
    parser = argparse.ArgumentParser(description="FloodBrief evaluation")
    parser.add_argument("--data-dir", type=str, default="./data/sen1floods11")
    parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint path")
    parser.add_argument("--output-dir", type=str, default="./eval_results")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save-visuals", action="store_true", default=True)
    parser.add_argument("--num-visuals", type=int, default=10, help="Number of visual samples to save")
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


@torch.no_grad()
def evaluate_model(model, test_loader, device, triage_engine):
    """Run full evaluation on the test set."""
    model.eval()
    metrics = SegmentationMetrics(num_classes=2)

    all_predictions = []
    all_labels = []
    all_images = []
    all_triage_results = []
    all_gt_has_flood = []
    all_chip_ids = []

    print("\n[Evaluation] Running inference on test set...")

    for batch in tqdm(test_loader, desc="Evaluating"):
        images = batch["image"].to(device)
        labels = batch["label"]
        chip_ids = batch["chip_id"]

        # Timed inference
        with InferenceTimer() as timer:
            output = model(images)
            logits = output["logits"]
            probs = output["probabilities"]

        # Predictions
        pred_classes = logits.argmax(dim=1).cpu().numpy()
        flood_probs = probs[:, 1].cpu().numpy()
        labels_np = labels.numpy()

        # Update segmentation metrics
        metrics.update(pred_classes, labels_np, latency_ms=timer.elapsed_ms)

        # Store for further analysis
        for i in range(images.size(0)):
            all_predictions.append(pred_classes[i])
            all_labels.append(labels_np[i])
            all_images.append(images[i].cpu().numpy())
            all_chip_ids.append(chip_ids[i])

            # Triage each tile
            triage_result = triage_engine.process(
                flood_probability=flood_probs[i],
                tile_id=chip_ids[i],
                inference_latency_ms=timer.elapsed_ms / images.size(0),
            )
            all_triage_results.append(triage_result)

            # Ground truth: does this tile actually have flood?
            gt_has_flood = bool(np.any(labels_np[i] == 1))
            all_gt_has_flood.append(gt_has_flood)

    return {
        "metrics": metrics,
        "predictions": all_predictions,
        "labels": all_labels,
        "images": all_images,
        "triage_results": all_triage_results,
        "gt_has_flood": all_gt_has_flood,
        "chip_ids": all_chip_ids,
    }


def main():
    args = parse_args()
    device = get_device(args.device)
    print(f"\n{'='*60}")
    print(f"  FloodBrief Evaluation")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    os.makedirs(args.output_dir, exist_ok=True)

    # Data
    _, _, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers,
        use_synthetic=args.synthetic,
        max_samples=args.max_samples,
    )

    # Model
    model = load_model(
        checkpoint_path=args.checkpoint,
        device=str(device),
        img_size=args.img_size,
    )

    # Triage engine
    triage_engine = TriageEngine()

    # Run evaluation
    eval_data = evaluate_model(model, test_loader, device, triage_engine)

    # --- Segmentation metrics ---
    seg_metrics = eval_data["metrics"].compute()
    print("\n" + "=" * 50)
    print("  SEGMENTATION METRICS (FloodBrief)")
    print("=" * 50)
    for k, v in seg_metrics.items():
        if k != "confusion_matrix":
            print(f"  {k:25s}: {v}")

    # --- Baseline: majority class ---
    baseline_mc = majority_class_baseline(eval_data["labels"])
    print("\n" + "=" * 50)
    print("  BASELINE: Majority Class (all 'no flood')")
    print("=" * 50)
    for k, v in baseline_mc.items():
        if k not in ("confusion_matrix", "baseline_name", "baseline_description"):
            print(f"  {k:25s}: {v}")

    # --- Baseline: random ---
    baseline_rand = random_baseline(eval_data["labels"])
    print("\n" + "=" * 50)
    print("  BASELINE: Random")
    print("=" * 50)
    for k, v in baseline_rand.items():
        if k not in ("confusion_matrix", "baseline_name", "baseline_description"):
            print(f"  {k:25s}: {v}")

    # --- Product metrics ---
    product_metrics = compute_product_metrics(
        eval_data["triage_results"],
        eval_data["gt_has_flood"],
    )
    print("\n" + "=" * 50)
    print("  PRODUCT METRICS (Triage)")
    print("=" * 50)
    for k, v in product_metrics.items():
        print(f"  {k:25s}: {v}")

    # --- Save results ---
    results = {
        "model_metrics": seg_metrics,
        "baseline_majority_class": baseline_mc,
        "baseline_random": baseline_rand,
        "product_metrics": product_metrics,
    }

    results_path = os.path.join(args.output_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[FloodBrief] Results saved: {results_path}")

    # --- Save visualization samples ---
    if args.save_visuals:
        vis_dir = os.path.join(args.output_dir, "visuals")
        os.makedirs(vis_dir, exist_ok=True)

        num_vis = min(args.num_visuals, len(eval_data["predictions"]))
        print(f"\n[FloodBrief] Saving {num_vis} visual samples...")

        for i in range(num_vis):
            save_path = os.path.join(vis_dir, f"sample_{i:03d}.png")
            plot_comparison(
                sar_image=eval_data["images"][i],
                ground_truth=eval_data["labels"][i],
                prediction=eval_data["predictions"][i],
                title=f"Tile: {eval_data['chip_ids'][i]}",
                save_path=save_path,
            )

        # Metrics comparison table
        plot_metrics_table(
            model_metrics=seg_metrics,
            baseline_metrics=baseline_mc,
            save_path=os.path.join(vis_dir, "metrics_comparison.png"),
        )
        plot_metrics_comparison_chart(
            model_metrics=seg_metrics,
            baseline_metrics=baseline_mc,
            save_path=os.path.join(vis_dir, "metrics_comparison_chart.png"),
        )

    print(f"\n{'='*60}")
    print(f"  Evaluation complete!")
    print(f"  Model mIoU:    {seg_metrics['mIoU']:.4f}")
    print(f"  Baseline mIoU: {baseline_mc['mIoU']:.4f}")
    print(f"  Delta mIoU:    +{seg_metrics['mIoU'] - baseline_mc['mIoU']:.4f}")
    print(f"  Bandwidth saved: {product_metrics.get('bandwidth_saving_pct', 0)}%")
    print(f"  Events retained: {product_metrics.get('event_retention_pct', 0)}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
