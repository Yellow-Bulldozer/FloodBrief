"""
FloodBrief - Visualization utilities.

Produces flood mask overlays, comparison plots, and demo visuals.
"""

import os
from typing import Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch


# ---------------------------------------------------------------------------
# Color maps
# ---------------------------------------------------------------------------

# Flood mask colormap: transparent for no-flood, blue for flood
FLOOD_CMAP = mcolors.ListedColormap(["none", "#1E90FF"])

# Urgency colors
URGENCY_COLORS = {
    "CRITICAL": "#FF0000",
    "HIGH": "#FF6600",
    "MODERATE": "#FFCC00",
    "LOW": "#00CC66",
    "NONE": "#888888",
}


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------

def plot_flood_overlay(
    sar_image: np.ndarray,
    flood_mask: np.ndarray,
    flood_probability: Optional[np.ndarray] = None,
    triage_result=None,
    title: str = "FloodBrief Analysis",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 5),
) -> plt.Figure:
    """
    Create a multi-panel visualization:
    1. SAR input (VV channel)
    2. Flood mask overlay
    3. Flood probability heatmap
    4. Triage summary card

    Args:
        sar_image: (2, H, W) or (H, W) SAR data.
        flood_mask: (H, W) binary flood mask.
        flood_probability: (H, W) flood probabilities.
        triage_result: TriageResult object (optional).
        title: Figure title.
        save_path: Path to save the figure.
        figsize: Figure size.

    Returns:
        matplotlib Figure.
    """
    n_panels = 3 if flood_probability is not None else 2
    if triage_result is not None:
        n_panels += 1

    fig, axes = plt.subplots(1, n_panels, figsize=figsize)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    # Prepare SAR display (VV channel)
    if sar_image.ndim == 3:
        sar_display = sar_image[0]  # VV channel
    else:
        sar_display = sar_image

    # Panel 1: SAR input
    ax = axes[0]
    ax.imshow(sar_display, cmap="gray", vmin=0, vmax=1)
    ax.set_title("Sentinel-1 SAR (VV)", fontsize=11)
    ax.axis("off")

    # Panel 2: Flood mask overlay
    ax = axes[1]
    ax.imshow(sar_display, cmap="gray", vmin=0, vmax=1)
    mask_display = np.ma.masked_where(flood_mask == 0, flood_mask)
    ax.imshow(mask_display, cmap=FLOOD_CMAP, alpha=0.6, vmin=0, vmax=1)
    ax.set_title("Flood Detection Overlay", fontsize=11)
    ax.axis("off")
    legend_elements = [
        Patch(facecolor="#1E90FF", alpha=0.6, label="Flood detected"),
        Patch(facecolor="gray", alpha=0.5, label="No flood"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8, framealpha=0.8)

    # Panel 3: Probability heatmap
    panel_idx = 2
    if flood_probability is not None:
        ax = axes[panel_idx]
        im = ax.imshow(flood_probability, cmap="RdYlBu_r", vmin=0, vmax=1)
        ax.set_title("Flood Probability", fontsize=11)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="P(flood)")
        panel_idx += 1

    # Panel 4: Triage summary card
    if triage_result is not None:
        ax = axes[panel_idx]
        ax.axis("off")

        # Urgency color
        urgency = getattr(triage_result, "urgency", "NONE")
        urgency_color = URGENCY_COLORS.get(urgency, "#888888")
        decision = getattr(triage_result, "downlink_decision", "unknown")

        # Build text card
        card_text = (
            f"{'-' * 30}\n"
            f"  FLOODBRIEF TRIAGE REPORT\n"
            f"{'-' * 30}\n\n"
            f"  Tile ID:  {getattr(triage_result, 'tile_id', 'N/A')}\n\n"
            f"  Flood Detected:  {'YES' if triage_result.flood_detected else 'NO'}\n"
            f"  Flooded Area:    {triage_result.flooded_area_km2:.3f} km2\n"
            f"  Flood Fraction:  {triage_result.flood_fraction:.1%}\n"
            f"  Confidence:      {triage_result.confidence:.1%}\n\n"
            f"  Urgency:         {urgency}\n"
            f"  Decision:        {decision.upper()}\n\n"
            f"  Latency:         {triage_result.inference_latency_ms:.0f} ms\n"
            f"{'-' * 30}"
        )

        # Background color based on decision
        bg_color = "#E8F5E9" if decision == "skip" else "#FFEBEE"
        ax.add_patch(plt.Rectangle(
            (0.02, 0.02), 0.96, 0.96, transform=ax.transAxes,
            facecolor=bg_color, edgecolor=urgency_color, linewidth=3,
            clip_on=False, zorder=0,
        ))

        ax.text(
            0.5, 0.5, card_text,
            transform=ax.transAxes,
            fontsize=9,
            fontfamily="monospace",
            verticalalignment="center",
            horizontalalignment="center",
        )
        ax.set_title(f"Triage: {decision.upper()}", fontsize=11,
                     color=urgency_color, fontweight="bold")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"[FloodBrief] Visualization saved: {save_path}")

    return fig


def plot_comparison(
    sar_image: np.ndarray,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    title: str = "FloodBrief: Prediction vs Ground Truth",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Side-by-side comparison of prediction vs ground truth."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # SAR display
    if sar_image.ndim == 3:
        sar_display = sar_image[0]
    else:
        sar_display = sar_image

    # Panel 1: SAR input
    axes[0].imshow(sar_display, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("SAR Input (VV)")
    axes[0].axis("off")

    # Panel 2: Ground truth
    axes[1].imshow(ground_truth, cmap="RdYlBu_r", vmin=0, vmax=1)
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    # Panel 3: Prediction
    axes[2].imshow(prediction, cmap="RdYlBu_r", vmin=0, vmax=1)
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")

    return fig


def plot_metrics_table(
    model_metrics: dict,
    baseline_metrics: dict,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot a comparison table of model vs baseline metrics."""

    rows = [
        ("mIoU", model_metrics.get("mIoU", 0), baseline_metrics.get("mIoU", 0)),
        ("Flood IoU", model_metrics.get("flood_IoU", 0), baseline_metrics.get("flood_IoU", 0)),
        ("F1 (flood)", model_metrics.get("f1_flood", 0), baseline_metrics.get("f1_flood", 0)),
        ("Precision", model_metrics.get("precision_flood", 0), baseline_metrics.get("precision_flood", 0)),
        ("Recall", model_metrics.get("recall_flood", 0), baseline_metrics.get("recall_flood", 0)),
        ("Accuracy", model_metrics.get("accuracy", 0), baseline_metrics.get("accuracy", 0)),
    ]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")

    col_labels = ["Metric", "FloodBrief", "Baseline (Majority)", "Delta"]
    table_data = []
    for name, model_val, base_val in rows:
        delta = model_val - base_val
        table_data.append([
            name,
            f"{model_val:.4f}",
            f"{base_val:.4f}",
            f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}",
        ])

    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Style header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Style delta column
    for i, row in enumerate(table_data, start=1):
        val = float(row[3].replace("+", ""))
        table[i, 3].set_text_props(
            color="green" if val > 0 else ("red" if val < 0 else "gray"),
            fontweight="bold",
        )

    ax.set_title("FloodBrief vs Baseline", fontsize=13, fontweight="bold", pad=20)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")

    return fig


def plot_metrics_comparison_chart(
    model_metrics: dict,
    baseline_metrics: dict,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot a grouped bar chart comparing FloodBrief against the baseline."""

    labels = [
        "mIoU",
        "Flood IoU",
        "F1",
        "Precision",
        "Recall",
        "Accuracy",
    ]
    model_values = np.array([
        model_metrics.get("mIoU", 0.0),
        model_metrics.get("flood_IoU", 0.0),
        model_metrics.get("f1_flood", 0.0),
        model_metrics.get("precision_flood", 0.0),
        model_metrics.get("recall_flood", 0.0),
        model_metrics.get("accuracy", 0.0),
    ])
    baseline_values = np.array([
        baseline_metrics.get("mIoU", 0.0),
        baseline_metrics.get("flood_IoU", 0.0),
        baseline_metrics.get("f1_flood", 0.0),
        baseline_metrics.get("precision_flood", 0.0),
        baseline_metrics.get("recall_flood", 0.0),
        baseline_metrics.get("accuracy", 0.0),
    ])

    x = np.arange(len(labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(10, 4.8))
    floodbrief_bars = ax.bar(
        x - width / 2,
        model_values,
        width,
        label="FloodBrief",
        color="#0EA5E9",
    )
    baseline_bars = ax.bar(
        x + width / 2,
        baseline_values,
        width,
        label="Majority baseline",
        color="#94A3B8",
    )

    ax.set_title("FloodBrief vs Majority Baseline", fontsize=13, fontweight="bold")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend(frameon=False)

    for bars in (floodbrief_bars, baseline_bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.02,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")

    return fig
