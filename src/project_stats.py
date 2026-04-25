"""
Shared benchmark metadata for the README and Gradio app.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY_PATH = ROOT_DIR / "docs" / "benchmark_summary.json"
LIVE_EVAL_PATH = ROOT_DIR / "eval_results" / "evaluation_results.json"

METRIC_FIELDS: List[Tuple[str, str]] = [
    ("mIoU", "mIoU"),
    ("Flood IoU", "flood_IoU"),
    ("F1 (flood)", "f1_flood"),
    ("Precision", "precision_flood"),
    ("Recall", "recall_flood"),
    ("Accuracy", "accuracy"),
]


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _artifact_size_mb(path: Path) -> float:
    return round(path.stat().st_size / 1_000_000, 2)


def load_project_summary() -> Dict[str, Any]:
    """Load tracked benchmark data and refresh it from local artifacts when possible."""
    summary = deepcopy(_read_json(DEFAULT_SUMMARY_PATH))
    if not summary:
        summary = {
            "artifacts": {},
            "model_metrics": {},
            "baseline_metrics": {},
            "product_metrics": {},
            "recorded_eval_speed": {},
            "deployment_estimate": {},
            "notes": [],
        }

    for artifact in summary.get("artifacts", {}).values():
        artifact_path = ROOT_DIR / artifact["path"]
        if artifact_path.exists():
            artifact["size_bytes"] = artifact_path.stat().st_size
            artifact["size_mb"] = _artifact_size_mb(artifact_path)

    live_results = _read_json(LIVE_EVAL_PATH)
    if live_results:
        summary["model_metrics"] = live_results.get(
            "model_metrics",
            summary.get("model_metrics", {}),
        )
        summary["baseline_metrics"] = live_results.get(
            "baseline_majority_class",
            summary.get("baseline_metrics", {}),
        )
        summary["product_metrics"] = live_results.get(
            "product_metrics",
            summary.get("product_metrics", {}),
        )

    latency_ms = summary.get("model_metrics", {}).get(
        "avg_latency_ms",
        summary.get("recorded_eval_speed", {}).get("latency_ms_per_tile", 0.0),
    )
    if latency_ms:
        summary.setdefault("recorded_eval_speed", {})
        summary["recorded_eval_speed"]["latency_ms_per_tile"] = round(latency_ms, 2)
        summary["recorded_eval_speed"]["tiles_per_second"] = round(1000.0 / latency_ms, 2)

    return summary


def metric_rows(summary: Dict[str, Any]) -> List[Tuple[str, float, float, float]]:
    model_metrics = summary.get("model_metrics", {})
    baseline_metrics = summary.get("baseline_metrics", {})
    rows = []
    for label, key in METRIC_FIELDS:
        model_value = float(model_metrics.get(key, 0.0))
        baseline_value = float(baseline_metrics.get(key, 0.0))
        rows.append((label, model_value, baseline_value, model_value - baseline_value))
    return rows


def render_benchmark_markdown(summary: Dict[str, Any]) -> str:
    artifacts = summary.get("artifacts", {})
    deployment = artifacts.get("deployment_checkpoint", {})
    training = artifacts.get("training_checkpoint", {})
    recorded_speed = summary.get("recorded_eval_speed", {})
    orbit_speed = summary.get("deployment_estimate", {})
    product_metrics = summary.get("product_metrics", {})

    return (
        "### Repo Benchmarks\n\n"
        "| Item | Value |\n"
        "|---|---|\n"
        f"| Deployment checkpoint | {deployment.get('size_mb', 0.0):.2f} MB on disk (`{deployment.get('path', 'n/a')}`) |\n"
        f"| Recorded inference speed | {recorded_speed.get('latency_ms_per_tile', 0.0):.2f} ms/tile (~{recorded_speed.get('tiles_per_second', 0.0):.2f} tiles/s) |\n"
        f"| Estimated orbital speed | {orbit_speed.get('latency_ms_per_tile', 0.0) / 1000.0:.2f} s/tile (~{orbit_speed.get('tiles_per_second', 0.0):.2f} tiles/s) on {orbit_speed.get('hardware', 'target edge hardware')} |\n"
        f"| Bandwidth saved in recorded eval | {product_metrics.get('bandwidth_saving_pct', 0.0):.1f}% |\n"
        f"| Flood-event retention | {product_metrics.get('event_retention_pct', 0.0):.1f}% |\n"
    )


def render_comparison_markdown(summary: Dict[str, Any]) -> str:
    lines = [
        "### Model vs Baseline",
        "",
        "| Metric | FloodBrief | Majority baseline | Delta |",
        "|---|---:|---:|---:|",
    ]
    for label, model_value, baseline_value, delta in metric_rows(summary):
        lines.append(
            f"| {label} | {model_value:.4f} | {baseline_value:.4f} | {delta:+.4f} |"
        )
    return "\n".join(lines)
