"""
FloodBrief - Triage engine.

Takes a flood probability map and produces a compact triage summary:
  - Flooded area in km2
  - Confidence score
  - Urgency level (CRITICAL / HIGH / MODERATE / LOW)
  - Downlink decision ("downlink" or "skip")
  - Compact JSON summary (~500 bytes)

This is the core "answer" that gets downlinked instead of raw imagery.
"""

import json
import time
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Sentinel-1/2 ground sample distance at 10 m/pixel
GSD_METERS = 10.0

# Pixel area in km2
PIXEL_AREA_KM2 = (GSD_METERS ** 2) / 1e6  # 0.0001 km2 per pixel

# Urgency thresholds (flooded area in km2)
URGENCY_THRESHOLDS = {
    "CRITICAL": 5.0,     # > 5 km2 -> critical
    "HIGH": 2.0,         # > 2 km2 -> high
    "MODERATE": 0.5,     # > 0.5 km2 -> moderate
    "LOW": 0.0,          # any detected flood -> low
}

# Minimum flood fraction to trigger downlink
MIN_FLOOD_FRACTION = 0.01  # At least 1% of tile must be flood to trigger

# Minimum confidence to trust the detection
MIN_CONFIDENCE = 0.3


# ---------------------------------------------------------------------------
# Triage output data class
# ---------------------------------------------------------------------------

@dataclass
class TriageResult:
    """Compact triage summary for downlink."""

    # Core flood metrics
    flood_detected: bool
    flooded_pixels: int
    total_pixels: int
    flood_fraction: float
    flooded_area_km2: float

    # Confidence & urgency
    confidence: float               # mean flood probability on flood pixels
    urgency: str                    # CRITICAL / HIGH / MODERATE / LOW / NONE
    downlink_decision: str          # "downlink" or "skip"

    # Metadata
    tile_id: str
    timestamp: str                  # ISO 8601
    inference_latency_ms: float     # milliseconds
    model_name: str

    # Downlink size estimate
    summary_bytes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def to_json(self, indent: Optional[int] = 2) -> str:
        """Compact JSON summary."""
        s = json.dumps(self.to_dict(), indent=indent)
        self.summary_bytes = len(s.encode("utf-8"))
        return s


# ---------------------------------------------------------------------------
# Triage engine
# ---------------------------------------------------------------------------

class TriageEngine:
    """
    FloodBrief triage engine.

    Takes a flood probability map and produces a downlink decision.
    This runs on the satellite after TerraMind inference.
    """

    def __init__(
        self,
        gsd_meters: float = GSD_METERS,
        min_flood_fraction: float = MIN_FLOOD_FRACTION,
        min_confidence: float = MIN_CONFIDENCE,
        urgency_thresholds: Optional[Dict[str, float]] = None,
        model_name: str = "terramind_v1_small",
    ):
        self.gsd_meters = gsd_meters
        self.pixel_area_km2 = (gsd_meters ** 2) / 1e6
        self.min_flood_fraction = min_flood_fraction
        self.min_confidence = min_confidence
        self.urgency_thresholds = urgency_thresholds or URGENCY_THRESHOLDS
        self.model_name = model_name

    def process(
        self,
        flood_probability: np.ndarray,
        tile_id: str = "unknown",
        threshold: float = 0.5,
        inference_latency_ms: float = 0.0,
    ) -> TriageResult:
        """
        Produce triage result from a flood probability map.

        Args:
            flood_probability: (H, W) array of flood probabilities [0, 1].
            tile_id: Unique tile identifier.
            threshold: Classification threshold.
            inference_latency_ms: Time taken for model inference.

        Returns:
            TriageResult with all triage fields populated.
        """
        # Binary mask
        binary_mask = (flood_probability >= threshold).astype(np.uint8)

        # Pixel counts
        total_pixels = binary_mask.size
        flooded_pixels = int(np.sum(binary_mask))
        flood_fraction = flooded_pixels / total_pixels if total_pixels > 0 else 0.0

        # Area estimation
        flooded_area_km2 = flooded_pixels * self.pixel_area_km2

        # Confidence: mean probability on flood pixels
        if flooded_pixels > 0:
            confidence = float(np.mean(flood_probability[binary_mask == 1]))
        else:
            confidence = 0.0

        # Flood detection decision
        flood_detected = (
            flood_fraction >= self.min_flood_fraction
            and confidence >= self.min_confidence
        )

        # Urgency level
        urgency = self._compute_urgency(flooded_area_km2, flood_detected)

        # Downlink decision
        downlink_decision = "downlink" if flood_detected else "skip"

        # Timestamp
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        result = TriageResult(
            flood_detected=flood_detected,
            flooded_pixels=flooded_pixels,
            total_pixels=total_pixels,
            flood_fraction=round(flood_fraction, 4),
            flooded_area_km2=round(flooded_area_km2, 4),
            confidence=round(confidence, 4),
            urgency=urgency,
            downlink_decision=downlink_decision,
            tile_id=tile_id,
            timestamp=timestamp,
            inference_latency_ms=round(inference_latency_ms, 2),
            model_name=self.model_name,
        )

        return result

    def _compute_urgency(self, flooded_area_km2: float, flood_detected: bool) -> str:
        """Assign urgency based on flooded area."""
        if not flood_detected:
            return "NONE"

        for level, threshold in sorted(
            self.urgency_thresholds.items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            if flooded_area_km2 >= threshold:
                return level

        return "LOW"

    def batch_process(
        self,
        flood_probabilities: list,
        tile_ids: list,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Process a batch of tiles and produce aggregate triage stats.

        Returns dict with per-tile results + aggregate summary.
        """
        results = []
        for prob, tid in zip(flood_probabilities, tile_ids):
            r = self.process(prob, tile_id=tid, threshold=threshold)
            results.append(r)

        # Aggregate stats
        total_tiles = len(results)
        downlinked = sum(1 for r in results if r.downlink_decision == "downlink")
        skipped = total_tiles - downlinked
        flood_positive = sum(1 for r in results if r.flood_detected)

        return {
            "tiles": [r.to_dict() for r in results],
            "summary": {
                "total_tiles": total_tiles,
                "downlinked": downlinked,
                "skipped": skipped,
                "skip_rate": round(skipped / total_tiles, 4) if total_tiles > 0 else 0,
                "flood_positive": flood_positive,
                "total_flooded_area_km2": round(
                    sum(r.flooded_area_km2 for r in results), 4
                ),
                "bandwidth_saving_pct": round(
                    100.0 * skipped / total_tiles, 1
                ) if total_tiles > 0 else 0,
            },
        }


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def triage_tile(
    flood_probability: np.ndarray,
    tile_id: str = "unknown",
    threshold: float = 0.5,
    inference_latency_ms: float = 0.0,
) -> TriageResult:
    """Quick triage of a single tile."""
    engine = TriageEngine()
    return engine.process(
        flood_probability,
        tile_id=tile_id,
        threshold=threshold,
        inference_latency_ms=inference_latency_ms,
    )
