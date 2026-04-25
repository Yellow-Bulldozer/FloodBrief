"""
Shared helpers for single-tile inference and the Gradio demo.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

try:
    import rasterio
except ImportError:
    rasterio = None

try:
    from PIL import Image
except ImportError:
    Image = None


S1_MEAN = np.array([-12.599, -20.293], dtype=np.float32).reshape(-1, 1, 1)
S1_STD = np.array([5.195, 5.890], dtype=np.float32).reshape(-1, 1, 1)


def normalize_s1(img: np.ndarray) -> np.ndarray:
    """Normalize Sentinel-1 SAR data using TerraMind statistics."""
    return (img - S1_MEAN) / (S1_STD + 1e-8)


def _ensure_two_channels(data: np.ndarray) -> np.ndarray:
    if data.shape[0] > 2:
        return data[:2]
    if data.shape[0] < 2:
        return np.repeat(data, 2, axis=0)[:2]
    return data


def _center_crop_and_resize(data: np.ndarray, img_size: int) -> np.ndarray:
    _, height, width = data.shape
    min_dim = min(height, width)
    top = (height - min_dim) // 2
    left = (width - min_dim) // 2
    data = data[:, top:top + min_dim, left:left + min_dim]

    if min_dim != img_size:
        tensor = torch.from_numpy(data).unsqueeze(0)
        tensor = torch.nn.functional.interpolate(
            tensor,
            size=(img_size, img_size),
            mode="bilinear",
            align_corners=False,
        )
        data = tensor.squeeze(0).numpy()

    return data


def load_tile(path: str, img_size: int = 224) -> np.ndarray:
    """Load and normalize a Sentinel-1 SAR tile or standard image."""
    tile_path = Path(path)

    if rasterio is not None and tile_path.suffix.lower() in {".tif", ".tiff"}:
        with rasterio.open(tile_path) as src:
            data = src.read().astype(np.float32)
    else:
        if Image is None:
            raise ImportError("Pillow is required to load non-TIFF images.")
        img = np.array(Image.open(tile_path), dtype=np.float32)
        if img.ndim == 2:
            data = img[np.newaxis]
        elif img.ndim == 3:
            data = img.transpose(2, 0, 1)
        else:
            raise ValueError(f"Unsupported image shape: {img.shape}")

    data = _ensure_two_channels(data)
    data = _center_crop_and_resize(data, img_size)
    data = normalize_s1(data)
    return np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)


def generate_synthetic_tile(
    img_size: int = 224,
    has_flood: bool = True,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate a synthetic SAR-like tile for demos and smoke tests."""
    rng = np.random.default_rng(seed)
    image = rng.uniform(0.3, 0.7, (2, img_size, img_size)).astype(np.float32)

    try:
        from scipy.ndimage import gaussian_filter
    except ImportError as exc:
        raise ImportError(
            "scipy is required for synthetic demo generation."
        ) from exc

    for channel in range(2):
        image[channel] = gaussian_filter(image[channel], sigma=5)
        image[channel] = (
            (image[channel] - image[channel].min())
            / (image[channel].max() - image[channel].min() + 1e-8)
        )
        image[channel] = image[channel] * 0.5 + 0.2

    if has_flood:
        center_x = rng.integers(60, img_size - 60)
        center_y = rng.integers(60, img_size - 60)
        radius_x = rng.integers(30, 70)
        radius_y = rng.integers(30, 70)

        yy, xx = np.ogrid[:img_size, :img_size]
        mask = (
            ((xx - center_x) ** 2) / (radius_x ** 2)
            + ((yy - center_y) ** 2) / (radius_y ** 2)
        ) <= 1.0

        image[0][mask] *= 0.15
        image[1][mask] *= 0.15

        edge = gaussian_filter(mask.astype(np.float32), sigma=3) > 0.3
        noise_mask = edge & ~mask
        image[0][noise_mask] *= 0.5
        image[1][noise_mask] *= 0.5

    return image
