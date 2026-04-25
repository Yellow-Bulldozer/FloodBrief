"""
FloodBrief - Sen1Floods11 data loading.

Handles loading and preprocessing the Sen1Floods11 v1.1 dataset
for training and evaluation.

Dataset: https://github.com/cloudtostreet/Sen1Floods11
Format: GeoTIFF tiles, 512x512 pixels at 10 m GSD, EPSG:4326
Channels: Sentinel-1 SAR (VV, VH) + optional Sentinel-2 (13 bands) + labels

v1.1 directory layout:
  sen1floods11_v1.1/
    data/
      S1GRDHand/   -> *_S1Hand.tif  (2-band: VV, VH in dB)
      S2L1CHand/   -> *_S2Hand.tif  (13-band Sentinel-2 L1C)
      LabelHand/   -> *_LabelHand.tif (0=no flood, 1=flood, -1=nodata)
    splits/
      flood_train_data.txt
      flood_valid_data.txt
      flood_test_data.txt
"""

import os
import glob
import argparse
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

try:
    import rasterio
except ImportError:
    rasterio = None

# ---------------------------------------------------------------------------
# IBM TerraMind official normalization statistics (pre-training values)
# ---------------------------------------------------------------------------

NORM_STATS = {
    "S1GRD": {
        "mean": [-12.599, -20.293],
        "std": [5.195, 5.890],
    },
    "S2L1C": {
        "mean": [2357.089, 2137.385, 2018.788, 2082.986, 2295.651,
                 2854.537, 3122.849, 3040.560, 3306.481, 1473.847,
                 506.070, 2472.825, 1838.929],
        "std": [1624.683, 1675.806, 1557.708, 1833.702, 1823.738,
                1733.977, 1732.131, 1679.732, 1727.260, 1024.687,
                442.165, 1331.411, 1160.419],
    },
}

# Label mapping
LABEL_NODATA = -1   # pixels without valid annotation
LABEL_NO_FLOOD = 0
LABEL_FLOOD = 1
IGNORE_INDEX = -1   # used in loss functions to skip nodata pixels


# ---------------------------------------------------------------------------
# Sen1Floods11 v1.1 Dataset
# ---------------------------------------------------------------------------

class Sen1Floods11Dataset(Dataset):
    """
    PyTorch dataset for Sen1Floods11 v1.1.

    Loads Sentinel-1 SAR tiles (VV + VH) and corresponding flood masks.
    Optionally loads Sentinel-2 L1C tiles for multimodal experiments.
    Uses official train/val/test split files.

    Label mapping:
        -1 -> ignore (nodata)
         0 -> no flood
         1 -> flood
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        img_size: int = 224,
        augment: bool = False,
        max_samples: Optional[int] = None,
        use_s2: bool = False,
        normalize: str = "terramind",
    ):
        """
        Args:
            data_dir: Root of sen1floods11_v1.1/ directory
            split: 'train', 'val', or 'test'
            img_size: Target image size (center-crop + resize)
            augment: Apply training augmentations (flips, rotations)
            max_samples: Limit number of samples (for debugging)
            use_s2: Also load Sentinel-2 L1C data
            normalize: 'terramind' (IBM stats) or 'minmax' (clip dB to [0,1])
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split = split
        self.img_size = img_size
        self.augment = augment
        self.use_s2 = use_s2
        self.normalize = normalize

        # Resolve directories
        self.s1_dir = self.data_dir / "data" / "S1GRDHand"
        self.s2_dir = self.data_dir / "data" / "S2L1CHand"
        self.label_dir = self.data_dir / "data" / "LabelHand"

        # Discover samples using split files
        self.samples = self._discover_samples()

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        print(f"[Sen1Floods11] {split} split: {len(self.samples)} samples "
              f"(S1{'+ S2' if use_s2 else ''}, {img_size}x{img_size}, "
              f"norm={normalize})")

    def _discover_samples(self) -> List[Dict[str, str]]:
        """
        Find sample pairs using official split files.

        Split files contain chip IDs like 'Bolivia_103757'.
        Each chip has corresponding S1, S2, and label TIFFs.
        """
        # Map split name to file
        split_map = {
            "train": "flood_train_data.txt",
            "val": "flood_valid_data.txt",
            "valid": "flood_valid_data.txt",
            "test": "flood_test_data.txt",
        }

        split_file = self.data_dir / "splits" / split_map.get(self.split, "flood_train_data.txt")

        if not split_file.exists():
            print(f"[Sen1Floods11] WARNING: Split file not found: {split_file}")
            print("[Sen1Floods11] Falling back to directory scan...")
            return self._discover_samples_by_scan()

        # Read chip IDs from split file
        with open(split_file, "r") as f:
            chip_ids = [line.strip() for line in f if line.strip()]

        # Build sample list
        samples = []
        missing_s1 = 0
        missing_label = 0

        for chip_id in chip_ids:
            s1_path = self.s1_dir / f"{chip_id}_S1Hand.tif"
            label_path = self.label_dir / f"{chip_id}_LabelHand.tif"

            if not s1_path.exists():
                missing_s1 += 1
                continue
            if not label_path.exists():
                missing_label += 1
                continue

            sample = {
                "chip_id": chip_id,
                "event": chip_id.split("_")[0],
                "s1": str(s1_path),
                "label": str(label_path),
            }

            # Optional: Sentinel-2
            if self.use_s2:
                s2_path = self.s2_dir / f"{chip_id}_S2Hand.tif"
                if s2_path.exists():
                    sample["s2"] = str(s2_path)
                else:
                    sample["s2"] = None

            samples.append(sample)

        if missing_s1 > 0 or missing_label > 0:
            print(f"[Sen1Floods11] WARNING: {missing_s1} missing S1, {missing_label} missing labels")

        return samples

    def _discover_samples_by_scan(self) -> List[Dict[str, str]]:
        """
        Fallback: scan S1GRDHand/ directory and match labels.
        Used when split files are not available.
        """
        if not self.s1_dir.exists():
            return []

        samples = []
        for s1_path in sorted(self.s1_dir.glob("*_S1Hand.tif")):
            chip_id = s1_path.stem.replace("_S1Hand", "")
            label_path = self.label_dir / f"{chip_id}_LabelHand.tif"

            if label_path.exists():
                sample = {
                    "chip_id": chip_id,
                    "event": chip_id.split("_")[0],
                    "s1": str(s1_path),
                    "label": str(label_path),
                }
                if self.use_s2:
                    s2_path = self.s2_dir / f"{chip_id}_S2Hand.tif"
                    sample["s2"] = str(s2_path) if s2_path.exists() else None
                samples.append(sample)

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Load SAR tile (2 bands: VV, VH in dB)
        s1 = self._load_tif(sample["s1"])  # (C, H, W)

        # Ensure 2-channel SAR (VV, VH)
        if s1.shape[0] > 2:
            s1 = s1[:2]
        elif s1.shape[0] < 2:
            s1 = np.repeat(s1, 2, axis=0)[:2]

        # Load and process label
        label_raw = self._load_tif(sample["label"])  # (1, H, W) or (H, W)
        if label_raw.ndim == 3:
            label_raw = label_raw[0]

        # Map label values:
        #   -1 (nodata) -> IGNORE_INDEX (-1)
        #    0 (no flood) -> 0
        #    1 (flood) -> 1
        #   255 (sometimes used for nodata) -> IGNORE_INDEX
        label = np.full_like(label_raw, IGNORE_INDEX, dtype=np.int64)
        label[label_raw == 0] = LABEL_NO_FLOOD
        label[label_raw == 1] = LABEL_FLOOD

        # Resize / center-crop to img_size
        s1 = self._resize(s1, self.img_size)
        label = self._resize_label(label, self.img_size)

        # Normalize SAR
        s1 = self._normalize_s1(s1, method=self.normalize)

        # Replace NaN/inf
        s1 = np.nan_to_num(s1, nan=0.0, posinf=0.0, neginf=0.0)

        # Augmentation (training only)
        if self.augment:
            s1, label = self._augment(s1, label)

        result = {
            "image": torch.from_numpy(s1).float(),
            "label": torch.from_numpy(label).long(),
            "chip_id": sample["chip_id"],
            "event": sample["event"],
        }

        # Optional Sentinel-2
        if self.use_s2 and sample.get("s2") is not None:
            s2 = self._load_tif(sample["s2"])
            s2 = self._resize(s2, self.img_size)
            s2 = self._normalize_s2(s2)
            s2 = np.nan_to_num(s2, nan=0.0, posinf=0.0, neginf=0.0)
            result["image_s2"] = torch.from_numpy(s2).float()

        return result

    # --- I/O ---

    @staticmethod
    def _load_tif(path: str) -> np.ndarray:
        """Load a GeoTIFF file. Returns (C, H, W) numpy array."""
        if rasterio is not None:
            with rasterio.open(path) as src:
                data = src.read()  # (C, H, W)
                return data.astype(np.float32)
        else:
            from PIL import Image
            img = Image.open(path)
            data = np.array(img, dtype=np.float32)
            if data.ndim == 2:
                data = data[np.newaxis]
            elif data.ndim == 3:
                data = data.transpose(2, 0, 1)
            return data

    # --- Preprocessing ---

    @staticmethod
    def _resize(img: np.ndarray, size: int) -> np.ndarray:
        """Center-crop and resize image to (C, size, size)."""
        C, H, W = img.shape

        # Center-crop to square
        min_dim = min(H, W)
        top = (H - min_dim) // 2
        left = (W - min_dim) // 2
        img = img[:, top:top + min_dim, left:left + min_dim]

        # Resize
        if min_dim != size:
            import torch.nn.functional as F
            t = torch.from_numpy(img).unsqueeze(0)  # (1, C, H, W)
            t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
            img = t.squeeze(0).numpy()

        return img

    @staticmethod
    def _resize_label(label: np.ndarray, size: int) -> np.ndarray:
        """Center-crop and resize label to (size, size) using nearest neighbor."""
        H, W = label.shape

        min_dim = min(H, W)
        top = (H - min_dim) // 2
        left = (W - min_dim) // 2
        label = label[top:top + min_dim, left:left + min_dim]

        if min_dim != size:
            import torch.nn.functional as F
            t = torch.from_numpy(label.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            t = F.interpolate(t, size=(size, size), mode="nearest")
            label = t.squeeze().numpy().astype(np.int64)

        return label

    @staticmethod
    def _normalize_s1(img: np.ndarray, method: str = "terramind") -> np.ndarray:
        """
        Normalize Sentinel-1 SAR data.

        Methods:
          'terramind': Use IBM's pre-training mean/std (recommended for fine-tuning)
          'minmax':    Clip dB to [-30, 0] and scale to [0, 1]
        """
        if method == "terramind":
            mean = np.array(NORM_STATS["S1GRD"]["mean"], dtype=np.float32).reshape(-1, 1, 1)
            std = np.array(NORM_STATS["S1GRD"]["std"], dtype=np.float32).reshape(-1, 1, 1)
            return (img - mean) / (std + 1e-8)
        else:
            # Legacy minmax normalization
            img = np.clip(img, -30.0, 0.0)
            img = (img + 30.0) / 30.0
            return img

    @staticmethod
    def _normalize_s2(img: np.ndarray) -> np.ndarray:
        """Normalize Sentinel-2 L1C data using IBM's pre-training stats."""
        num_bands = min(img.shape[0], len(NORM_STATS["S2L1C"]["mean"]))
        mean = np.array(NORM_STATS["S2L1C"]["mean"][:num_bands], dtype=np.float32).reshape(-1, 1, 1)
        std = np.array(NORM_STATS["S2L1C"]["std"][:num_bands], dtype=np.float32).reshape(-1, 1, 1)
        img = img[:num_bands]
        return (img - mean) / (std + 1e-8)

    # --- Augmentation ---

    @staticmethod
    def _augment(img: np.ndarray, label: np.ndarray):
        """D4 augmentation: random flips and 90-degree rotations."""
        # Random horizontal flip
        if np.random.random() > 0.5:
            img = img[:, :, ::-1].copy()
            label = label[:, ::-1].copy()

        # Random vertical flip
        if np.random.random() > 0.5:
            img = img[:, ::-1, :].copy()
            label = label[::-1, :].copy()

        # Random 90-degree rotation
        k = np.random.randint(0, 4)
        if k > 0:
            img = np.rot90(img, k, axes=(1, 2)).copy()
            label = np.rot90(label, k, axes=(0, 1)).copy()

        return img, label


# ---------------------------------------------------------------------------
# Synthetic data fallback (for testing without real data)
# ---------------------------------------------------------------------------

class SyntheticFloodDataset(Dataset):
    """
    Generates synthetic SAR-like tiles + flood masks for testing
    the pipeline when Sen1Floods11 is not available.
    """

    def __init__(self, num_samples: int = 100, img_size: int = 224):
        self.num_samples = num_samples
        self.img_size = img_size
        np.random.seed(42)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Synthetic SAR: random noise + circular "water" region
        img = np.random.uniform(0.2, 0.8, (2, self.img_size, self.img_size)).astype(np.float32)

        # Create a random flood region (ellipse)
        has_flood = np.random.random() > 0.3  # 70% of tiles have flood
        label = np.zeros((self.img_size, self.img_size), dtype=np.int64)

        if has_flood:
            cx = np.random.randint(40, self.img_size - 40)
            cy = np.random.randint(40, self.img_size - 40)
            rx = np.random.randint(20, 60)
            ry = np.random.randint(20, 60)

            yy, xx = np.ogrid[:self.img_size, :self.img_size]
            mask = ((xx - cx) ** 2 / rx ** 2 + (yy - cy) ** 2 / ry ** 2) <= 1.0
            label[mask] = 1

            # Make SAR darker in flooded areas (water absorbs SAR)
            img[0][mask] *= 0.3
            img[1][mask] *= 0.3

        return {
            "image": torch.from_numpy(img).float(),
            "label": torch.from_numpy(label).long(),
            "chip_id": f"synthetic_{idx:04d}",
            "event": "synthetic",
        }


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def get_dataloaders(
    data_dir: str,
    batch_size: int = 8,
    img_size: int = 224,
    num_workers: int = 4,
    use_synthetic: bool = False,
    max_samples: Optional[int] = None,
    use_s2: bool = False,
    normalize: str = "terramind",
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.

    Args:
        data_dir: Path to sen1floods11_v1.1/ root
        batch_size: Batch size
        img_size: Input image size
        num_workers: Number of dataloader workers
        use_synthetic: Force synthetic data
        max_samples: Limit dataset size
        use_s2: Include Sentinel-2 data
        normalize: 'terramind' or 'minmax'

    Returns:
        (train_loader, val_loader, test_loader)
    """
    if use_synthetic:
        print("[FloodBrief] Using synthetic data (no real dataset found).")
        train_ds = SyntheticFloodDataset(num_samples=200, img_size=img_size)
        val_ds = SyntheticFloodDataset(num_samples=50, img_size=img_size)
        test_ds = SyntheticFloodDataset(num_samples=50, img_size=img_size)
    else:
        train_ds = Sen1Floods11Dataset(
            data_dir, split="train", img_size=img_size,
            augment=True, max_samples=max_samples,
            use_s2=use_s2, normalize=normalize,
        )
        val_ds = Sen1Floods11Dataset(
            data_dir, split="val", img_size=img_size,
            max_samples=max_samples, use_s2=use_s2, normalize=normalize,
        )
        test_ds = Sen1Floods11Dataset(
            data_dir, split="test", img_size=img_size,
            max_samples=max_samples, use_s2=use_s2, normalize=normalize,
        )

        # Fall back to synthetic if no real data found
        if len(train_ds) == 0:
            print("[FloodBrief] No Sen1Floods11 data found. Falling back to synthetic data.")
            return get_dataloaders(
                data_dir, batch_size, img_size, num_workers,
                use_synthetic=True,
            )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def download_sen1floods11(data_dir: str, method: str = "gdown"):
    """
    Download Sen1Floods11 v1.1 dataset.

    Methods:
        'gdown':  Download from Google Drive (recommended, ~14 GB)
        'gsutil': Download from GCS bucket
    """
    import subprocess
    os.makedirs(data_dir, exist_ok=True)
    parent = str(Path(data_dir).parent)

    if method == "gdown":
        print("[FloodBrief] Downloading Sen1Floods11 v1.1 from Google Drive...")
        print("[FloodBrief] File size: ~14 GB. This will take a while.")
        tar_path = os.path.join(parent, "sen1floods11_v1.1.tar.gz")
        try:
            import gdown
            gdown.download(
                "https://drive.google.com/uc?id=1lRw3X7oFNq_WyzBO6uyUJijyTuYm23VS",
                tar_path, quiet=False,
            )
            print(f"[FloodBrief] Extracting to {parent}...")
            subprocess.run(["tar", "-xzf", tar_path, "-C", parent], check=True)
            print("[FloodBrief] Download + extraction complete!")
        except Exception as e:
            print(f"[FloodBrief] gdown failed: {e}")
            print("[FloodBrief] Install gdown: pip install gdown")
            print("[FloodBrief] Or download manually from:")
            print("  https://drive.google.com/uc?id=1lRw3X7oFNq_WyzBO6uyUJijyTuYm23VS")

    elif method == "gsutil":
        cmd = f"gsutil -m rsync -r gs://sen1floods11 {data_dir}"
        print(f"[FloodBrief] Running: {cmd}")
        try:
            subprocess.run(cmd, shell=True, check=True)
            print("[FloodBrief] Download complete.")
        except Exception as e:
            print(f"[FloodBrief] gsutil failed: {e}")


# ---------------------------------------------------------------------------
# CLI: verify dataset
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sen1Floods11 data utilities")
    parser.add_argument("--download", action="store_true", help="Download the dataset")
    parser.add_argument("--download-method", type=str, default="gdown",
                        choices=["gdown", "gsutil"], help="Download method")
    parser.add_argument("--data-dir", type=str, default="./data/sen1floods11_v1.1",
                        help="Dataset directory (root of sen1floods11_v1.1/)")
    parser.add_argument("--verify", action="store_true", help="Verify dataset and print stats")
    args = parser.parse_args()

    if args.download:
        download_sen1floods11(args.data_dir, method=args.download_method)

    if args.verify:
        print(f"\n{'='*60}")
        print(f"  Sen1Floods11 Dataset Verification")
        print(f"  Root: {args.data_dir}")
        print(f"{'='*60}\n")

        for split in ["train", "val", "test"]:
            ds = Sen1Floods11Dataset(args.data_dir, split=split)
            if len(ds) > 0:
                sample = ds[0]
                events = set(s["event"] for s in ds.samples)
                print(f"  {split:5s}: {len(ds):4d} samples | "
                      f"Image: {tuple(sample['image'].shape)} | "
                      f"Label unique: {torch.unique(sample['label']).tolist()} | "
                      f"Events: {sorted(events)}")
            else:
                print(f"  {split:5s}: 0 samples (no data found)")

        print(f"\n{'='*60}")
        print("  Normalization: IBM TerraMind S1GRD stats")
        print(f"  S1 mean: {NORM_STATS['S1GRD']['mean']}")
        print(f"  S1 std:  {NORM_STATS['S1GRD']['std']}")
        print(f"  Label mapping: -1=ignore, 0=no_flood, 1=flood")
        print(f"{'='*60}")
