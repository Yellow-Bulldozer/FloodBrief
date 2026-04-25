# Sen1Floods11 Dataset Guide

## Overview

Sen1Floods11 is a georeferenced dataset for training and testing flood algorithms
on Sentinel-1 SAR imagery. It covers **11 flood events** across 6 continents with
**4,831 labeled chips** at 10m resolution.

**Citation:**
Bonafilia, D., Tellman, B., Anderson, T., Issenberg, E. 2020. "Sen1Floods11: a
georeferenced dataset to train and test deep learning flood algorithms for Sentinel-1."
CVPR Workshops 2020.

---

## Download

### Method 1: Google Drive (recommended)

```bash
pip install gdown
gdown https://drive.google.com/uc?id=1lRw3X7oFNq_WyzBO6uyUJijyTuYm23VS
tar -xzf sen1floods11_v1.1.tar.gz -C ./data/
```

### Method 2: Google Cloud Storage

```bash
gsutil -m rsync -r gs://sen1floods11 ./data/sen1floods11_v1.1
```

### Method 3: Using FloodBrief's download helper

```bash
python src/data_loader.py --download --data-dir ./data/sen1floods11_v1.1
```

**Size:** ~14 GB (compressed ~4 GB)

---

## Expected Directory Structure

After extraction, the dataset should look like this:

```
data/sen1floods11_v1.1/
  data/
    S1GRDHand/           # Sentinel-1 SAR GRD tiles (2-band: VV, VH)
      Bolivia_103757_S1Hand.tif
      Bolivia_103758_S1Hand.tif
      Ghana_109001_S1Hand.tif
      ...
    S2L1CHand/           # Sentinel-2 L1C optical tiles (13-band)
      Bolivia_103757_S2Hand.tif
      ...
    LabelHand/           # Hand-annotated flood labels
      Bolivia_103757_LabelHand.tif
      ...
  splits/
    flood_train_data.txt    # Training chip IDs (one per line)
    flood_valid_data.txt    # Validation chip IDs
    flood_test_data.txt     # Test chip IDs
```

---

## File Naming Convention

Each file follows the pattern: `{Event}_{ChipID}_{Layer}.tif`

- **Event:** Country name (Bolivia, Ghana, India, Mekong, Nigeria, Pakistan, Paraguay, Somalia, Spain, SriLanka, USA)
- **ChipID:** Unique integer identifier
- **Layer:** Data type (S1Hand, S2Hand, LabelHand)

Example: `Bolivia_103757_S1Hand.tif`

---

## Data Format

### Sentinel-1 SAR (S1GRDHand/)
- **Bands:** 2 (VV, VH polarization)
- **Values:** Backscatter in decibels (dB), typically [-30, 0]
- **Resolution:** 10m
- **CRS:** EPSG:4326 (WGS 84)
- **Tile size:** ~512x512 pixels (varies slightly)

### Sentinel-2 Optical (S2L1CHand/) — optional
- **Bands:** 13 (all S2 L1C bands)
- **Values:** Top-of-atmosphere reflectance
- **Resolution:** 10m (resampled)

### Labels (LabelHand/)
- **Bands:** 1
- **Values:**
  - `0` = **No flood**
  - `1` = **Flood**
  - `-1` = **No data** (ignore in training/evaluation)

> **Important:** The `-1` no-data label must be handled carefully.
> FloodBrief uses `ignore_index=-1` in the loss function (CrossEntropyLoss
> and DiceLoss) to skip these pixels during training.

---

## Official Splits

The dataset uses the following official train/val/test partition:

| Split | File | Events | Approx. Samples |
|-------|------|--------|-----------------|
| Train | `flood_train_data.txt` | Bolivia, Ghana, India, Mekong, Nigeria, Pakistan, Paraguay | ~3,400 |
| Val | `flood_valid_data.txt` | Somalia, Spain | ~700 |
| Test | `flood_test_data.txt` | SriLanka, USA | ~700 |

Split files contain chip IDs (e.g., `Bolivia_103757`), one per line.

---

## Normalization

FloodBrief uses **IBM TerraMind pre-training statistics** for normalization:

### Sentinel-1 GRD
| Channel | Mean | Std |
|---------|------|-----|
| VV | -12.599 | 5.195 |
| VH | -20.293 | 5.890 |

### Sentinel-2 L1C
13-band means and stds from IBM's official config — see `data_loader.py`.

---

## Verify Your Download

```bash
python src/data_loader.py --verify --data-dir ./data/sen1floods11_v1.1
```

Expected output:
```
  train:  XXX samples | Image: (2, 224, 224) | Label unique: [-1, 0, 1]
  val:    XXX samples | Image: (2, 224, 224) | Label unique: [-1, 0, 1]
  test:   XXX samples | Image: (2, 224, 224) | Label unique: [-1, 0, 1]
```

---

## Training Commands

### Custom Pipeline (FloodBrief)

```bash
# Fine-tune with unfrozen encoder (recommended for best results)
python train.py --data-dir ./data/sen1floods11_v1.1 --epochs 50 --batch-size 8 --lr 2e-5 --loss combined

# Quick test with frozen encoder (faster)
python train.py --data-dir ./data/sen1floods11_v1.1 --epochs 20 --freeze-encoder --lr 5e-4
```

### TerraTorch CLI (IBM's recommended approach)

```bash
# Train
terratorch fit -c configs/terramind_flood.yaml

# Or use the wrapper:
python train_terratorch.py --config configs/terramind_flood.yaml --epochs 50

# Test
terratorch test -c configs/terramind_flood.yaml --ckpt_path output/floodbrief/checkpoints/best-mIoU.ckpt
```

---

## Troubleshooting

### "No Sen1Floods11 data found"
- Check your `--data-dir` path points to the root of `sen1floods11_v1.1/`
- The directory must contain `data/S1GRDHand/` and `splits/flood_train_data.txt`

### Labels show only 0 and 1 (no -1)
- This is normal for some tiles that are fully annotated
- The data loader handles this correctly

### Out of memory
- Reduce `--batch-size` (try 4 or 2)
- Use `--freeze-encoder` to reduce memory
- Use `--amp` for mixed precision (GPU only)
