# FloodBrief — Model Card

## Model Overview

| Field | Value |
|-------|-------|
| **Model name** | FloodBrief v1.0 |
| **Base model** | [TerraMind-1.0-small](https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-small) |
| **Architecture** | TerraMind ViT encoder (frozen) + UPerNet segmentation head (trainable) |
| **Task** | Binary flood segmentation (flood / no flood) |
| **Input** | Sentinel-1 SAR (VV + VH), 224×224 pixels, 10 m GSD |
| **Output** | Per-pixel flood probability map (224×224) + triage JSON |
| **Parameters** | ~100M encoder (frozen) + ~2M head (trainable) |
| **Model size** | ~200 MB (FP16) |
| **Training data** | [Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11) |
| **License** | MIT (this project), Apache-2.0 (TerraMind) |

## Intended Use

- **Primary:** Flood detection on satellite SAR imagery for disaster response triage
- **Deployment target:** Nvidia Jetson Orin Nano aboard TakeMe2Space MOI satellites
- **Users:** Disaster response agencies, crop insurers, humanitarian organizations

## Training Details

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW |
| Learning rate | 5e-4 (head only) |
| Weight decay | 1e-4 |
| Scheduler | CosineAnnealingLR |
| Epochs | 20 |
| Batch size | 8 |
| Image size | 224×224 |
| Loss | Cross-entropy, class weights [1.0, 3.0] |
| Encoder | Frozen |
| Data augmentation | Random flips, 90° rotations |

## Evaluation Results

### Segmentation metrics (Sen1Floods11 test split)

| Metric | FloodBrief | Majority-class baseline |
|--------|-----------|------------------------|
| mIoU | 0.72 | 0.33 |
| Flood IoU | 0.58 | 0.00 |
| F1 (flood) | 0.73 | 0.00 |
| Precision | 0.76 | 0.00 |
| Recall | 0.71 | 0.00 |

### Inference latency

| Hardware | Latency (batch 1) |
|----------|-------------------|
| Colab T4 GPU | ~0.8 sec |
| Jetson Orin Nano (estimated) | ~2 sec |
| CPU only | ~12 sec |

## Limitations

1. **SAR-only.** Does not use optical bands (Sentinel-2) even when available.
2. **Binary mask.** No flood depth or severity estimation.
3. **Single temporal.** No pre-/post-flood comparison.
4. **Region bias.** Trained on 11 specific flood events; may underperform on unseen geographies.
5. **Flat terrain assumption.** Area estimates assume flat ground + 10 m GSD.
6. **Not validated on Jetson.** Latency is estimated from FLOPs, not measured.

## Ethical Considerations

- **False negatives can cost lives.** This system is a triage aid, not a replacement for human analysts or in-situ monitoring.
- **Calibration matters.** Confidence scores should be validated per-region before operational use.
- **Bias in training data.** Sen1Floods11 covers only 11 events across limited geographies.

## Citation

```bibtex
@article{jakubik2025terramind,
  title={TerraMind: Large-Scale Generative Multimodality for Earth Observation},
  author={Jakubik, Johannes and Yang, Felix and others},
  journal={IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2025}
}

@inproceedings{bonafilia2020sen1floods11,
  title={Sen1Floods11: A Georeferenced Dataset to Train and Test Deep Learning Flood Algorithms for Sentinel-1},
  author={Bonafilia, Derrick and Tellman, Beth and Anderson, Tyler and Issenberg, Erica},
  booktitle={CVPR Workshops},
  year={2020}
}
```
