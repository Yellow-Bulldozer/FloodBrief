"""
FloodBrief - Model definition.

Builds a TerraMind-small encoder + UPerNet segmentation head for
binary flood segmentation on Sentinel-1 SAR (VV + VH).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any


# ---------------------------------------------------------------------------
# Lightweight UPerNet-style decoder
# ---------------------------------------------------------------------------

class ConvBNReLU(nn.Module):
    """Conv2d -> BatchNorm -> ReLU helper."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PPM(nn.Module):
    """Pyramid Pooling Module (simplified)."""

    def __init__(self, in_ch: int, pool_sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = nn.ModuleList()
        for s in pool_sizes:
            self.stages.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(s),
                ConvBNReLU(in_ch, in_ch // len(pool_sizes), kernel=1, padding=0),
            ))
        self.bottleneck = ConvBNReLU(
            in_ch + (in_ch // len(pool_sizes)) * len(pool_sizes),
            in_ch,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2:]
        pools = [x]
        for stage in self.stages:
            p = stage(x)
            p = F.interpolate(p, size=(h, w), mode="bilinear", align_corners=False)
            pools.append(p)
        return self.bottleneck(torch.cat(pools, dim=1))


class SegmentationHead(nn.Module):
    """UPerNet-inspired segmentation head.

    Takes a sequence of patch embeddings from TerraMind encoder and
    decodes them into a per-pixel flood probability map.
    """

    def __init__(
        self,
        embed_dim: int = 384,
        patch_size: int = 16,
        img_size: int = 224,
        num_classes: int = 2,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size  # 14 for 224/16
        self.embed_dim = embed_dim

        # Reshape + project patch embeddings into spatial feature map
        self.proj = ConvBNReLU(embed_dim, hidden_dim, kernel=1, padding=0)

        # PPM for multi-scale context
        self.ppm = PPM(hidden_dim)

        # Final classifier
        self.classifier = nn.Sequential(
            ConvBNReLU(hidden_dim, hidden_dim),
            nn.Conv2d(hidden_dim, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Patch embeddings from TerraMind encoder.
               Shape: (B, num_patches, embed_dim) - e.g. (B, 196, 384)

        Returns:
            logits: (B, num_classes, H, W) - e.g. (B, 2, 224, 224)
        """
        B, N, C = x.shape
        H = W = self.grid_size  # 14

        # Reshape to spatial: (B, embed_dim, H, W)
        x = x.transpose(1, 2).reshape(B, C, H, W)

        # Project channels
        x = self.proj(x)

        # Multi-scale pooling
        x = self.ppm(x)

        # Per-pixel classification
        x = self.classifier(x)

        # Upsample to original resolution
        x = F.interpolate(
            x,
            size=(self.grid_size * self.patch_size, self.grid_size * self.patch_size),
            mode="bilinear",
            align_corners=False,
        )
        return x


# ---------------------------------------------------------------------------
# Full FloodBrief model
# ---------------------------------------------------------------------------

class FloodBriefModel(nn.Module):
    """
    TerraMind encoder (frozen) + UPerNet segmentation head (trainable).

    Produces per-pixel flood logits for Sentinel-1 SAR input.
    """

    def __init__(
        self,
        model_name: str = "terramind_v1_small",
        pretrained: bool = True,
        num_classes: int = 2,
        freeze_encoder: bool = True,
        embed_dim: int = 384,
        img_size: int = 224,
    ):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.img_size = img_size
        self.embed_dim = embed_dim
        self._encoder_loaded = False
        self.use_heuristic_inference = False
        self.inference_backend = "terramind"

        # Try to load the TerraMind encoder via TerraTorch registry
        try:
            from terratorch import BACKBONE_REGISTRY

            self.encoder = BACKBONE_REGISTRY.build(
                model_name,
                pretrained=pretrained,
                modalities=["S1GRD"],
            )
            self._encoder_loaded = True

            if freeze_encoder:
                for param in self.encoder.parameters():
                    param.requires_grad = False

            print(f"[FloodBrief] TerraMind encoder loaded: {model_name}")
            total_params = sum(p.numel() for p in self.encoder.parameters())
            print(f"[FloodBrief] Encoder params: {total_params / 1e6:.1f}M")

        except Exception as e:
            print(f"[FloodBrief] WARNING: Could not load TerraMind encoder: {e}")
            print("[FloodBrief] Using a fallback CNN encoder for demo purposes.")
            self.encoder = self._build_fallback_encoder()
            self._encoder_loaded = False
            self.inference_backend = "fallback_cnn"

        # Segmentation head (always trainable)
        self.head = SegmentationHead(
            embed_dim=embed_dim,
            patch_size=16,
            img_size=img_size,
            num_classes=num_classes,
        )

        total_head_params = sum(p.numel() for p in self.head.parameters())
        print(f"[FloodBrief] Head params: {total_head_params / 1e6:.1f}M")

    def _build_fallback_encoder(self) -> nn.Module:
        """Simple CNN encoder as a fallback when TerraMind is unavailable."""

        class FallbackEncoder(nn.Module):
            def __init__(self, embed_dim: int = 384):
                super().__init__()
                self.embed_dim = embed_dim
                self.features = nn.Sequential(
                    ConvBNReLU(2, 64, kernel=7, padding=3),
                    nn.MaxPool2d(2),
                    ConvBNReLU(64, 128),
                    nn.MaxPool2d(2),
                    ConvBNReLU(128, 256),
                    nn.MaxPool2d(2),
                    ConvBNReLU(256, embed_dim),
                    nn.AdaptiveAvgPool2d(14),  # -> (B, embed_dim, 14, 14)
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # x: (B, 2, 224, 224)
                feat = self.features(x)  # (B, embed_dim, 14, 14)
                B, C, H, W = feat.shape
                return feat.reshape(B, C, H * W).transpose(1, 2)  # (B, 196, embed_dim)

        return FallbackEncoder(self.embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Sentinel-1 SAR input. Shape: (B, 2, 224, 224) - channels are VV, VH.
            return_features: If True, also return encoder features.

        Returns:
            dict with:
                'logits': (B, num_classes, 224, 224)
                'probabilities': (B, num_classes, 224, 224)
                'features': (B, 196, embed_dim) - only if return_features=True
        """
        # Encode
        if self._encoder_loaded:
            # TerraMind expects a dict of modalities
            features = self.encoder({"S1GRD": x})  # (B, 196, embed_dim)
        else:
            features = self.encoder(x)

        # Ensure features shape
        if isinstance(features, (list, tuple)):
            features = features[-1]  # Take last layer if multi-scale

        # Decode
        logits = self.head(features)
        probs = F.softmax(logits, dim=1)

        result = {
            "logits": logits,
            "probabilities": probs,
        }
        if return_features:
            result["features"] = features

        return result

    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run inference and produce flood mask + probabilities."""
        self.eval()
        with torch.no_grad():
            if self.use_heuristic_inference:
                return self._predict_with_heuristic(x)
            out = self.forward(x)
            flood_mask = out["probabilities"][:, 1, :, :]  # flood class probability
            binary_mask = (flood_mask > 0.5).long()
            out["flood_probability"] = flood_mask
            out["binary_mask"] = binary_mask
            out["inference_backend"] = self.inference_backend
        return out

    def _predict_with_heuristic(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Heuristic fallback for demo environments where TerraMind weights cannot be
        loaded into the active runtime. Flooded regions are typically dark and
        consistent across VV and VH, so we score low-backscatter smooth areas.
        """
        vv = x[:, 0:1]
        vh = x[:, 1:2] if x.shape[1] > 1 else vv
        sar_mean = (vv + vh) / 2.0
        sar_std = sar_mean.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        sar_centered = sar_mean - sar_mean.mean(dim=(2, 3), keepdim=True)

        darkness_score = -sar_centered / sar_std
        channel_consistency = torch.exp(-torch.abs(vv - vh))
        smoothness = 1.0 / (
            1.0 + torch.abs(sar_mean - F.avg_pool2d(sar_mean, kernel_size=9, stride=1, padding=4))
        )

        flood_probability = torch.sigmoid(
            1.6 * darkness_score + 0.9 * channel_consistency + 0.7 * smoothness - 1.1
        ).squeeze(1)
        flood_probability = F.avg_pool2d(
            flood_probability.unsqueeze(1),
            kernel_size=5,
            stride=1,
            padding=2,
        ).squeeze(1)

        binary_mask = (flood_probability > 0.5).long()
        probabilities = torch.stack((1.0 - flood_probability, flood_probability), dim=1)
        logits = torch.log(probabilities.clamp(min=1e-6))

        return {
            "logits": logits,
            "probabilities": probabilities,
            "flood_probability": flood_probability,
            "binary_mask": binary_mask,
            "inference_backend": "heuristic_fallback",
        }


def load_model(
    checkpoint_path: Optional[str] = None,
    model_name: str = "terramind_v1_small",
    device: str = "cuda",
    **kwargs,
) -> FloodBriefModel:
    """Load FloodBriefModel, optionally from a checkpoint."""
    model = FloodBriefModel(model_name=model_name, **kwargs)

    if checkpoint_path is not None:
        print(f"[FloodBrief] Loading checkpoint: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location="cpu")
        load_result = None
        if "model_state_dict" in state:
            load_result = model.load_state_dict(state["model_state_dict"], strict=False)
        else:
            load_result = model.load_state_dict(state, strict=False)
        print("[FloodBrief] Checkpoint loaded.")

        if (
            not model._encoder_loaded
            and load_result is not None
            and any(key.startswith("encoder.") for key in load_result.unexpected_keys)
        ):
            print(
                "[FloodBrief] TerraMind checkpoint detected without TerraMind runtime; "
                "using heuristic fallback inference for the app."
            )
            model.use_heuristic_inference = True
            model.inference_backend = "heuristic_fallback"
    elif not model._encoder_loaded:
        model.use_heuristic_inference = True
        model.inference_backend = "heuristic_fallback"

    model = model.to(device)
    return model
