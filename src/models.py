"""
models.py — DeepLense GSoC 2026 Model Definitions
===================================================
Architecture family:
    1. ResNetBaseline      — 1-channel ResNet-18, trained from scratch (64×64)
    2. ResNetTransfer      — 3-channel ResNet-18, ImageNet pre-trained   (224×224)
    3. ViTChampion         — ViT-B/16, ImageNet pre-trained              (224×224)
    4. DeepLenseEnsemble   — [UPGRADED] Stacking Meta-Learner fusion
    5. EquivariantCNN      — [UPGRADED] C8-equivariant ResNet via escnn  (224×224)
                             (Phase-2 upgrade — the GSoC winning move)

Design contract shared by ALL models (1–5):
    • forward() returns RAW LOGITS (not softmax probabilities).
      Softmax is applied externally where needed (inference, ensemble fusion).
      This keeps models compatible with nn.CrossEntropyLoss during training.

    [GSOC UPGRADE NOTE]: DeepLenseEnsemble previously returned averaged probabilities.
    It has been upgraded to a Stacking Meta-Learner with a learnable linear head.
    It now returns LOGITS, making it fully compatible with standard training loops.

    • Input tensors follow the shape convention:
          (batch_size, channels, height, width)
      where channels=1 for Baseline/Equivariant (grayscale) and channels=3
      for Transfer/ViT/Ensemble.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# Module-level cache for the optional escnn/e2cnn import.
_E2NN_MODULE = None
_GSPACES_MODULE = None


# ─────────────────────────────────────────────────────────────────────────────
# 1. BASELINE — ResNet-18 from scratch, 1-channel grayscale, 64×64
# ─────────────────────────────────────────────────────────────────────────────

class ResNetBaseline(nn.Module):
    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.model = models.resnet18(weights=None)
        self.model.conv1 = nn.Conv2d(
            in_channels=1, out_channels=64, kernel_size=7,
            stride=2, padding=3, bias=False,
        )
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ─────────────────────────────────────────────────────────────────────────────
# 2. TRANSFER — ResNet-18, ImageNet weights, 3-channel RGB, 224×224
# ─────────────────────────────────────────────────────────────────────────────

class ResNetTransfer(nn.Module):
    def __init__(self, num_classes: int = 3, freeze_backbone: bool = False) -> None:
        super().__init__()
        self.model = models.resnet18(weights='IMAGENET1K_V1')
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.fc.parameters():
                param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ─────────────────────────────────────────────────────────────────────────────
# 3. ViT CHAMPION — ViT-B/16, ImageNet weights, 224×224
# ─────────────────────────────────────────────────────────────────────────────

class ViTChampion(nn.Module):
    def __init__(self, num_classes: int = 3, freeze_backbone: bool = False) -> None:
        super().__init__()
        self.model = models.vit_b_16(weights='IMAGENET1K_V1')

        in_features = self.model.heads.head.in_features
        self.model.heads.head = nn.Linear(in_features, num_classes)

        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.heads.head.parameters():
                param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ─────────────────────────────────────────────────────────────────────────────
# 4. ENSEMBLE — [GSOC UPGRADE 1: Stacking Meta-Learner]
# ─────────────────────────────────────────────────────────────────────────────

class DeepLenseEnsemble(nn.Module):
    """
    [UPGRADED] Stacking Meta-Learner fusion of ResNetTransfer and ViTChampion.

    Fusion strategy: 
        Instead of a naive 50/50 average, this model concatenates the logits 
        from both base models and passes them through a learnable Linear layer.
        This allows the network to *learn* that ResNet is more reliable for CDM 
        and ViT is more reliable for Vortex, dynamically adjusting weights.

    Output Contract:
        Returns raw LOGITS (B, 3). This fixes the previous probability output 
        and allows this fusion head to be trained using standard CrossEntropyLoss.
    """

    def __init__(
        self,
        resnet_model: ResNetTransfer,
        vit_model: ViTChampion,
        freeze_base: bool = True,
        learnable_fusion: bool = True, # Set to False to fallback to old soft-voting
    ) -> None:
        super().__init__()
        self.resnet = resnet_model
        self.vit    = vit_model
        self.learnable_fusion = learnable_fusion

        # Freeze the heavy feature extractors so we ONLY train the fusion head
        if freeze_base:
            for param in self.resnet.parameters():
                param.requires_grad = False
            for param in self.vit.parameters():
                param.requires_grad = False

        if self.learnable_fusion:
            # 3 logits from ResNet + 3 logits from ViT = 6 input features
            self.fusion_head = nn.Linear(6, 3)
            
            # Optional: initialize weights to mimic the old 50/50 split initially
            # to give the meta-learner a good starting point.
            nn.init.constant_(self.fusion_head.weight, 0.0)
            nn.init.constant_(self.fusion_head.bias, 0.0) # <--- CRITICAL FIX APPLIED
            with torch.no_grad():
                self.fusion_head.weight[0, 0] = 0.5  # ResNet class 0
                self.fusion_head.weight[0, 3] = 0.5  # ViT class 0
                self.fusion_head.weight[1, 1] = 0.5  # ResNet class 1
                self.fusion_head.weight[1, 4] = 0.5  # ViT class 1
                self.fusion_head.weight[2, 2] = 0.5  # ResNet class 2
                self.fusion_head.weight[2, 5] = 0.5  # ViT class 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != 3:
            raise ValueError(f"DeepLenseEnsemble expects 3-ch RGB, got {x.shape}")

        resnet_logits = self.resnet(x)   # (B, 3)
        vit_logits = self.vit(x)         # (B, 3)

        if self.learnable_fusion:
            # Concatenate logits -> (B, 6)
            combined_logits = torch.cat([resnet_logits, vit_logits], dim=1)
            # Pass through Meta-Learner -> (B, 3) LOGITS
            return self.fusion_head(combined_logits)
        else:
            # Legacy fallback (returns probabilities)
            resnet_probs  = F.softmax(resnet_logits, dim=1)
            vit_probs  = F.softmax(vit_logits, dim=1)
            return (resnet_probs + vit_probs) / 2.0


# ─────────────────────────────────────────────────────────────────────────────
# 5. EQUIVARIANT CNN — [GSOC UPGRADE 2: C8 Continuous Approximation]
# ─────────────────────────────────────────────────────────────────────────────

class EquivariantCNN(nn.Module):
    """
    [UPGRADED] C8-Equivariant CNN for gravitational lens classification.

    Scientific motivation:
        Upgraded from C4 (90° steps) to C8 (45° steps). C8 closely approximates 
        continuous SO(2) symmetry while allowing the use of standard ReLU and 
        MaxPool operations (which require regular representations). This makes 
        the network mathematically robust against virtually any arbitrary 
        rotational augmentation, perfectly aligning with the physics of 
        gravitational lensing.

    Returns raw logits. 
    Input shape:  (B, 1, 224, 224) — grayscale
    """

    # Changed default n_rotations to 8 (C8 group)
    def __init__(self, num_classes: int = 3, n_rotations: int = 8) -> None:
        super().__init__()

        global _E2NN_MODULE, _GSPACES_MODULE

        if _E2NN_MODULE is None:
            try:
                from escnn import gspaces, nn as e2nn
                _GSPACES_MODULE = gspaces
                _E2NN_MODULE    = e2nn
            except ImportError:
                raise ImportError("\n\n  pip install escnn is required.\n")

        e2nn    = _E2NN_MODULE
        gspaces = _GSPACES_MODULE

        # ── Define the symmetry group ─────────────────────────────────────
        if hasattr(gspaces, 'rot2dOnR2'):
            self.r2_act = gspaces.rot2dOnR2(N=n_rotations)
        else:
            self.r2_act = gspaces.Rot2dOnR2(N=n_rotations)

        # ── Feature field types ───────────────────────────────────────────
        in_type = e2nn.FieldType(self.r2_act, [self.r2_act.trivial_repr])
        out16   = e2nn.FieldType(self.r2_act, 16  * [self.r2_act.regular_repr])
        out32   = e2nn.FieldType(self.r2_act, 32  * [self.r2_act.regular_repr])
        out64   = e2nn.FieldType(self.r2_act, 64  * [self.r2_act.regular_repr])
        out128  = e2nn.FieldType(self.r2_act, 128 * [self.r2_act.regular_repr])

        self.input_type = in_type

        # ── Equivariant backbone ──────────────────────────────────────────
        self.backbone = e2nn.SequentialModule(
            e2nn.R2Conv(in_type, out16, kernel_size=7, stride=2, padding=3, bias=False),
            e2nn.InnerBatchNorm(out16),
            e2nn.ReLU(out16, inplace=True),
            e2nn.PointwiseMaxPool(out16, kernel_size=3, stride=2, padding=1),

            e2nn.R2Conv(out16, out32, kernel_size=3, stride=2, padding=1, bias=False),
            e2nn.InnerBatchNorm(out32),
            e2nn.ReLU(out32, inplace=True),

            e2nn.R2Conv(out32, out64, kernel_size=3, stride=2, padding=1, bias=False),
            e2nn.InnerBatchNorm(out64),
            e2nn.ReLU(out64, inplace=True),

            e2nn.R2Conv(out64, out128, kernel_size=3, stride=2, padding=1, bias=False),
            e2nn.InnerBatchNorm(out128),
            e2nn.ReLU(out128, inplace=True),
        )

        # ── Group pooling → invariant features ───────────────────────────
        self.group_pool = e2nn.GroupPooling(out128)
        pooled_channels = len(self.group_pool.out_type.representations)

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(pooled_channels, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e2nn = _E2NN_MODULE
        x_geo = e2nn.GeometricTensor(x, self.input_type)
        features = self.backbone(x_geo)
        features = self.group_pool(features).tensor
        features = self.gap(features).flatten(start_dim=1)
        return self.classifier(features)


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY — Load a saved model cleanly
# ─────────────────────────────────────────────────────────────────────────────
def load_model(model: nn.Module, weights_path: str, device: torch.device) -> nn.Module:
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"✅ Loaded weights from '{weights_path}' → device: {device}")
    return model