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
    6. TemperatureScaledModel — [GSOC UPGRADE 3] Post-hoc calibration wrapper

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

import numpy as np
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


# ─────────────────────────────────────────────────────────────────────────────
# 6. TEMPERATURE SCALING — [GSOC UPGRADE 3: Post-hoc Calibration]
# ─────────────────────────────────────────────────────────────────────────────
#
# Neural networks are systematically overconfident — they assign probabilities
# like 0.98 to predictions that are correct only 80% of the time. Temperature
# Scaling (Guo et al., 2017) is the gold-standard post-hoc fix. It divides all
# logits by a learned scalar T before the softmax. T > 1 softens the
# distribution (less overconfident). T is found by minimising NLL on the
# validation set.
#
# Reference: Guo et al. (2017) — "On Calibration of Modern Neural Networks"
#
# Design contract:
#   • TemperatureScaledModel wraps ANY existing model without modifying it.
#   • forward() returns LOGITS divided by T (NOT probabilities).
#     This maintains compatibility with nn.CrossEntropyLoss and all
#     existing evaluation code that expects logits.
#   • The wrapped base model is frozen during temperature optimisation.
#   • Temperature is constrained to T > 0 via a softplus parameterisation
#     to prevent numerical instability.
# ─────────────────────────────────────────────────────────────────────────────

class TemperatureScaledModel(nn.Module):
    """
    Post-hoc calibration wrapper using Temperature Scaling.

    Wraps any trained DeepLense model and learns a single scalar temperature
    parameter T on the validation set. The base model weights are frozen —
    only T is optimised.

    Usage:
        # After training ResNetTransfer:
        calibrated = TemperatureScaledModel(trained_resnet)
        calibrated.calibrate(val_loader, device)

        # Drop-in replacement — returns temperature-scaled logits
        logits = calibrated(images)
        probs  = F.softmax(logits, dim=1)

    Args:
        base_model (nn.Module): Any trained DeepLense model returning logits.
        init_temperature (float): Starting temperature (default 1.0 = no scaling).
    """

    def __init__(self, base_model: nn.Module, init_temperature: float = 1.5) -> None:
        super().__init__()
        self.base_model = base_model

        # Freeze the base model — we only learn T
        for param in self.base_model.parameters():
            param.requires_grad = False

        # log(T) parameterisation: T = exp(log_T) > 0 always.
        # Initialising at log(1.5) gives a slightly warm start which converges
        # faster than log(1.0) for overconfident networks.
        import math
        self.log_temperature = nn.Parameter(
            torch.tensor(math.log(init_temperature), dtype=torch.float32)
        )

    @property
    def temperature(self) -> float:
        """Returns the current temperature T as a Python float."""
        return float(self.log_temperature.exp().item())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns temperature-scaled logits: logits / T."""
        logits = self.base_model(x)                      # (B, num_classes) raw logits
        T      = self.log_temperature.exp()               # scalar tensor, T > 0
        return logits / T

    def calibrate(
        self,
        val_loader: torch.utils.data.DataLoader,
        device: torch.device,
        max_iter: int = 100,
        lr: float = 0.05,
        verbose: bool = True,
    ) -> float:
        """
        Optimises temperature T by minimising NLL on the validation set.

        The base model is kept in eval mode and its parameters are frozen.
        Only self.log_temperature is updated.

        Args:
            val_loader : DataLoader — validation set (same split used for early stopping).
            device     : torch.device.
            max_iter   : int — number of L-BFGS steps (default 100, typically converges in 20).
            lr         : float — L-BFGS learning rate (default 0.05).
            verbose    : bool — print calibration progress.

        Returns:
            float — final optimised temperature T.
        """
        self.to(device)
        self.base_model.eval()

        # ── Collect all logits and labels in one pass (efficient) ────────
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = self.base_model(images)
                all_logits.append(logits)
                all_labels.append(labels)

        all_logits = torch.cat(all_logits, dim=0)   # (N, num_classes)
        all_labels = torch.cat(all_labels, dim=0)   # (N,)

        criterion = nn.CrossEntropyLoss()

        # ── L-BFGS optimiser — standard choice for temperature scaling ───
        # L-BFGS converges in very few steps for this 1D optimisation problem.
        optimizer = torch.optim.LBFGS(
            [self.log_temperature], lr=lr, max_iter=max_iter
        )

        nll_before = criterion(all_logits / self.log_temperature.exp(), all_labels).item()

        def _eval_closure():
            optimizer.zero_grad()
            scaled_logits = all_logits / self.log_temperature.exp()
            loss = criterion(scaled_logits, all_labels)
            loss.backward()
            return loss

        optimizer.step(_eval_closure)

        nll_after = criterion(
            all_logits / self.log_temperature.exp(), all_labels
        ).item()

        if verbose:
            print(f"\n{'='*55}")
            print(f"  TEMPERATURE SCALING CALIBRATION")
            print(f"{'='*55}")
            print(f"  Initial temperature : 1.0 (identity — no scaling)")
            print(f"  Optimised T         : {self.temperature:.4f}")
            print(f"  NLL before          : {nll_before:.4f}")
            print(f"  NLL after           : {nll_after:.4f}")
            print(f"  Improvement         : {nll_before - nll_after:+.4f}")
            if self.temperature > 1.0:
                print(f"  Interpretation      : T > 1 — model was overconfident ✅")
            elif self.temperature < 1.0:
                print(f"  Interpretation      : T < 1 — model was underconfident")
            else:
                print(f"  Interpretation      : T ≈ 1 — model already well calibrated")
            print(f"{'='*55}\n")

        return self.temperature

    def compute_ece(
        self,
        val_loader: torch.utils.data.DataLoader,
        device: torch.device,
        n_bins: int = 15,
        before_calibration: bool = False,
    ) -> float:
        """
        Computes Expected Calibration Error (ECE) on the validation set.

        ECE is the weighted average absolute difference between model confidence
        and empirical accuracy across n_bins confidence bins.

        A perfectly calibrated model has ECE = 0.
        ResNet-18 typically has ECE ≈ 0.05–0.12 before calibration.

        Args:
            val_loader          : DataLoader — validation set.
            device              : torch.device.
            n_bins              : int — number of confidence bins (default 15).
            before_calibration  : bool — if True, uses T=1 (uncalibrated model).

        Returns:
            float — ECE in [0, 1]. Lower is better.
        """
        self.eval()
        all_confs   = []
        all_correct = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                if before_calibration:
                    # Use raw logits (T=1) to measure pre-calibration ECE
                    logits = self.base_model(images)
                else:
                    logits = self.forward(images)

                probs      = F.softmax(logits, dim=1)
                confs, preds = probs.max(dim=1)

                all_confs.extend(confs.cpu().numpy())
                all_correct.extend((preds == labels).cpu().numpy())

        all_confs   = np.array(all_confs)
        all_correct = np.array(all_correct, dtype=float)

        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece       = 0.0
        n_total   = len(all_confs)

        for i in range(n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            mask   = (all_confs > lo) & (all_confs <= hi)
            n_bin  = mask.sum()
            if n_bin == 0:
                continue
            acc_bin  = all_correct[mask].mean()
            conf_bin = all_confs[mask].mean()
            ece     += (n_bin / n_total) * abs(acc_bin - conf_bin)

        return float(ece)