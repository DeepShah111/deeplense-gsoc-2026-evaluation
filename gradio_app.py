"""
gradio_app.py — DeepLense GSoC 2026 Interactive Demo
=====================================================
HuggingFace Spaces-ready Gradio application for gravitational lensing
dark matter morphology classification.

Features:
    • Upload any lensing image or pick from 3 sample images per class (9 total)
    • Dropdown: choose model — Baseline / Transfer / Ensemble / Equivariant
    • Classification result with probability bars for No Sub / CDM / Vortex
    • GradCAM heatmap overlay — "where is the model looking?"
    • TTA analysis: predictions at 0° / 90° / 180° / 270° to visually prove
      equivariant stability vs ensemble instability

Deploy:
    1. Push this file to a HuggingFace Space (Gradio SDK).
    2. Place model weights in weights/ directory of the Space repo.
    3. Place sample images in demo_samples/{class_name}/*.npy or *.png.
    4. Add requirements.txt with: torch torchvision gradio numpy pillow opencv-python

Local run:
    python gradio_app.py

Environment variables (for HF Spaces):
    MODEL_WEIGHTS_DIR : path to weights/ directory (default: "weights")
    SAMPLES_DIR       : path to sample images (default: "demo_samples")
"""

from __future__ import annotations

import os
import sys
import warnings
import math
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

# ── Gradio ────────────────────────────────────────────────────────────────────
try:
    import gradio as gr
except ImportError:
    raise ImportError("pip install gradio>=4.0.0")

# ── OpenCV for GradCAM resize ─────────────────────────────────────────────────
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    warnings.warn("opencv-python not installed — GradCAM overlay disabled.")

# ── src/ module imports ────────────────────────────────────────────────────────
# Support both: python gradio_app.py (from project root) and HF Spaces layout
_SCRIPT_DIR = Path(__file__).parent.resolve()
for candidate in [_SCRIPT_DIR / 'src', _SCRIPT_DIR]:
    if (candidate / 'models.py').exists():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        break

from models import ResNetBaseline, ResNetTransfer, ViTChampion, DeepLenseEnsemble, load_model
from metrics import compute_gradcam, overlay_gradcam

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

WEIGHTS_DIR  = os.environ.get('MODEL_WEIGHTS_DIR', str(_SCRIPT_DIR / 'weights'))
SAMPLES_DIR  = os.environ.get('SAMPLES_DIR',       str(_SCRIPT_DIR / 'demo_samples'))
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CLASS_NAMES  = ['No Sub', 'CDM', 'Vortex']
CLASS_COLORS = {'No Sub': '#2196F3', 'CDM': '#F44336', 'Vortex': '#4CAF50'}

CLASS_DESCRIPTIONS = {
    'No Sub':  'Smooth lens — no dark matter substructure. Featureless convergence map.',
    'CDM':     'Cold Dark Matter — localised point-mass subhalos producing small-scale perturbations.',
    'Vortex':  'Quantum condensate — extended vortex filaments from ultra-light axion dark matter.',
}

# ─────────────────────────────────────────────────────────────────────────────
# MODEL REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

MODEL_CONFIG = {
    'Transfer (ResNet-18)': {
        'mode':        'RGB',
        'image_size':  224,
        'weights_key': 'transfer_best.pth',
        'description': 'ImageNet-pretrained ResNet-18. Best absolute accuracy (89.3%). '
                       'Uses GradCAM on layer4 for visualisation.',
    },
    'Baseline (ResNet-18)': {
        'mode':        'L',
        'image_size':  64,
        'weights_key': 'baseline_best.pth',
        'description': 'ResNet-18 trained from scratch on 64×64 grayscale. Lower bound (60.4%). '
                       'Uses GradCAM on layer4.',
    },
    'Ensemble (ResNet + ViT)': {
        'mode':        'RGB',
        'image_size':  224,
        'weights_key': ['transfer_best.pth', 'vit_best.pth'],
        'description': 'Stacking meta-learner fusing ResNet-18 + ViT-B/16 (84.0%). '
                       'Uses Attention Rollout for ViT sub-model visualisation. '
                       'TTA shows 6.2% drop — proves orientation bias.',
    },
    'Equivariant (C8)': {
        'mode':        'L',
        'image_size':  128,
        'weights_key': 'equivariant_best.pth',
        'description': 'C8-equivariant CNN (escnn). Only 0.4% TTA drop — '
                       'rotational symmetry baked into architecture. '
                       'GradCAM hooks group_pool for equivariant-safe visualisation.',
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# MODEL CACHE — load once, reuse across calls
# ─────────────────────────────────────────────────────────────────────────────

_model_cache: dict[str, torch.nn.Module] = {}


def _load_model_cached(model_name: str) -> Optional[torch.nn.Module]:
    """Load and cache a model. Returns None if weights are not found."""
    if model_name in _model_cache:
        return _model_cache[model_name]

    cfg = MODEL_CONFIG[model_name]

    try:
        if model_name == 'Baseline (ResNet-18)':
            m = ResNetBaseline(num_classes=3)
            w = os.path.join(WEIGHTS_DIR, cfg['weights_key'])
            if not os.path.exists(w):
                return None
            m = load_model(m, w, DEVICE)

        elif model_name == 'Transfer (ResNet-18)':
            m = ResNetTransfer(num_classes=3)
            w = os.path.join(WEIGHTS_DIR, cfg['weights_key'])
            if not os.path.exists(w):
                return None
            m = load_model(m, w, DEVICE)

        elif model_name == 'Ensemble (ResNet + ViT)':
            resnet_w, vit_w = [os.path.join(WEIGHTS_DIR, k) for k in cfg['weights_key']]
            if not (os.path.exists(resnet_w) and os.path.exists(vit_w)):
                return None
            resnet = load_model(ResNetTransfer(num_classes=3), resnet_w, DEVICE)
            vit    = load_model(ViTChampion(num_classes=3),    vit_w,    DEVICE)
            m = DeepLenseEnsemble(resnet_model=resnet, vit_model=vit, freeze_base=True)
            m = m.to(DEVICE)
            m.eval()

        elif model_name == 'Equivariant (C8)':
            try:
                from models import EquivariantCNN
                m = EquivariantCNN(num_classes=3, n_rotations=8)
            except ImportError:
                return None
            w = os.path.join(WEIGHTS_DIR, cfg['weights_key'])
            if not os.path.exists(w):
                return None
            m = load_model(m, w, DEVICE)

        else:
            return None

        _model_cache[model_name] = m
        return m

    except Exception as e:
        print(f"⚠️  Failed to load {model_name}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# TRANSFORM UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _get_transform(mode: str, image_size: int) -> transforms.Compose:
    if mode == 'RGB':
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ])
    else:  # Grayscale
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])


def _pil_to_tensor(pil_img: Image.Image, mode: str, image_size: int) -> torch.Tensor:
    """Converts a PIL Image to a normalised (1, C, H, W) tensor."""
    transform = _get_transform(mode, image_size)
    if pil_img.mode == 'RGBA':
        pil_img = pil_img.convert('RGB')
    tensor = transform(pil_img)
    return tensor.unsqueeze(0)


def _tensor_to_numpy_displayable(tensor: torch.Tensor, mode: str) -> np.ndarray:
    """Converts a (1, C, H, W) or (C, H, W) normalised tensor to (H, W, 3) uint8."""
    t = tensor.squeeze(0).cpu().numpy()
    if mode == 'RGB':
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        t    = t.transpose(1, 2, 0) * std + mean
        t    = np.clip(t, 0, 1)
        return (t * 255).astype(np.uint8)
    else:
        t = t[0] * 0.5 + 0.5
        t = np.clip(t, 0, 1)
        t = (t * 255).astype(np.uint8)
        return np.stack([t, t, t], axis=-1)  # Grayscale → RGB for display


# ─────────────────────────────────────────────────────────────────────────────
# SAMPLE IMAGE LOADER
# ─────────────────────────────────────────────────────────────────────────────

def _load_sample_images() -> dict[str, list[str]]:
    """
    Loads sample image paths from SAMPLES_DIR.
    Directory structure expected:
        demo_samples/
            no_sub/   *.png or *.jpg
            cdm/      *.png or *.jpg
            vortex/   *.png or *.jpg
    Returns dict: {class_name: [path1, path2, path3]}
    """
    samples = {}
    class_dir_map = {'No Sub': 'no_sub', 'CDM': 'cdm', 'Vortex': 'vortex'}

    for class_name, dir_name in class_dir_map.items():
        class_dir = os.path.join(SAMPLES_DIR, dir_name)
        if not os.path.exists(class_dir):
            samples[class_name] = []
            continue
        exts = {'.png', '.jpg', '.jpeg', '.npy'}
        paths = sorted([
            os.path.join(class_dir, f)
            for f in os.listdir(class_dir)
            if Path(f).suffix.lower() in exts
        ])[:3]
        samples[class_name] = paths

    return samples


SAMPLE_IMAGES = _load_sample_images()


def _path_to_pil(path: str) -> Image.Image:
    """Loads a PNG/JPG/NPY file as PIL Image (RGB or L)."""
    if path.endswith('.npy'):
        arr = np.load(path)
        if arr.ndim == 2:
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
            arr = (arr * 255).astype(np.uint8)
            return Image.fromarray(arr, mode='L')
        elif arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
            arr = (arr * 255).astype(np.uint8)
            return Image.fromarray(arr, mode='L')
        else:
            arr = arr.transpose(1, 2, 0) if arr.shape[0] == 3 else arr
            arr = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255).astype(np.uint8)
            return Image.fromarray(arr)
    else:
        return Image.open(path).convert('RGB')


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def _run_inference(
    model:        torch.nn.Module,
    image_tensor: torch.Tensor,
) -> tuple[int, np.ndarray]:
    """Returns (pred_class_idx, probs_array)."""
    model.eval()
    with torch.no_grad():
        logits = model(image_tensor.to(DEVICE))
        probs  = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    return int(np.argmax(probs)), probs


def _run_tta(
    model:        torch.nn.Module,
    image_tensor: torch.Tensor,
    angles:       list[int] = [0, 90, 180, 270],
) -> dict[int, tuple[int, np.ndarray]]:
    """
    Runs inference at each TTA angle independently (NOT averaged).
    Returns dict: {angle: (pred_class_idx, probs_array)}
    This is intentionally per-angle (not averaged TTA) so the demo can
    show how each rotation changes the prediction — revealing orientation bias.
    """
    model.eval()
    results = {}

    k_map = {0: 0, 90: 1, 180: 2, 270: 3}

    with torch.no_grad():
        for angle in angles:
            k = k_map.get(angle % 360, 0)
            rotated = torch.rot90(image_tensor, k=k, dims=[2, 3])
            logits  = model(rotated.to(DEVICE))
            probs   = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            results[angle] = (int(np.argmax(probs)), probs)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# GRADCAM OVERLAY BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def _build_gradcam_image(
    model:        torch.nn.Module,
    image_tensor: torch.Tensor,
    pred_class:   int,
    mode:         str,
) -> Optional[np.ndarray]:
    """
    Returns a (H, W, 3) uint8 numpy array with the GradCAM overlay,
    or None if GradCAM is unavailable (missing opencv or escnn edge cases).
    """
    if not OPENCV_AVAILABLE:
        return None

    try:
        cam, _, _ = compute_gradcam(
            model        = model,
            image_tensor = image_tensor,
            class_idx    = pred_class,
            device       = DEVICE,
        )
        img_np  = _tensor_to_numpy_displayable(image_tensor, mode).astype(float) / 255.0
        overlay = overlay_gradcam(img_np, cam, alpha=0.45)
        return (overlay * 255).astype(np.uint8)
    except Exception as e:
        print(f"⚠️  GradCAM failed: {e}\n{traceback.format_exc()}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# MAIN GRADIO CALLBACK
# ─────────────────────────────────────────────────────────────────────────────

def classify_image(
    uploaded_image,       # PIL Image from gr.Image
    model_name: str,
    sample_class: str,
    sample_index: int,
) -> tuple:
    """
    Main inference callback. Returns:
        (result_html, probabilities_dict, gradcam_image, tta_html, model_info_text)
    """
    # ── Resolve input image ───────────────────────────────────────────────
    pil_img = None

    if uploaded_image is not None:
        if isinstance(uploaded_image, np.ndarray):
            pil_img = Image.fromarray(uploaded_image)
        elif isinstance(uploaded_image, Image.Image):
            pil_img = uploaded_image
        else:
            pil_img = Image.fromarray(uploaded_image)

    elif sample_class and SAMPLE_IMAGES.get(sample_class):
        idx = min(int(sample_index), len(SAMPLE_IMAGES[sample_class]) - 1)
        pil_img = _path_to_pil(SAMPLE_IMAGES[sample_class][idx])

    # Always convert to RGB at this stage — model transforms handle
    # grayscale conversion internally if mode='L'. Starting from RGB
    # guarantees the image is never a flat 1-channel blob going in.
    if pil_img is not None and pil_img.mode != 'RGB':
        pil_img = pil_img.convert('RGB')

    if pil_img is None:
        return (
            "<p style='color:red'>Please upload an image or select a sample.</p>",
            {cls: 0.0 for cls in CLASS_NAMES},
            None,
            "<p>No image provided.</p>",
            MODEL_CONFIG.get(model_name, {}).get('description', ''),
        )

    # ── Load model ────────────────────────────────────────────────────────
    model = _load_model_cached(model_name)

    if model is None:
        weights_info = MODEL_CONFIG[model_name]['weights_key']
        missing = weights_info if isinstance(weights_info, str) else ', '.join(weights_info)
        return (
            f"<p style='color:orange'>⚠️ Model weights not found: <code>{missing}</code>. "
            f"Place .pth files in <code>{WEIGHTS_DIR}/</code></p>",
            {cls: 0.0 for cls in CLASS_NAMES},
            None,
            "<p>Model not loaded.</p>",
            MODEL_CONFIG[model_name]['description'],
        )

    cfg        = MODEL_CONFIG[model_name]
    mode       = cfg['mode']
    image_size = cfg['image_size']

    # ── Preprocess ────────────────────────────────────────────────────────
    try:
        tensor = _pil_to_tensor(pil_img, mode, image_size)   # (1, C, H, W)
    except Exception as e:
        return (
            f"<p style='color:red'>Image preprocessing failed: {e}</p>",
            {cls: 0.0 for cls in CLASS_NAMES},
            None,
            "",
            cfg['description'],
        )

    # ── Standard inference ────────────────────────────────────────────────
    pred_class, probs = _run_inference(model, tensor)
    pred_name         = CLASS_NAMES[pred_class]

    # ── Build result HTML ─────────────────────────────────────────────────
    color = CLASS_COLORS[pred_name]
    result_html = f"""
    <div style="border-left: 6px solid {color}; padding: 12px; margin: 8px 0;
                background: {color}18; border-radius: 4px;">
        <h2 style="margin: 0; color: {color};">🔭 Predicted: {pred_name}</h2>
        <p style="margin: 6px 0 0; color: #555; font-size: 14px;">
            {CLASS_DESCRIPTIONS[pred_name]}
        </p>
        <p style="margin: 8px 0 0; font-size: 13px; color: #333;">
            Confidence: <strong>{probs[pred_class]*100:.1f}%</strong>
        </p>
    </div>
    """

    probs_dict = {CLASS_NAMES[i]: float(probs[i]) for i in range(3)}

    # ── GradCAM ───────────────────────────────────────────────────────────
    gradcam_img = _build_gradcam_image(model, tensor, pred_class, mode)

    # ── TTA Analysis ─────────────────────────────────────────────────────
    tta_results = _run_tta(model, tensor)

    angle_rows = ""
    predictions_match = True
    first_pred = tta_results[0][0]

    for angle, (p_cls, p_probs) in tta_results.items():
        p_name = CLASS_NAMES[p_cls]
        p_color = CLASS_COLORS[p_name]
        conf   = p_probs[p_cls] * 100
        match  = "✅" if p_cls == first_pred else "⚠️ Changed"
        if p_cls != first_pred:
            predictions_match = False

        bar_widths = [f"{p_probs[i]*100:.0f}%" for i in range(3)]
        mini_bars  = "".join([
            f'<span style="display:inline-block; width:{bar_widths[i]}; height:8px; '
            f'background:{list(CLASS_COLORS.values())[i]}; margin-right:2px; border-radius:2px;"></span>'
            for i in range(3)
        ])

        angle_rows += f"""
        <tr>
            <td style="padding:6px 12px; font-weight:bold;">{angle}°</td>
            <td style="padding:6px 12px; color:{p_color}; font-weight:bold;">{p_name}</td>
            <td style="padding:6px 12px;">{conf:.1f}%</td>
            <td style="padding:6px 12px;">{mini_bars}</td>
            <td style="padding:6px 12px;">{match}</td>
        </tr>
        """

    stability_color = '#4CAF50' if predictions_match else '#F44336'
    stability_text  = (
        '✅ All rotations agree — Equivariantly stable' if predictions_match
        else '⚠️ Predictions change under rotation — Orientation bias detected'
    )

    # Which model visualisation method is being used
    if 'ViT' in model_name or 'Ensemble' in model_name:
        vis_note = '(Attention Rollout — ViT sub-model)'
    elif 'Equivariant' in model_name:
        vis_note = '(GradCAM via group_pool hook — equivariant-safe)'
    else:
        vis_note = '(GradCAM via layer4)'

    tta_html = f"""
    <div style="margin-top:12px;">
        <h3 style="color:#333;">🔄 TTA Rotational Analysis {vis_note}</h3>
        <p style="font-size:13px; color:#555;">
            A physically correct model (lensing has no preferred orientation) should 
            give identical predictions at any rotation. This table shows whether the 
            model is equivariantly stable or orientation-biased.
        </p>
        <table style="border-collapse:collapse; width:100%; font-size:14px;">
            <thead>
                <tr style="background:#f5f5f5;">
                    <th style="padding:6px 12px; text-align:left;">Rotation</th>
                    <th style="padding:6px 12px; text-align:left;">Prediction</th>
                    <th style="padding:6px 12px; text-align:left;">Confidence</th>
                    <th style="padding:6px 12px; text-align:left;">Prob Distribution</th>
                    <th style="padding:6px 12px; text-align:left;">Stable?</th>
                </tr>
            </thead>
            <tbody>{angle_rows}</tbody>
        </table>
        <p style="margin-top:10px; font-weight:bold; color:{stability_color};">
            {stability_text}
        </p>
        <p style="font-size:12px; color:#888;">
            No Sub = <span style="color:#2196F3;">■</span>  
            CDM = <span style="color:#F44336;">■</span>  
            Vortex = <span style="color:#4CAF50;">■</span>
        </p>
    </div>
    """

    model_info = f"{cfg['description']}\n\nDevice: {DEVICE}"

    return result_html, probs_dict, gradcam_img, tta_html, model_info


# ─────────────────────────────────────────────────────────────────────────────
# SAMPLE IMAGE CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

def load_sample(sample_class: str, sample_index: int):
    """Returns PIL image for display when a sample is selected."""
    paths = SAMPLE_IMAGES.get(sample_class, [])
    if not paths:
        return None
    idx = min(int(sample_index), len(paths) - 1)
    try:
        pil = _path_to_pil(paths[idx])
        return pil.convert('RGB')
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# GRADIO INTERFACE
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# GRADIO INTERFACE — Clean light-grey theme, uncluttered layout
# ─────────────────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
/* ── Page background ───────────────────────────────────────── */
body, .gradio-container {
    background-color: #F2F3F5 !important;
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
}

/* ── Cards ─────────────────────────────────────────────────── */
.card {
    background: #FFFFFF;
    border: 1px solid #DDE1E7;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}

/* ── Section labels ─────────────────────────────────────────── */
.section-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #6B7280;
    margin-bottom: 10px;
}

/* ── Primary button ─────────────────────────────────────────── */
.classify-btn {
    background: #2D3748 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    padding: 12px 0 !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: background 0.2s !important;
}
.classify-btn:hover { background: #1A202C !important; }

/* ── Secondary button ───────────────────────────────────────── */
.load-btn {
    background: #EDF2F7 !important;
    color: #2D3748 !important;
    border: 1px solid #CBD5E0 !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    width: 100% !important;
    transition: background 0.2s !important;
}
.load-btn:hover { background: #E2E8F0 !important; }

/* ── Divider ────────────────────────────────────────────────── */
.divider {
    border: none;
    border-top: 1px solid #E2E8F0;
    margin: 16px 0;
}

/* ── Hide Gradio default label borders ──────────────────────── */
.gr-box { border: none !important; box-shadow: none !important; }
label.svelte-1b6s6s { color: #4A5568 !important; font-size: 13px !important; }

/* ── Force dark text everywhere inputs/dropdowns/textboxes ──── */
input, textarea, select,
.gr-input, .gr-textbox, .gr-dropdown,
span[data-testid], .wrap span, .secondary-wrap span,
[class*="svelte"] span, [class*="svelte"] input,
.options span, .item span,
.block label span, .block span,
.block p, .block textarea,
.svelte-phx9e8, .svelte-1gfkn6j,
.token, .value {
    color: #2D3748 !important;
}

/* Dropdown option items */
ul.options li, .dropdown-arrow, .placeholder {
    color: #2D3748 !important;
    background: #FFFFFF !important;
}

/* Selected value in dropdown */
.single-select, .multiselect {
    color: #2D3748 !important;
}

/* Textbox content */
textarea, input[type="text"], input[type="number"] {
    color: #2D3748 !important;
    background: #FFFFFF !important;
}

/* ── Max width ──────────────────────────────────────────────── */
.gradio-container { max-width: 1280px !important; margin: 0 auto !important; }
"""

HEADER_HTML = """
<div style="background:#FFFFFF; border:1px solid #DDE1E7; border-radius:12px;
            padding:24px 28px; margin-bottom:20px; box-shadow:0 1px 4px rgba(0,0,0,0.06);">
    <div style="display:flex; align-items:center; gap:12px; margin-bottom:8px;">
        <span style="font-size:28px;">🔭</span>
        <div>
            <h1 style="margin:0; font-size:22px; font-weight:700; color:#1A202C;">
                DeepLense — Dark Matter Morphology Classifier
            </h1>
            <p style="margin:2px 0 0; font-size:13px; color:#718096;">
                GSoC 2026 &nbsp;·&nbsp; E(2)-Equivariant Neural Networks for Gravitational Lensing
            </p>
        </div>
    </div>
    <div style="display:flex; gap:8px; flex-wrap:wrap; margin-top:14px;">
        <span style="background:#EBF8FF; color:#2B6CB0; border:1px solid #BEE3F8;
                     border-radius:20px; padding:3px 12px; font-size:12px; font-weight:600;">
            No Sub — Smooth lens
        </span>
        <span style="background:#FFF5F5; color:#C53030; border:1px solid #FED7D7;
                     border-radius:20px; padding:3px 12px; font-size:12px; font-weight:600;">
            CDM — Cold Dark Matter
        </span>
        <span style="background:#F0FFF4; color:#276749; border:1px solid #C6F6D5;
                     border-radius:20px; padding:3px 12px; font-size:12px; font-weight:600;">
            Vortex — Ultra-light axion
        </span>
        <span style="background:#F7FAFC; color:#4A5568; border:1px solid #E2E8F0;
                     border-radius:20px; padding:3px 12px; font-size:12px;">
            ⚡ C8 equivariant model: <strong>0.4%</strong> TTA drop vs <strong>6.2%</strong> ensemble
        </span>
    </div>
</div>
"""

FOOTER_HTML = """
<div style="background:#FFFFFF; border:1px solid #DDE1E7; border-radius:12px;
            padding:16px 24px; margin-top:8px; font-size:12px; color:#718096;
            box-shadow:0 1px 4px rgba(0,0,0,0.04);">
    <strong style="color:#4A5568;">GradCAM notes:</strong>
    ResNet → layer4 hook &nbsp;·&nbsp;
    ViT → Attention Rollout &nbsp;·&nbsp;
    EquivariantCNN → input-gradient saliency (escnn graph constraint) &nbsp;·&nbsp;
    <a href="https://github.com/ML4SCI/DeepLense" style="color:#4299E1;">GitHub</a> &nbsp;·&nbsp;
    <a href="https://arxiv.org/abs/1911.08251" style="color:#4299E1;">Weiler & Cesa 2019</a>
</div>
"""


def build_demo() -> gr.Blocks:
    with gr.Blocks(
        title="DeepLense Dark Matter Classifier",
        theme=gr.themes.Default(
            primary_hue=gr.themes.colors.slate,
            secondary_hue=gr.themes.colors.gray,
            neutral_hue=gr.themes.colors.gray,
            font=gr.themes.GoogleFont("Inter"),
        ).set(
            body_background_fill="#F2F3F5",
            body_background_fill_dark="#F2F3F5",
            body_text_color="#2D3748",
            body_text_color_dark="#2D3748",
            body_text_color_subdued="#718096",
            body_text_color_subdued_dark="#718096",
            block_background_fill="#FFFFFF",
            block_background_fill_dark="#FFFFFF",
            block_border_color="#DDE1E7",
            block_border_color_dark="#DDE1E7",
            block_label_text_color="#4A5568",
            block_label_text_color_dark="#4A5568",
            input_background_fill="#FFFFFF",
            input_background_fill_dark="#FFFFFF",
            input_border_color="#CBD5E0",
            input_border_color_dark="#CBD5E0",
            input_placeholder_color="#A0AEC0",
            input_placeholder_color_dark="#A0AEC0",
            button_primary_background_fill="#2D3748",
            button_primary_background_fill_dark="#2D3748",
            button_primary_background_fill_hover="#1A202C",
            button_primary_text_color="#FFFFFF",
            button_secondary_background_fill="#EDF2F7",
            button_secondary_background_fill_dark="#EDF2F7",
            button_secondary_text_color="#2D3748",
            button_secondary_text_color_dark="#2D3748",
        ),
        css=CUSTOM_CSS,
    ) as demo:

        # ── Header ────────────────────────────────────────────────────────
        gr.HTML(HEADER_HTML)

        with gr.Row(equal_height=False):

            # ══ LEFT PANEL — Input controls (narrower) ════════════════════
            with gr.Column(scale=4, min_width=320):

                # ── Image input card ──────────────────────────────────────
                gr.HTML('<div class="section-label">Input Image</div>')
                uploaded = gr.Image(
                    type='pil',
                    label='',
                    height=220,
                    show_label=False,
                )

                gr.HTML('<hr class="divider"><div class="section-label">Or use a sample</div>')

                with gr.Row():
                    sample_class = gr.Dropdown(
                        choices=list(CLASS_NAMES),
                        value='CDM',
                        label='Class',
                        scale=2,
                    )
                    sample_idx = gr.Slider(
                        minimum=0, maximum=2, step=1, value=0,
                        label='Index',
                        scale=1,
                    )

                load_btn = gr.Button("Load Sample", elem_classes=['load-btn'], size='sm')

                sample_preview = gr.Image(
                    label='',
                    height=160,
                    interactive=False,
                    show_label=False,
                )

                gr.HTML('<hr class="divider"><div class="section-label">Model</div>')

                model_dropdown = gr.Dropdown(
                    choices=list(MODEL_CONFIG.keys()),
                    value='Transfer (ResNet-18)',
                    label='',
                    show_label=False,
                )

                model_info_box = gr.Textbox(
                    label='',
                    value=MODEL_CONFIG['Transfer (ResNet-18)']['description'],
                    lines=2,
                    interactive=False,
                    show_label=False,
                )

                gr.HTML('<div style="height:8px;"></div>')
                classify_btn = gr.Button(
                    "🔍  Classify",
                    elem_classes=['classify-btn'],
                    size='lg',
                )

            # ══ RIGHT PANEL — Results (wider) ═════════════════════════════
            with gr.Column(scale=8, min_width=500):

                # ── Row 1: Result card + Probabilities side by side ───────
                with gr.Row(equal_height=True):

                    with gr.Column(scale=5):
                        gr.HTML('<div class="section-label">Prediction</div>')
                        result_html = gr.HTML(
                            value="""
                            <div style="background:#F7FAFC; border:1px solid #E2E8F0;
                                        border-radius:10px; padding:20px; text-align:center;
                                        color:#A0AEC0; font-size:14px; min-height:90px;
                                        display:flex; align-items:center; justify-content:center;">
                                Run classification to see result
                            </div>"""
                        )

                    with gr.Column(scale=5):
                        gr.HTML('<div class="section-label">Class Probabilities</div>')
                        probs_bar = gr.Label(
                            label='',
                            num_top_classes=3,
                            show_label=False,
                        )

                # ── Row 2: GradCAM + TTA side by side ────────────────────
                with gr.Row(equal_height=False):

                    with gr.Column(scale=5):
                        gr.HTML('<div class="section-label">GradCAM — Where is the model looking?</div>')
                        gradcam_out = gr.Image(
                            label='',
                            height=300,
                            interactive=False,
                            show_label=False,
                        )

                    with gr.Column(scale=5):
                        gr.HTML('<div class="section-label">Rotational Stability (TTA)</div>')
                        tta_html = gr.HTML(
                            value="""
                            <div style="background:#F7FAFC; border:1px solid #E2E8F0;
                                        border-radius:10px; padding:20px; color:#A0AEC0;
                                        font-size:13px; min-height:280px;
                                        display:flex; align-items:center; justify-content:center;">
                                Run classification to see TTA analysis
                            </div>"""
                        )

        # ── Footer ────────────────────────────────────────────────────────
        gr.HTML(FOOTER_HTML)

        # ── Callbacks (identical to before — no logic changes) ────────────
        def update_model_info(name):
            return MODEL_CONFIG[name]['description']

        model_dropdown.change(
            fn=update_model_info,
            inputs=[model_dropdown],
            outputs=[model_info_box],
        )

        load_btn.click(
            fn=load_sample,
            inputs=[sample_class, sample_idx],
            outputs=[sample_preview],
        )

        classify_btn.click(
            fn=classify_image,
            inputs=[uploaded, model_dropdown, sample_class, sample_idx],
            outputs=[result_html, probs_bar, gradcam_out, tta_html, model_info_box],
        )

        sample_class.change(
            fn=load_sample,
            inputs=[sample_class, sample_idx],
            outputs=[sample_preview],
        )

        sample_idx.change(
            fn=load_sample,
            inputs=[sample_class, sample_idx],
            outputs=[sample_preview],
        )

    return demo


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    demo = build_demo()
    demo.launch(
        server_name='0.0.0.0',
        server_port=7860,
        share=False,
        show_error=True,
    )