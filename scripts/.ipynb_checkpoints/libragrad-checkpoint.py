"""
scripts/libragrad.py

LibraGrad saliency computation for CLIP vision and text encoders.

Responsibilities:
    - Compute LibraGrad vision saliency maps   (image patches)
    - Compute LibraGrad text saliency maps     (tokens)
    - Normalize and reshape outputs for visualization
    - Provide Option C standalone baseline maps

Pipeline per call:
    1. Install LibraGrad backward hooks        (LayerNorm + Attention)
    2. Install ActivationGradientRecorder      (at target layer)
    3. Forward pass through ClipWrapper
    4. Compute cosine similarity loss
    5. Backward pass                           (corrected gradients)
    6. Extract saliency = sum(act * grad, dim=-1)
    7. Remove all hooks
    8. Normalize and return

References:
    LibraGrad: https://github.com/NightMachinery/LibraGrad
    M2IB:      https://github.com/YingWANGG/M2IB
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional

from scripts.hooks import (
    HookManager,
    ActivationGradientRecorder,
    install_libragrad_hooks,
    remove_libragrad_hooks,
)
from scripts.utils import normalize


# ============================================================
# Core Saliency Extraction
# ============================================================

def _compute_saliency(
    model: nn.Module,
    image: torch.Tensor,
    text: torch.Tensor,
    target_layer: nn.Module,
    modality: str,
    use_libragrad: bool = True,
) -> torch.Tensor:
    """
    Core saliency extraction routine.

    Runs a forward + backward pass through ClipWrapper,
    optionally with LibraGrad hooks active, and extracts
    activation * gradient saliency at the target layer.

    Args:
        model:          ClipWrapper instance
        image:          [1, 3, 224, 224] preprocessed image tensor
        text:           [1, seq_len] tokenized text tensor
        target_layer:   ResidualAttentionBlock to record at
        modality:       "vision" or "text"
        use_libragrad:  If True, install LibraGrad backward hooks

    Returns:
        saliency: [seq_len] tensor (raw, not normalized)
    """

    assert modality in ("vision", "text"), \
        f"modality must be 'vision' or 'text', got {modality}"

    device = next(model.parameters()).device
    image = image.to(device)
    text = text.to(device)

    # --------------------------------------------------------
    # Step 1: Install LibraGrad backward hooks
    # --------------------------------------------------------
    libra_manager: Optional[HookManager] = None

    if use_libragrad:
        libra_manager = install_libragrad_hooks(model)

    # --------------------------------------------------------
    # Step 2: Install activation + gradient recorder
    # --------------------------------------------------------
    recorder = ActivationGradientRecorder(target_layer)
    recorder.install()

    # --------------------------------------------------------
    # Step 3 + 4 + 5: Forward, loss, backward
    # --------------------------------------------------------
    model.zero_grad()
    model.eval()

    # Forward
    image_features = model.get_image_features(image)
    text_features  = model.get_text_features(text)

    # Cosine similarity loss (same as M2IB fitting term)
    cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    loss = -cos(image_features, text_features).mean()

    # Backward (LibraGrad hooks active if installed)
    loss.backward()

    # --------------------------------------------------------
    # Step 6: Extract saliency
    # --------------------------------------------------------
    saliency = recorder.get_saliency()   # [seq_len]

    # --------------------------------------------------------
    # Step 7: Remove all hooks
    # --------------------------------------------------------
    recorder.remove()

    if libra_manager is not None:
        remove_libragrad_hooks(libra_manager)

    model.zero_grad()

    if saliency is None:
        raise RuntimeError(
            f"[libragrad] Saliency is None for modality={modality}. "
            "Check that target_layer is a direct child of the transformer."
        )

    return saliency


# ============================================================
# Vision Saliency
# ============================================================

def compute_libragrad_vision_saliency(
    model: nn.Module,
    image: torch.Tensor,
    text: torch.Tensor,
    target_layer: nn.Module,
    use_libragrad: bool = True,
    output_size: int = 224,
) -> np.ndarray:
    """
    Compute LibraGrad vision saliency map.

    Args:
        model:          ClipWrapper instance
        image:          [1, 3, 224, 224] preprocessed image tensor
        text:           [1, seq_len] tokenized text tensor
        target_layer:   ResidualAttentionBlock (e.g. resblocks[9])
        use_libragrad:  If True, use corrected backward (default)
        output_size:    Spatial size to upsample map to (default 224)

    Returns:
        saliency_map: np.ndarray of shape [output_size, output_size]
                      values in [0, 1]
    """

    saliency = _compute_saliency(
        model=model,
        image=image,
        text=text,
        target_layer=target_layer,
        modality="vision",
        use_libragrad=use_libragrad,
    )
    # saliency: [seq_len] = [50] for ViT-B/32 (49 patches + 1 CLS)

    # --------------------------------------------------------
    # Discard CLS token (index 0), keep patch tokens
    # --------------------------------------------------------
    #cls_val = saliency[0]
    #patch_saliency = saliency[1:] / (cls_val + 1e-6)
    patch_saliency = saliency[1:]
    grid = patch_saliency.reshape(1, 1, 7, 7).float()
    grid = torch.nn.functional.interpolate(
        grid, size=output_size, mode="bilinear", align_corners=False
    )[0, 0]
    saliency_map = grid.detach().cpu().numpy()
    return normalize(saliency_map)


# ============================================================
# Text Saliency
# ============================================================

def compute_libragrad_text_saliency(
    model: nn.Module,
    image: torch.Tensor,
    text: torch.Tensor,
    target_layer: nn.Module,
    use_libragrad: bool = True,
) -> np.ndarray:
    """
    Compute LibraGrad text saliency map.

    Args:
        model:          ClipWrapper instance
        image:          [1, 3, 224, 224] preprocessed image tensor
        text:           [1, seq_len] tokenized text tensor
        target_layer:   ResidualAttentionBlock (e.g. resblocks[9])
        use_libragrad:  If True, use corrected backward (default)

    Returns:
        saliency_map: np.ndarray of shape [seq_len]
                      values in [0, 1]
                      includes SOT and EOT tokens
    """

    saliency = _compute_saliency(
        model=model,
        image=image,
        text=text,
        target_layer=target_layer,
        modality="text",
        use_libragrad=use_libragrad,
    )
    # saliency: [seq_len]

    saliency_map = saliency.cpu().detach().numpy()

    return normalize(saliency_map)


# ============================================================
# Option C: Fusion Map
#
# Combines M2IB IBA map and LibraGrad map
# via element-wise product then renormalize.
# Used as a post-hoc baseline in compare_methods.py.
# ============================================================

def compute_fusion_map(
    iba_map: np.ndarray,
    libragrad_map: np.ndarray,
) -> np.ndarray:
    """
    Option C: element-wise product fusion of IBA and LibraGrad maps.

    Both maps must be normalized to [0, 1] before calling.

    Args:
        iba_map:       np.ndarray, M2IB output map
        libragrad_map: np.ndarray, LibraGrad output map

    Returns:
        fused_map: np.ndarray normalized to [0, 1]
    """

    assert iba_map.shape == libragrad_map.shape, (
        f"Shape mismatch: iba_map={iba_map.shape}, "
        f"libragrad_map={libragrad_map.shape}"
    )

    fused = iba_map * libragrad_map

    return normalize(fused)


# ============================================================
# Prior Preparation for LC-IBA (Option B)
#
# Converts a 2D saliency map (vision) or 1D map (text)
# into the [1, seq_len, hidden_dim] prior tensor
# expected by InformationBottleneck.libragrad_initialize_alpha
# ============================================================

def prepare_vision_prior(
    saliency_map: np.ndarray,
    hidden_dim: int = 768,
    grid_size: int = 7,
) -> torch.Tensor:
    """
    Convert a [224, 224] vision saliency map into a
    [1, num_patches+1, hidden_dim] prior tensor for LC-IBA.

    Pipeline:
        [224, 224] -> downsample -> [7, 7] -> flatten -> [49]
        -> prepend CLS=1.0 -> [50]
        -> expand -> [1, 50, hidden_dim]

    Args:
        saliency_map: np.ndarray [224, 224], values in [0, 1]
        hidden_dim:   CLIP hidden dimension (768 for ViT-B/32)
        grid_size:    Patch grid size (7 for ViT-B/32 with 32px patches)

    Returns:
        prior: torch.Tensor [1, num_patches+1, hidden_dim]
    """

    # [224, 224] -> tensor -> [1, 1, 224, 224]
    t = torch.tensor(saliency_map).float().unsqueeze(0).unsqueeze(0)

    # Downsample to patch grid
    t = torch.nn.functional.interpolate(
        t,
        size=(grid_size, grid_size),
        mode="bilinear",
        align_corners=False,
    )

    # [1, 1, 7, 7] -> [49]
    patch_prior = t[0, 0].reshape(-1)

    # Prepend CLS token prior = 1.0 (always preserve)
    cls_prior   = torch.ones(1)
    token_prior = torch.cat([cls_prior, patch_prior])   # [50]

    # Expand to hidden dim: [50] -> [50, hidden_dim]
    token_prior = token_prior.unsqueeze(-1).expand(-1, hidden_dim)

    # Add batch dim: [1, 50, hidden_dim]
    return token_prior.unsqueeze(0)


def prepare_text_prior(
    saliency_map: np.ndarray,
    hidden_dim: int = 512,
) -> torch.Tensor:
    """
    Convert a [seq_len] text saliency map into a
    [1, seq_len, hidden_dim] prior tensor for LC-IBA.

    Args:
        saliency_map: np.ndarray [seq_len], values in [0, 1]
        hidden_dim:   CLIP text hidden dimension (512 for ViT-B/32)

    Returns:
        prior: torch.Tensor [1, seq_len, hidden_dim]
    """

    t = torch.tensor(saliency_map).float()   # [seq_len]

    # Expand to hidden dim
    t = t.unsqueeze(-1).expand(-1, hidden_dim)   # [seq_len, hidden_dim]

    # Add batch dim
    return t.unsqueeze(0)                         # [1, seq_len, hidden_dim]


# ============================================================
# Convenience: Both modalities in one call
# ============================================================

def compute_libragrad_both(
    model: nn.Module,
    image: torch.Tensor,
    text: torch.Tensor,
    vision_layer: nn.Module,
    text_layer: nn.Module,
    use_libragrad: bool = True,
    output_size: int = 224,
) -> dict:
    """
    Compute LibraGrad saliency for both vision and text
    in a single call.

    Args:
        model:          ClipWrapper instance
        image:          [1, 3, 224, 224]
        text:           [1, seq_len]
        vision_layer:   Target ResidualAttentionBlock for vision
        text_layer:     Target ResidualAttentionBlock for text
        use_libragrad:  Whether to use corrected backward
        output_size:    Vision map spatial size

    Returns:
        dict with keys:
            'vision_map':   np.ndarray [output_size, output_size]
            'text_map':     np.ndarray [seq_len]
            'vision_prior': torch.Tensor [1, 50, 768]
            'text_prior':   torch.Tensor [1, seq_len, 512]
    """

    vision_map = compute_libragrad_vision_saliency(
        model=model,
        image=image,
        text=text,
        target_layer=vision_layer,
        use_libragrad=use_libragrad,
        output_size=output_size,
    )

    text_map = compute_libragrad_text_saliency(
        model=model,
        image=image,
        text=text,
        target_layer=text_layer,
        use_libragrad=use_libragrad,
    )

    vision_prior = prepare_vision_prior(vision_map)
    text_prior   = prepare_text_prior(text_map)

    return {
        "vision_map":   vision_map,
        "text_map":     text_map,
        "vision_prior": vision_prior,
        "text_prior":   text_prior,
    }