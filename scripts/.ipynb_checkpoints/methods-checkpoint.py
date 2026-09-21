"""
scripts/methods.py

Heatmap generation methods for all four experimental variants.

Based on:
    https://github.com/bazingagin/IBA
    https://github.com/YingWANGG/M2IB

Methods provided:
    1. M2IB baseline (original):
        vision_heatmap_iba()
        text_heatmap_iba()

    2. LC-IBA Option A+B (main contribution):
        vision_heatmap_lciba()
        text_heatmap_lciba()

    3. LibraGrad only Option C standalone:
        vision_heatmap_libragrad()
        text_heatmap_libragrad()

    4. Fusion Option C combined:
        vision_heatmap_fusion()
        text_heatmap_fusion()

All methods share the same signature pattern:
    heatmap_fn(text_t, image_t, model, layer_idx, ...)
    -> np.ndarray

This keeps experiments/compare_methods.py clean and uniform.
"""

import numpy as np
import torch

from scripts.iba import IBAInterpreter, Estimator
from scripts.libragrad import (
    compute_libragrad_vision_saliency,
    compute_libragrad_text_saliency,
    compute_fusion_map,
    prepare_vision_prior,
    prepare_text_prior,
    compute_libragrad_both,
)
from scripts.utils import normalize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Shared Utilities
# ============================================================

def extract_feature_map(model, layer_idx, x):
    """
    Extract hidden state at layer_idx from model.

    Args:
        model:     vision_model or text_model from ClipWrapper
        layer_idx: transformer block index (0-based)
        x:         input tensor

    Returns:
        feature: [1, seq_len, hidden_dim] tensor
    """
    with torch.no_grad():
        states  = model(x, output_hidden_states=True)
        # +1 because index 0 is embedding output
        feature = states["hidden_states"][layer_idx + 1]
    return feature


def extract_bert_layer(model, layer_idx):
    """
    Extract the ResidualAttentionBlock at layer_idx.

    Args:
        model:     vision_model or text_model
        layer_idx: transformer block index (0-based)

    Returns:
        layer: nn.Module (ResidualAttentionBlock)
    """
    for _, submodule in model.named_children():
        for n, s in submodule.named_children():
            if n in ("layers", "resblocks"):
                for n2, s2 in s.named_children():
                    if n2 == str(layer_idx):
                        return s2
    raise ValueError(
        f"Could not find layer {layer_idx} in model. "
        "Check that layer_idx is within range."
    )


def get_compression_estimator(var, layer, features):
    """
    Build a simple Estimator with zero mean and fixed variance.

    Args:
        var:      Variance scalar for the IB prior
        layer:    Target layer (ResidualAttentionBlock)
        features: [1, seq_len, hidden_dim] feature tensor

    Returns:
        estimator: Estimator instance ready for IBAInterpreter
    """
    estimator   = Estimator(layer)
    estimator.M = torch.zeros_like(features)
    estimator.S = var * np.ones(features.shape)
    estimator.N = 1
    return estimator


# ============================================================
# Method 1: M2IB Baseline (original)
# ============================================================

def text_heatmap_iba(
    text_t,
    image_t,
    model,
    layer_idx,
    beta,
    var,
    lr=1.0,
    train_steps=10,
    progbar=True,
):
    """
    Original M2IB text attribution (no LibraGrad).

    Args:
        text_t:      [1, seq_len] tokenized text tensor
        image_t:     [1, 3, 224, 224] preprocessed image tensor
        model:       ClipWrapper instance
        layer_idx:   Target transformer block index
        beta:        IB compression weight
        var:         Prior variance
        lr:          Adam learning rate
        train_steps: Optimization steps
        progbar:     Show tqdm progress bar

    Returns:
        np.ndarray [seq_len] normalized to [0, 1]
    """
    features = extract_feature_map(
        model.text_model, layer_idx, text_t
    )
    layer    = extract_bert_layer(model.text_model, layer_idx)
    estimator = get_compression_estimator(var, layer, features)

    reader = IBAInterpreter(
        model=model,
        estim=estimator,
        beta=beta,
        lr=lr,
        steps=train_steps,
        progbar=progbar,
        use_libragrad=False,
        libragrad_init=False,
    )
    return reader.text_heatmap(text_t, image_t)


def vision_heatmap_iba(
    text_t,
    image_t,
    model,
    layer_idx,
    beta,
    var,
    lr=1.0,
    train_steps=10,
    progbar=True,
):
    """
    Original M2IB vision attribution (no LibraGrad).

    Args:
        text_t:      [1, seq_len] tokenized text tensor
        image_t:     [1, 3, 224, 224] preprocessed image tensor
        model:       ClipWrapper instance
        layer_idx:   Target transformer block index
        beta:        IB compression weight
        var:         Prior variance
        lr:          Adam learning rate
        train_steps: Optimization steps
        progbar:     Show tqdm progress bar

    Returns:
        np.ndarray [224, 224] normalized to [0, 1]
    """
    features  = extract_feature_map(
        model.vision_model, layer_idx, image_t
    )
    layer     = extract_bert_layer(model.vision_model, layer_idx)
    estimator = get_compression_estimator(var, layer, features)

    reader = IBAInterpreter(
        model=model,
        estim=estimator,
        beta=beta,
        lr=lr,
        steps=train_steps,
        progbar=progbar,
        use_libragrad=False,
        libragrad_init=False,
    )
    return reader.vision_heatmap(text_t, image_t)


# ============================================================
# Method 2: LC-IBA Option A+B (main contribution)
# ============================================================

def text_heatmap_lciba(
    text_t,
    image_t,
    model,
    layer_idx,
    beta,
    var,
    lr=1.0,
    train_steps=10,
    progbar=True,
    prior_strength=3.0,
):
    """
    LC-IBA text attribution.
    Option A: LibraGrad hooks active during IB optimization.
    Option B: alpha initialized from LibraGrad saliency prior.

    Args:
        text_t:         [1, seq_len] tokenized text tensor
        image_t:        [1, 3, 224, 224] preprocessed image tensor
        model:          ClipWrapper instance
        layer_idx:      Target transformer block index
        beta:           IB compression weight
        var:            Prior variance
        lr:             Adam learning rate
        train_steps:    Optimization steps
        progbar:        Show tqdm progress bar
        prior_strength: Logit scaling for alpha initialization

    Returns:
        np.ndarray [seq_len] normalized to [0, 1]
    """

    # ----------------------------------------------------------
    # Option B: Compute LibraGrad prior first
    # ----------------------------------------------------------
    target_layer = extract_bert_layer(model.text_model, layer_idx)

    text_map = compute_libragrad_text_saliency(
        model=model,
        image=image_t,
        text=text_t,
        target_layer=target_layer,
        use_libragrad=True,
    )

    # Determine hidden dim from feature map
    features  = extract_feature_map(
        model.text_model, layer_idx, text_t
    )
    hidden_dim = features.shape[-1]

    text_prior = prepare_text_prior(text_map, hidden_dim=hidden_dim)

    # ----------------------------------------------------------
    # Build estimator and interpreter
    # ----------------------------------------------------------
    estimator = get_compression_estimator(var, target_layer, features)

    reader = IBAInterpreter(
        model=model,
        estim=estimator,
        beta=beta,
        lr=lr,
        steps=train_steps,
        progbar=progbar,
        use_libragrad=False,    # Option A
        libragrad_init=True,   # Option B
    )

    # Pass prior into heatmap method
    reader.bottleneck.libragrad_initialize_alpha(
        text_prior, strength=prior_strength
    )

    return reader.text_heatmap(text_t, image_t, prior=text_prior)


def vision_heatmap_lciba(
    text_t,
    image_t,
    model,
    layer_idx,
    beta,
    var,
    lr=1.0,
    train_steps=10,
    progbar=True,
    prior_strength=3.0,
):
    """
    LC-IBA vision attribution.
    Option A: LibraGrad hooks active during IB optimization.
    Option B: alpha initialized from LibraGrad saliency prior.

    Args:
        text_t:         [1, seq_len] tokenized text tensor
        image_t:        [1, 3, 224, 224] preprocessed image tensor
        model:          ClipWrapper instance
        layer_idx:      Target transformer block index
        beta:           IB compression weight
        var:            Prior variance
        lr:             Adam learning rate
        train_steps:    Optimization steps
        progbar:        Show tqdm progress bar
        prior_strength: Logit scaling for alpha initialization

    Returns:
        np.ndarray [224, 224] normalized to [0, 1]
    """

    # ----------------------------------------------------------
    # Option B: Compute LibraGrad prior first
    # ----------------------------------------------------------
    target_layer = extract_bert_layer(model.vision_model, layer_idx)

    vision_map = compute_libragrad_vision_saliency(
        model=model,
        image=image_t,
        text=text_t,
        target_layer=target_layer,
        use_libragrad=True,
    )

    # Determine hidden dim from feature map
    features   = extract_feature_map(
        model.vision_model, layer_idx, image_t
    )
    hidden_dim = features.shape[-1]

    # Determine patch grid size
    num_patches = features.shape[1] - 1  # subtract CLS
    grid_size   = int(num_patches ** 0.5)

    vision_prior = prepare_vision_prior(
        vision_map,
        hidden_dim=hidden_dim,
        grid_size=grid_size,
    )

    # ----------------------------------------------------------
    # Build estimator and interpreter
    # ----------------------------------------------------------
    estimator = get_compression_estimator(var, target_layer, features)

    reader = IBAInterpreter(
        model=model,
        estim=estimator,
        beta=beta,
        lr=lr,
        steps=train_steps,
        progbar=progbar,
        use_libragrad=True,    # Option A
        libragrad_init=False,   # Option B
    )

    return reader.vision_heatmap(text_t, image_t, prior=None)


# ============================================================
# Method 3: LibraGrad Only (Option C standalone)
# ============================================================

def text_heatmap_libragrad(
    text_t,
    image_t,
    model,
    layer_idx,
    **kwargs,
):
    """
    LibraGrad-only text saliency. No IB optimization.
    Used as Option C standalone baseline.

    Args:
        text_t:    [1, seq_len] tokenized text tensor
        image_t:   [1, 3, 224, 224] preprocessed image tensor
        model:     ClipWrapper instance
        layer_idx: Target transformer block index
        **kwargs:  Ignored (for uniform signature compatibility)

    Returns:
        np.ndarray [seq_len] normalized to [0, 1]
    """
    target_layer = extract_bert_layer(model.text_model, layer_idx)

    return compute_libragrad_text_saliency(
        model=model,
        image=image_t,
        text=text_t,
        target_layer=target_layer,
        use_libragrad=True,
    )


def vision_heatmap_libragrad(
    text_t,
    image_t,
    model,
    layer_idx,
    **kwargs,
):
    """
    LibraGrad-only vision saliency. No IB optimization.
    Used as Option C standalone baseline.

    Args:
        text_t:    [1, seq_len] tokenized text tensor
        image_t:   [1, 3, 224, 224] preprocessed image tensor
        model:     ClipWrapper instance
        layer_idx: Target transformer block index
        **kwargs:  Ignored (for uniform signature compatibility)

    Returns:
        np.ndarray [224, 224] normalized to [0, 1]
    """
    target_layer = extract_bert_layer(model.vision_model, layer_idx)

    return compute_libragrad_vision_saliency(
        model=model,
        image=image_t,
        text=text_t,
        target_layer=target_layer,
        use_libragrad=True,
    )


# ============================================================
# Method 4: Fusion (Option C combined)
# ============================================================

def text_heatmap_fusion(
    text_t,
    image_t,
    model,
    layer_idx,
    beta,
    var,
    lr=1.0,
    train_steps=10,
    progbar=True,
):
    """
    Fusion text attribution: element-wise product of
    M2IB and LibraGrad maps.
    Used as Option C combined baseline.

    Returns:
        np.ndarray [seq_len] normalized to [0, 1]
    """
    iba_map = text_heatmap_iba(
        text_t, image_t, model, layer_idx,
        beta, var, lr, train_steps, progbar
    )
    lg_map = text_heatmap_libragrad(
        text_t, image_t, model, layer_idx
    )

    # Align lengths (IBA trims SOT/EOT in eval but not here)
    min_len = min(len(iba_map), len(lg_map))

    return compute_fusion_map(
        iba_map[:min_len],
        lg_map[:min_len],
    )


def vision_heatmap_fusion(
    text_t,
    image_t,
    model,
    layer_idx,
    beta,
    var,
    lr=1.0,
    train_steps=10,
    progbar=True,
):
    """
    Fusion vision attribution: element-wise product of
    M2IB and LibraGrad maps.
    Used as Option C combined baseline.

    Returns:
        np.ndarray [224, 224] normalized to [0, 1]
    """
    iba_map = vision_heatmap_iba(
        text_t, image_t, model, layer_idx,
        beta, var, lr, train_steps, progbar
    )
    lg_map = vision_heatmap_libragrad(
        text_t, image_t, model, layer_idx
    )

    return compute_fusion_map(iba_map, lg_map)


# ============================================================
# Registry
#
# Used by compare_methods.py and benchmark_layers.py
# to iterate over all methods uniformly.
# ============================================================

VISION_METHODS = {
    "m2ib":      vision_heatmap_iba,
    "lciba":     vision_heatmap_lciba,
    "libragrad": vision_heatmap_libragrad,
    "fusion":    vision_heatmap_fusion,
}

TEXT_METHODS = {
    "m2ib":      text_heatmap_iba,
    "lciba":     text_heatmap_lciba,
    "libragrad": text_heatmap_libragrad,
    "fusion":    text_heatmap_fusion,
}