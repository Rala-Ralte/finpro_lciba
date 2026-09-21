"""
scripts/visualization.py

Visualization utilities for LC-IBA experiment outputs.

Based on scripts/plot.py from M2IB:
    https://github.com/YingWANGG/M2IB

Provides:
    - Vision heatmap overlay on image
    - Text token highlighting
    - Side-by-side multi-method comparison grid
    - Single-sample result saving
    - Batch result saving
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.text import Text
from PIL import Image

# Use non-interactive backend when running experiments
matplotlib.use("Agg")


# ============================================================
# Color Utilities
# ============================================================

def _to_display_image(image: np.ndarray) -> np.ndarray:
    """
    Ensure image is float32 in [0, 1] for display.

    Accepts:
        - uint8 [0, 255]
        - float [0, 1]
        - float [-any, any] (clips and normalizes)
    """
    img = image.astype(np.float32)

    if img.max() > 1.0:
        img = img / 255.0

    return np.clip(img, 0.0, 1.0)


def _reverse_clip_normalization(tensor_chw: np.ndarray) -> np.ndarray:
    """
    Reverse CLIP preprocessing normalization.

    CLIP normalizes with:
        mean = [0.48145466, 0.4578275,  0.40821073]
        std  = [0.26862954, 0.26130258, 0.40821073]

    Args:
        tensor_chw: np.ndarray [3, H, W]

    Returns:
        np.ndarray [H, W, 3] in [0, 1]
    """
    mean = np.array([0.48145466, 0.4578275,  0.40821073])
    std  = np.array([0.26862954, 0.26130258, 0.40821073])

    img = tensor_chw.transpose(1, 2, 0)   # [H, W, 3]
    img = img * std + mean
    return np.clip(img, 0.0, 1.0)


def _generate_token_colors(
    scores: np.ndarray,
    min_alpha: float = 0.15,
    max_alpha: float = 0.85,
) -> list:
    """
    Map attribution scores to RGBA blue highlight colors.

    Args:
        scores:    1D array of attribution scores
        min_alpha: Minimum alpha for lowest score
        max_alpha: Maximum alpha for highest score

    Returns:
        List of (R, G, B, A) tuples
    """
    s_min = scores.min()
    s_max = scores.max()

    if s_max - s_min < 1e-8:
        normalized = np.zeros_like(scores)
    else:
        normalized = (scores - s_min) / (s_max - s_min)

    colors = []
    for s in normalized:
        alpha = min_alpha + s * (max_alpha - min_alpha)
        colors.append((0.2, 0.4, 1.0, alpha))

    return colors


# ============================================================
# Vision Heatmap Overlay
# ============================================================

def overlay_vision_heatmap(image, heatmap, alpha=0.5, colormap="jet"):
    import matplotlib.cm as cm
    from scripts.utils import normalize
    image   = _to_display_image(image)
    heatmap = np.clip(normalize(heatmap), 0.0, 1.0)
    colored = cm.get_cmap(colormap)(heatmap)[:, :, :3]
    blended = (1 - alpha) * image + alpha * colored
    return np.clip(blended, 0.0, 1.0)

# ============================================================
# Text Token Visualization
# ============================================================

class _TokenText(Text):
    """Matplotlib Text with colored background box."""

    def __init__(self, x, y, text, bgcolor, *args, **kwargs):
        super().__init__(x, y, text, *args, **kwargs)
        self.bgcolor = bgcolor

    def draw(self, renderer, *args, **kwargs):
        r, g, b, a = self.bgcolor
        bbox = dict(
            facecolor=(r, g, b),
            edgecolor=(r, g, b),
            boxstyle="round,pad=0.01",
            alpha=a,
        )
        self.set_bbox(bbox)
        super().draw(renderer, *args, **kwargs)


def _plot_tokens(
    ax,
    tokens: list,
    colors: list,
    max_width: float,
    max_height: float,
    fontsize: int = 12,
):
    """
    Draw tokens with colored background onto an axes.

    Args:
        ax:         Matplotlib Axes
        tokens:     List of token strings
        colors:     List of RGBA tuples
        max_width:  Axes x-limit
        max_height: Axes y-limit
        fontsize:   Font size for tokens
    """
    x = 0.0
    y = max_height
    space_w = fontsize * 0.2

    for i, (token, color) in enumerate(zip(tokens, colors)):
        t = _TokenText(
            x, y * 0.7, token, color,
            fontsize=fontsize,
        )
        ax.add_artist(t)

        token_w = fontsize * 0.4 * len(token)

        if i + 1 < len(tokens):
            next_w = fontsize * 0.4 * len(tokens[i + 1]) + space_w
            if x + token_w + next_w >= max_width:
                x  = 0.0
                y -= 1.0
            else:
                x += token_w + space_w
        else:
            x += token_w


# ============================================================
# Single Sample Visualization
# ============================================================

def visualize_single(
    image: np.ndarray,
    vmap: np.ndarray,
    tmap: np.ndarray,
    tokens: list,
    title: str = None,
    save_path: str = None,
    show: bool = False,
    bboxes: list = None,
) -> plt.Figure:
    """
    Visualize vision heatmap and text attribution for one sample.

    Args:
        image:     np.ndarray [224, 224, 3] in [0, 1]
        vmap:      np.ndarray [224, 224] vision saliency
        tmap:      np.ndarray [seq_len] text saliency
        tokens:    List of token strings (including SOT/EOT)
        title:     Optional figure title
        save_path: If given, saves figure to this path
        show:      If True, calls plt.show()
        bboxes:    Optional list of (x, y, w, h) bounding boxes

    Returns:
        plt.Figure
    """
    # Strip SOT and EOT tokens for display
    display_tokens = [t.split("<")[0] for t in tokens[1:-1]]
    display_tmap   = tmap[1:-1] if len(tmap) > 2 else tmap

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # ---- Vision panel ----
    overlaid = overlay_vision_heatmap(image, vmap)
    axes[0].imshow(overlaid)

    if bboxes:
        for x, y, w, h in bboxes:
            rect = mpatches.Rectangle(
                (x, y), w, h,
                linewidth=2, edgecolor="red", facecolor="none"
            )
            axes[0].add_patch(rect)

    axes[0].axis("off")
    axes[0].set_title("Vision Saliency", fontsize=10)

    # ---- Text panel ----
    colors = _generate_token_colors(display_tmap)
    _plot_tokens(
        axes[1], display_tokens, colors,
        max_width=100, max_height=4, fontsize=13,
    )
    axes[1].set_xlim(0, 100)
    axes[1].set_ylim(-4, 4)
    axes[1].axis("off")
    axes[1].set_title("Text Attribution", fontsize=10)

    if title:
        fig.suptitle(title, fontsize=11, y=1.01)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)

    if show:
        plt.show()

    return fig


# ============================================================
# Multi-Method Comparison Grid
# ============================================================

def visualize_comparison(
    image: np.ndarray,
    method_vmaps: dict,
    method_tmaps: dict,
    tokens: list,
    title: str = None,
    save_path: str = None,
    show: bool = False,
) -> plt.Figure:
    """
    Side-by-side comparison of multiple attribution methods.

    Layout:
        Row per method.
        Col 0: Vision heatmap overlay.
        Col 1: Text token attribution.

    Args:
        image:         np.ndarray [224, 224, 3] in [0, 1]
        method_vmaps:  dict {method_name: np.ndarray [224, 224]}
        method_tmaps:  dict {method_name: np.ndarray [seq_len]}
        tokens:        List of token strings
        title:         Optional figure suptitle
        save_path:     If given, saves figure here
        show:          If True, calls plt.show()

    Returns:
        plt.Figure
    """
    methods = list(method_vmaps.keys())
    n       = len(methods)

    display_tokens = [t.split("<")[0] for t in tokens[1:-1]]

    fig, axes = plt.subplots(n, 2, figsize=(12, 3.5 * n))

    if n == 1:
        axes = [axes]

    for i, method in enumerate(methods):
        vmap = method_vmaps[method]
        tmap = method_tmaps[method]

        display_tmap = tmap[1:-1] if len(tmap) > 2 else tmap

        # Vision
        overlaid = overlay_vision_heatmap(image, vmap)
        axes[i][0].imshow(overlaid)
        axes[i][0].axis("off")
        axes[i][0].set_title(f"{method} — Vision", fontsize=9)

        # Text
        colors = _generate_token_colors(display_tmap)
        _plot_tokens(
            axes[i][1], display_tokens, colors,
            max_width=100, max_height=4, fontsize=12,
        )
        axes[i][1].set_xlim(0, 100)
        axes[i][1].set_ylim(-4, 4)
        axes[i][1].axis("off")
        axes[i][1].set_title(f"{method} — Text", fontsize=9)

    if title:
        fig.suptitle(title, fontsize=12, y=1.01)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)

    if show:
        plt.show()

    plt.close(fig)
    return fig


# ============================================================
# Metrics Bar Chart
# ============================================================

def plot_metrics_comparison(
    metrics: dict,
    metric_keys: list = None,
    title: str = "Method Comparison",
    save_path: str = None,
    show: bool = False,
) -> plt.Figure:
    """
    Bar chart comparing methods across evaluation metrics.

    Args:
        metrics:     dict {method_name: {metric_key: value}}
                     e.g. {"m2ib": {"vdrop": 12.3, "vincr": 5.1}, ...}
        metric_keys: List of metric keys to plot.
                     Default: ["vdrop", "vincr", "tdrop", "tincr"]
        title:       Chart title
        save_path:   If given, saves figure here
        show:        If True, calls plt.show()

    Returns:
        plt.Figure
    """
    if metric_keys is None:
        metric_keys = ["vdrop", "vincr", "tdrop", "tincr"]

    methods = list(metrics.keys())
    n_metrics = len(metric_keys)
    n_methods = len(methods)

    x      = np.arange(n_metrics)
    width  = 0.8 / n_methods
    colors = plt.cm.tab10(np.linspace(0, 0.5, n_methods))

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (method, color) in enumerate(zip(methods, colors)):
        vals = [metrics[method].get(k, 0.0) for k in metric_keys]
        ax.bar(
            x + i * width - (n_methods - 1) * width / 2,
            vals,
            width,
            label=method,
            color=color,
            alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_keys, fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)

    if show:
        plt.show()

    plt.close(fig)
    return fig


# ============================================================
# Layer Benchmark Plot
# ============================================================

def plot_layer_benchmark(
    layer_results: dict,
    metric: str = "vdrop",
    title: str = "Layer Benchmark",
    save_path: str = None,
    show: bool = False,
) -> plt.Figure:
    """
    Line plot of metric vs layer index for each method.

    Args:
        layer_results: dict {method_name: {layer_idx: metric_val}}
        metric:        Which metric to plot
        title:         Chart title
        save_path:     If given, saves here
        show:          If True, calls plt.show()

    Returns:
        plt.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    for method, layer_dict in layer_results.items():
        layers = sorted(layer_dict.keys())
        values = [layer_dict[l] for l in layers]
        ax.plot(layers, values, marker="o", label=method)

    ax.set_xlabel("Layer Index", fontsize=11)
    ax.set_ylabel(metric, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)

    if show:
        plt.show()

    plt.close(fig)
    return fig