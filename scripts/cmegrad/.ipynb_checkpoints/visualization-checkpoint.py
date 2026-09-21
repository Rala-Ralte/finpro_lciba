"""
scripts/cmegrad/visualization.py

Heatmap and text attribution visualization for CMEGrad (LXMERT).

Vision: center-point Gaussian heatmap from FRCNN bounding boxes.
Text:   blue-alpha token highlighting.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.text import Text
from scipy.ndimage import gaussian_filter
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image


# ============================================================
# Text Visualization
# ============================================================

class TextWithBGColor(Text):
    """Matplotlib Text artist with colored background box."""

    def __init__(self, x, y, text, bgcolor, *args, **kwargs):
        super().__init__(x, y, text, *args, **kwargs)
        self.bgcolor = bgcolor

    def draw(self, renderer, *args, **kwargs):
        bbox = dict(
            facecolor=self.bgcolor,
            edgecolor=self.bgcolor,
            boxstyle="round,pad=0.01",
            alpha=self.bgcolor[3],
        )
        self.set_bbox(bbox)
        super().draw(renderer, *args, **kwargs)


def plot_text_with_colors(
    ax,
    tokens: list,
    rgba_colors: list,
    max_width: float = 100,
    max_height: float = 4,
    fontsize: int = 12,
):
    """
    Draw tokens with colored backgrounds on a matplotlib axes.

    Args:
        ax:          Matplotlib Axes
        tokens:      List of token strings (PAD/CLS/SEP already filtered)
        rgba_colors: List of (R, G, B, A) tuples
        max_width:   x-axis limit
        max_height:  y-axis start
        fontsize:    Token font size
    """
    x       = 0.0
    y       = max_height
    space_w = fontsize * 0.2

    for i, (token, color) in enumerate(zip(tokens, rgba_colors)):
        t = TextWithBGColor(x, y * 0.7, token, color, fontsize=fontsize)
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


def generate_shades_with_alpha(
    scores: list,
    min_alpha: float = 0.0,
    max_alpha: float = 0.85,
) -> list:
    """
    Map attribution scores to blue RGBA colors with alpha proportional to score.

    Returns list of (0.2, 0.4, 1.0, alpha) tuples.
    """
    mn   = min(scores)
    mx   = max(scores)
    norm = [(s - mn) / (mx - mn + 1e-10) for s in scores]
    return [
        (0.2, 0.4, 1.0, min_alpha + s * (max_alpha - min_alpha))
        for s in norm
    ]


# ============================================================
# Vision Heatmap
# ============================================================

def boxes_to_heatmap(
    img_pil: Image.Image,
    boxes: "torch.Tensor",
    attr: "torch.Tensor",
    sigma: float = 15,
    top_k: int = 3,
) -> np.ndarray:
    """
    Convert FRCNN region attribution to a pixel heatmap.

    Places attribution scores at region centers, applies Gaussian blur.

    Args:
        img_pil: PIL Image (for W, H)
        boxes:   [36, 4] normalized (x1, y1, x2, y2) tensor
        attr:    [36] attribution scores tensor
        sigma:   Gaussian blur sigma
        top_k:   Number of top regions to include

    Returns:
        heatmap: np.ndarray [H, W] normalized to [0, 1]
    """
    W, H     = img_pil.size
    canvas   = np.zeros((H, W), dtype=np.float32)
    scores   = attr.float().numpy()
    scores   = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
    boxes_np = boxes.numpy()
    top_idx  = np.argsort(scores)[::-1][:top_k]

    for i in top_idx:
        cx = int((boxes_np[i, 0] + boxes_np[i, 2]) / 2 * W)
        cy = int((boxes_np[i, 1] + boxes_np[i, 3]) / 2 * H)
        cx = max(0, min(W - 1, cx))
        cy = max(0, min(H - 1, cy))
        canvas[cy, cx] += scores[i] ** 2.0

    canvas = gaussian_filter(canvas, sigma=sigma)
    mn, mx = canvas.min(), canvas.max()
    return (canvas - mn) / (mx - mn + 1e-10)


# ============================================================
# Single Sample Visualization
# ============================================================

def visualize_cmegrad(
    img_pil: Image.Image,
    vis_attr_raw: "torch.Tensor",
    text_attr: "torch.Tensor",
    boxes: "torch.Tensor",
    tokens: list,
    info: dict,
    save_path: str = None,
    show: bool = False,
    sigma: float = 25,
    top_k: int = 5,
    max_width: float = 100,
    max_height: float = 4,
    fontsize: int = 13,
    dpi: int = 150,
) -> plt.Figure:
    """
    Full CMEGrad visualization: vision heatmap + text attribution.

    Args:
        img_pil:      PIL Image
        vis_attr_raw: [36] raw (pre-refinement) visual attribution
        text_attr:    [T] refined text attribution
        boxes:        [36, 4] normalized bounding boxes
        tokens:       List of all token strings
        info:         Dict from CMEGradLXMERT.generate() with H_vis, H_lang, etc.
        save_path:    If given, saves figure here
        show:         If True, calls plt.show()

    Returns:
        plt.Figure
    """
    # Filter tokens for display
    display_tokens = [
        t for t in tokens
        if t not in ["[PAD]", "[CLS]", "[SEP]"]
    ]

    # Vision heatmap
    img_np      = np.array(img_pil).astype(np.float32) / 255.0
    vmap        = boxes_to_heatmap(img_pil, boxes, vis_attr_raw,
                                   sigma=sigma, top_k=top_k)
    vis_overlay = show_cam_on_image(img_np, vmap, use_rgb=True)

    # Text colors
    tmap_list   = text_attr.float().tolist()
    tmap_clipped = tmap_list[:len(display_tokens)]
    rgba_colors  = generate_shades_with_alpha(tmap_clipped)

    # Plot
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(
        f'Q: "{info.get("question", "")}"   →   A: {info["pred_answer"]}\n'
        f'H_vis={info["H_vis"]:.3f}  H_lang={info["H_lang"]:.3f}'
        f'  H_gap={info["H_gap"]:.3f}  w_vis={info["w_vis"]:.3f}',
        fontsize=10, fontweight="bold",
    )

    axs[0].imshow(vis_overlay)
    axs[0].set_title("CMEGrad Vision (LibraGrad IxG)", fontsize=9)
    axs[0].axis("off")

    plot_text_with_colors(
        axs[1], display_tokens, rgba_colors,
        max_width=max_width, max_height=max_height, fontsize=fontsize,
    )
    axs[1].set_xlim(0, max_width)
    axs[1].set_ylim(-max_height, max_height)
    axs[1].set_title("CMEGrad Text Attribution (LibraGrad IxG)", fontsize=9)
    axs[1].axis("off")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")

    if show:
        plt.show()

    plt.close(fig)
    return fig