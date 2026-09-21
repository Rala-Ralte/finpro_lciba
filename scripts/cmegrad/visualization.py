import os, time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# import torch
# import matplotlib.cm as cm

from matplotlib.text import Text
from scipy.ndimage import gaussian_filter
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image


class TextWithBGColor(Text):

    def __init__(self, x, y, text, bgcolor, *args, **kwargs):
        super().__init__(x, y, text, *args, **kwargs)
        self.bgcolor = bgcolor

    def draw(self, renderer, *args, **kwargs):

        # bbox stuff
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

    # starting point
    x = 0.0
    y = max_height

    # space between tokens
    # space_w = 2
    space_w = fontsize * 0.2

    for i, (token, color) in enumerate(
        zip(tokens, rgba_colors)
    ):

        t = TextWithBGColor(
            x,
            y * 0.7,
            token,
            color,
            fontsize=fontsize
        )

        ax.add_artist(t)

        # rough text width
        # token_w = fontsize * len(token)
        token_w = fontsize * 0.4 * len(token)

        if i + 1 < len(tokens):

            next_w = (
                fontsize * 0.4 * len(tokens[i + 1])
                + space_w
            )

            if x + token_w + next_w >= max_width:
                x = 0.0
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

    # normalize scores first
    mn = min(scores)
    mx = max(scores)

    # norm = []
    # for s in scores:
    #     norm.append((s-mn)/(mx-mn))

    norm = [
        (s - mn) / (mx - mn + 1e-10)
        for s in scores
    ]

    colors = [
        (
            0.2,
            0.4,
            1.0,
            min_alpha + s * (max_alpha - min_alpha)
        )
        for s in norm
    ]

    return colors


def boxes_to_heatmap(
    img_pil: Image.Image,
    boxes: "torch.Tensor",
    attr: "torch.Tensor",
    sigma: float = 15,
    top_k: int = 3,
) -> np.ndarray:

    # image dimensions
    W, H = img_pil.size

    # blank heatmap
    canvas = np.zeros(
        (H, W),
        dtype=np.float32
    )

    scores = attr.float().numpy()

    # normalize
    scores = (
        scores - scores.min()
    ) / (
        scores.max() - scores.min() + 1e-10
    )

    boxes_np = boxes.numpy()

    # get top boxes
    # top_idx = np.argsort(scores)[-top_k:]
    top_idx = np.argsort(scores)[::-1][:top_k]

    for i in top_idx:

        # center point
        cx = int(
            (boxes_np[i, 0] + boxes_np[i, 2])
            / 2 * W
        )

        cy = int(
            (boxes_np[i, 1] + boxes_np[i, 3])
            / 2 * H
        )

        # keep inside image
        cx = max(0, min(W - 1, cx))
        cy = max(0, min(H - 1, cy))

        # add score
        canvas[cy, cx] += scores[i] ** 2.0

    # blur it
    canvas = gaussian_filter(
        canvas,
        sigma=sigma
    )

    mn, mx = canvas.min(), canvas.max()

    return (
        canvas - mn
    ) / (
        mx - mn + 1e-10
    )


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

    # remove special tokens
    # display_tokens = tokens
    display_tokens = [
        t for t in tokens
        if t not in ["[PAD]", "[CLS]", "[SEP]"]
    ]

    # image -> numpy
    img_np = (
        np.array(img_pil)
        .astype(np.float32)
        / 255.0
    )

    # vision heatmap
    vmap = boxes_to_heatmap(
        img_pil,
        boxes,
        vis_attr_raw,
        sigma=sigma,
        top_k=top_k,
    )

    vis_overlay = show_cam_on_image(
        img_np,
        vmap,
        use_rgb=True
    )

    # text colors
    tmap_list = text_attr.float().tolist()

    # don't use more scores than tokens
    tmap_clipped = tmap_list[:len(display_tokens)]

    rgba_colors = generate_shades_with_alpha(
        tmap_clipped
    )

    # plot
    fig, axs = plt.subplots(
        1,
        2,
        figsize=(12, 4)
    )

    # title
    fig.suptitle(
        f'Q: "{info.get("question", "")}"   →   A: {info["pred_answer"]}\n'
        f'H_vis={info["H_vis"]:.3f}  '
        f'H_lang={info["H_lang"]:.3f}'
        f'  H_gap={info["H_gap"]:.3f}  '
        f'w_vis={info["w_vis"]:.3f}',
        fontsize=10,
        fontweight="bold",
    )

    axs[0].imshow(vis_overlay)
    axs[0].set_title(
        "CMEGrad Vision (LibraGrad IxG)",
        fontsize=9
    )
    axs[0].axis("off")

    plot_text_with_colors(
        axs[1],
        display_tokens,
        rgba_colors,
        max_width=max_width,
        max_height=max_height,
        fontsize=fontsize,
    )

    axs[1].set_xlim(
        0,
        max_width
    )

    axs[1].set_ylim(
        -max_height,
        max_height
    )

    axs[1].set_title(
        "CMEGrad Text Attribution (LibraGrad IxG)",
        fontsize=9
    )

    axs[1].axis("off")

    plt.tight_layout()

    if save_path:
        # make sure folder exists
        os.makedirs(
            os.path.dirname(save_path),
            exist_ok=True
        )

        # plt.savefig(save_path)
        plt.savefig(
            save_path,
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white"
        )

    if show:
        plt.show()

    # close after saving
    plt.close(fig)

    return fig
