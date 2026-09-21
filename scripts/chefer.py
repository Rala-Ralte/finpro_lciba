import os, sys
import numpy as np
import torch
import torch.nn.functional as F

# import matplotlib.pyplot as plt
# from copy import deepcopy
# import random

from scripts.utils import normalize
from scripts.cmegrad.metrics import entropy_weight  # not used here, old

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# lazy cache stuff
_CHEFER_MODEL = None
_CHEFER_PREPROCESS = None


def load_chefer_model(
    clip_variant: str = "ViT-B/32",
    chefer_clip_path: str = None,
):
    global _CHEFER_MODEL, _CHEFER_PREPROCESS

    if _CHEFER_MODEL is not None:
        return _CHEFER_MODEL, _CHEFER_PREPROCESS

    # path thing
    if chefer_clip_path is None:
        chefer_clip_path = os.environ.get(
            "CHEFER_CLIP_PATH",
            os.path.join(os.getcwd(), "CLIP"),
        )

    parent = os.path.dirname(chefer_clip_path)

    if parent not in sys.path:
        sys.path.insert(0, parent)

    try:
        import CLIP.clip as chefer_clip
    except ImportError as e:
        raise ImportError(
            f"Cannot import Chefer's CLIP fork from {chefer_clip_path}.\n"
            f"Copy the CLIP/ directory from Transformer-MM-Explainability "
            f"to your project root, or set CHEFER_CLIP_PATH.\n"
            f"Original error: {e}"
        )

    model, preprocess = chefer_clip.load(
        clip_variant,
        device=device,
        jit=False,
    )

    model = model.float().eval()

    _CHEFER_MODEL = model
    _CHEFER_PREPROCESS = preprocess

    print(
        f"[chefer] Loaded Chefer CLIP {clip_variant} on {device}"
    )

    return model, preprocess


# rollout
def _chefer_interpret(
    image_t: torch.Tensor,
    text_t: torch.Tensor,
    model,
    start_layer: int = -1,
    start_layer_text: int = -1,
):

    batch_size = text_t.shape[0]  # should be 1
    images = image_t.repeat(
        batch_size,
        1,
        1,
        1,
    )

    logits_per_image, _ = model(
        images,
        text_t,
    )

    # one hot target
    index = list(range(batch_size))

    one_hot = np.zeros(
        (
            logits_per_image.shape[0],
            logits_per_image.shape[1],
        ),
        dtype=np.float32,
    )

    one_hot[
        torch.arange(logits_per_image.shape[0]),
        index
    ] = 1

    one_hot = (
        torch.from_numpy(one_hot)
        .requires_grad_(True)
        .to(device)
    )

    one_hot = torch.sum(
        one_hot * logits_per_image
    )

    model.zero_grad()

    # need graph because grad is called for each block
    one_hot.backward(retain_graph=True)

    # vision blocks
    image_blocks = list(
        dict(
            model.visual.transformer.resblocks
            .named_children()
        ).values()
    )

    if start_layer == -1:
        start_layer = len(image_blocks) - 1

    num_tokens = (
        image_blocks[0]
        .attn_probs
        .shape[-1]
    )

    R = torch.eye(
        num_tokens,
        num_tokens,
        dtype=image_blocks[0].attn_probs.dtype,
    ).to(device)

    R = (
        R.unsqueeze(0)
        .expand(
            batch_size,
            num_tokens,
            num_tokens,
        )
    )

    for i, blk in enumerate(image_blocks):

        if i < start_layer:
            continue

        grad = torch.autograd.grad(
            one_hot,
            [blk.attn_probs],
            retain_graph=True,
        )[0].detach()

        cam = blk.attn_probs.detach()

        cam = cam.reshape(
            -1,
            cam.shape[-1],
            cam.shape[-1],
        )

        grad = grad.reshape(
            -1,
            grad.shape[-1],
            grad.shape[-1],
        )

        cam = grad * cam

        cam = cam.reshape(
            batch_size,
            -1,
            cam.shape[-1],
            cam.shape[-1],
        )

        cam = (
            cam
            .clamp(min=0)
            .mean(dim=1)
        )

        R = R + torch.bmm(
            cam,
            R,
        )

    R_image = R[:, 0, 1:]

    # text rollout
    text_blocks = list(
        dict(
            model.transformer.resblocks
            .named_children()
        ).values()
    )

    if start_layer_text == -1:
        start_layer_text = len(text_blocks) - 1

    num_tokens_t = (
        text_blocks[0]
        .attn_probs
        .shape[-1]
    )

    R_text = torch.eye(
        num_tokens_t,
        num_tokens_t,
        dtype=text_blocks[0].attn_probs.dtype,
    ).to(device)

    R_text = (
        R_text
        .unsqueeze(0)
        .expand(
            batch_size,
            num_tokens_t,
            num_tokens_t,
        )
    )

    for i, blk in enumerate(text_blocks):

        if i < start_layer_text:
            continue

        grad = torch.autograd.grad(
            one_hot,
            [blk.attn_probs],
            retain_graph=True,
        )[0].detach()

        cam = blk.attn_probs.detach()

        cam = cam.reshape(
            -1,
            cam.shape[-1],
            cam.shape[-1],
        )

        grad = grad.reshape(
            -1,
            grad.shape[-1],
            grad.shape[-1],
        )

        cam = grad * cam

        cam = cam.reshape(
            batch_size,
            -1,
            cam.shape[-1],
            cam.shape[-1],
        )

        cam = (
            cam
            .clamp(min=0)
            .mean(dim=1)
        )

        R_text = R_text + torch.bmm(
            cam,
            R_text,
        )

    model.zero_grad()

    return R_image, R_text


# map helpers
def _vision_map_from_R(
    R_image: torch.Tensor,
    output_size: int = 224,
) -> np.ndarray:

    rel = R_image[0]
    dim = int(rel.numel() ** 0.5)

    grid = rel.reshape(
        1,
        1,
        dim,
        dim,
    ).float()

    grid = F.interpolate(
        grid,
        size=output_size,
        mode="bilinear",
        align_corners=False,
    )

    arr = (
        grid[0, 0]
        .detach()
        .cpu()
        .numpy()
    )

    return normalize(arr)


def _text_map_from_R(
    R_text: torch.Tensor,
    text_t: torch.Tensor,
) -> np.ndarray:

    seq_len = text_t.shape[1]

    # eot is highest token id
    eot_idx = int(
        text_t[0]
        .argmax()
        .item()
    )

    R_row = R_text[0]

    rel = R_row[
        eot_idx,
        1:eot_idx,
    ]

    # full length again
    out = np.zeros(
        seq_len,
        dtype=np.float32,
    )

    content_len = rel.shape[0]

    out[
        1:1 + content_len
    ] = (
        rel
        .detach()
        .cpu()
        .float()
        .numpy()
    )

    mn, mx = out.min(), out.max()

    if mx - mn > 1e-8:
        out = (
            out - mn
        ) / (
            mx - mn
        )

    return out


def _preprocess_image_for_chefer(
    image_t: torch.Tensor,
    chefer_preprocess,
) -> torch.Tensor:

    # already preprocessed by wrapper
    # keeping this separate in case that changes later
    return image_t.float().to(device)


# public funcs
def vision_heatmap_chefer(
    text_t: torch.Tensor,
    image_t: torch.Tensor,
    model,
    layer_idx: int = 9,
    chefer_clip_path: str = None,
    start_layer: int = -1,
    start_layer_text: int = -1,
    **kwargs,
) -> np.ndarray:

    chefer_model, chefer_prep = load_chefer_model(
        chefer_clip_path=chefer_clip_path
    )

    img = _preprocess_image_for_chefer(
        image_t,
        chefer_prep,
    )

    R_image, _ = _chefer_interpret(
        image_t=img,
        text_t=text_t.to(device),
        model=chefer_model,
        start_layer=start_layer,
        start_layer_text=start_layer_text,
    )

    return _vision_map_from_R(R_image)


def text_heatmap_chefer(
    text_t: torch.Tensor,
    image_t: torch.Tensor,
    model,
    layer_idx: int = 9,
    chefer_clip_path: str = None,
    start_layer: int = -1,
    start_layer_text: int = -1,
    **kwargs,
) -> np.ndarray:

    chefer_model, chefer_prep = load_chefer_model(
        chefer_clip_path=chefer_clip_path
    )

    img = _preprocess_image_for_chefer(
        image_t,
        chefer_prep,
    )

    _, R_text = _chefer_interpret(
        image_t=img,
        text_t=text_t.to(device),
        model=chefer_model,
        start_layer=start_layer,
        start_layer_text=start_layer_text,
    )

    return _text_map_from_R(
        R_text,
        text_t,
    )


# both at once
def both_heatmaps_chefer(
    text_t: torch.Tensor,
    image_t: torch.Tensor,
    model,
    chefer_clip_path: str = None,
    start_layer: int = -1,
    start_layer_text: int = -1,
) -> dict:

    chefer_model, chefer_prep = load_chefer_model(
        chefer_clip_path=chefer_clip_path
    )

    img = _preprocess_image_for_chefer(
        image_t,
        chefer_prep,
    )

    R_image, R_text = _chefer_interpret(
        image_t=img,
        text_t=text_t.to(device),
        model=chefer_model,
        start_layer=start_layer,
        start_layer_text=start_layer_text,
    )

    return {
        "vision_map": _vision_map_from_R(
            R_image
        ),
        "text_map": _text_map_from_R(
            R_text,
            text_t,
        ),
    }
