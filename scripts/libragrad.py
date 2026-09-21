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


# core saliency stuff
def _compute_saliency(
    model: nn.Module,
    image: torch.Tensor,
    text: torch.Tensor,
    target_layer: nn.Module,
    modality: str,
    use_libragrad: bool = True,
) -> torch.Tensor:

    assert modality in ("vision", "text"), \
        f"modality must be 'vision' or 'text', got {modality}"

    device = next(model.parameters()).device
    image = image.to(device)
    text = text.to(device)

    libra_manager: Optional[HookManager] = None

    if use_libragrad:
        libra_manager = install_libragrad_hooks(model)

    # recorder
    recorder = ActivationGradientRecorder(
        target_layer
    )
    recorder.install()

    model.zero_grad()
    model.eval()

    image_features = model.get_image_features(
        image
    )

    text_features = model.get_text_features(
        text
    )

    # cosine loss, same fitting thing
    cos = torch.nn.CosineSimilarity(
        dim=-1,
        eps=1e-6
    )

    loss = -cos(
        image_features,
        text_features
    ).mean()

    loss.backward()

    saliency = recorder.get_saliency()

    recorder.remove()

    if libra_manager is not None:
        remove_libragrad_hooks(
            libra_manager
        )

    model.zero_grad()

    if saliency is None:
        raise RuntimeError(
            f"[libragrad] Saliency is None for modality={modality}. "
            "Check that target_layer is a direct child of the transformer."
        )

    return saliency


# vision
def compute_libragrad_vision_saliency(
    model: nn.Module,
    image: torch.Tensor,
    text: torch.Tensor,
    target_layer: nn.Module,
    use_libragrad: bool = True,
    output_size: int = 224,
) -> np.ndarray:

    saliency = _compute_saliency(
        model=model,
        image=image,
        text=text,
        target_layer=target_layer,
        modality="vision",
        use_libragrad=use_libragrad,
    )

    # 50 tokens = cls + 49 patches
    # cls_val = saliency[0]
    # patch_saliency = saliency[1:] / (cls_val + 1e-6)
    patch_saliency = saliency[1:]

    grid = patch_saliency.reshape(
        1,
        1,
        7,
        7,
    ).float()

    grid = torch.nn.functional.interpolate(
        grid,
        size=output_size,
        mode="bilinear",
        align_corners=False,
    )[0, 0]

    saliency_map = (
        grid
        .detach()
        .cpu()
        .numpy()
    )

    return normalize(saliency_map)


# text stuff
def compute_libragrad_text_saliency(
    model: nn.Module,
    image: torch.Tensor,
    text: torch.Tensor,
    target_layer: nn.Module,
    use_libragrad: bool = True,
) -> np.ndarray:

    saliency = _compute_saliency(
        model=model,
        image=image,
        text=text,
        target_layer=target_layer,
        modality="text",
        use_libragrad=use_libragrad,
    )

    saliency_map = (
        saliency
        .cpu()
        .detach()
        .numpy()
    )

    return normalize(saliency_map)


# fusion thing
def compute_fusion_map(
    iba_map: np.ndarray,
    libragrad_map: np.ndarray,
) -> np.ndarray:

    assert iba_map.shape == libragrad_map.shape, (
        f"Shape mismatch: iba_map={iba_map.shape}, "
        f"libragrad_map={libragrad_map.shape}"
    )

    fused = iba_map * libragrad_map

    return normalize(fused)


# lc-iba prior preparation
def prepare_vision_prior(
    saliency_map: np.ndarray,
    hidden_dim: int = 768,
    grid_size: int = 7,
) -> torch.Tensor:

    # image -> patch grid
    t = (
        torch.tensor(saliency_map)
        .float()
        .unsqueeze(0)
        .unsqueeze(0)
    )

    t = torch.nn.functional.interpolate(
        t,
        size=(grid_size, grid_size),
        mode="bilinear",
        align_corners=False,
    )

    patch_prior = (
        t[0, 0]
        .reshape(-1)
    )

    # cls stays 1
    cls_prior = torch.ones(1)

    token_prior = torch.cat([
        cls_prior,
        patch_prior
    ])

    # [50] -> [50, hidden]
    token_prior = (
        token_prior
        .unsqueeze(-1)
        .expand(-1, hidden_dim)
    )

    return token_prior.unsqueeze(0)


def prepare_text_prior(
    saliency_map: np.ndarray,
    hidden_dim: int = 512,
) -> torch.Tensor:

    t = torch.tensor(
        saliency_map
    ).float()

    t = (
        t
        .unsqueeze(-1)
        .expand(-1, hidden_dim)
    )

    return t.unsqueeze(0)


# both
def compute_libragrad_both(
    model: nn.Module,
    image: torch.Tensor,
    text: torch.Tensor,
    vision_layer: nn.Module,
    text_layer: nn.Module,
    use_libragrad: bool = True,
    output_size: int = 224,
) -> dict:

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

    vision_prior = prepare_vision_prior(
        vision_map
    )

    text_prior = prepare_text_prior(
        text_map
    )

    return {
        "vision_map": vision_map,
        "text_map": text_map,
        "vision_prior": vision_prior,
        "text_prior": text_prior,
    }
