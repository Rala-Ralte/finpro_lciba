import torch
import torch.nn as nn
from functools import partial
from typing import List, Optional, Tuple

# import copy
# import numpy as np
# from torch import Tensor


class HookManager:
    def __init__(self):
        self.handles = []

    def add(self, handle):
        self.handles.append(handle)

    def remove_all(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def __len__(self):
        return len(self.handles)

    def __repr__(self):
        return f"HookManager(active_hooks={len(self.handles)})"


# layernorm stuff
def layernorm_backward_hook(
    module: nn.LayerNorm,
    grad_input: Tuple[torch.Tensor, ...],
    grad_output: Tuple[torch.Tensor, ...]
) -> Tuple[torch.Tensor, ...]:

    if grad_input[0] is None:
        return grad_input

    gx = grad_input[0]
    d = gx.shape[-1]

    if hasattr(module, "weight") and module.weight is not None:

        gamma = module.weight.detach()

        # old version was just scale=1
        scale = d / (
            torch.sum(gamma ** 2) + 1e-6
        )

    else:
        scale = 1.0

    gx_scaled = gx * scale

    return (gx_scaled,) + grad_input[1:]


# attention fix
# CLIP attention is a little annoying here
#
# original:

#

#
# don't touch forward values, only make backward use V
def _libra_permute_then_forward(
    self,
    x: torch.Tensor
) -> torch.Tensor:

    x = x.permute(
        1, 0, 2
    )

    ln1_out = self.ln_1(x)

    # normal forward path
    with torch.no_grad():
        attn_fwd, _ = self.attn(
            ln1_out,
            ln1_out,
            ln1_out
        )

    # q/k detached, v still connected
    attn_bwd, _ = self.attn(
        ln1_out.detach(),
        ln1_out.detach(),
        ln1_out,
    )

    # forward from first one
    # gradients from second one
    attn_out = (
        attn_fwd
        + (
            attn_bwd
            - attn_bwd.detach()
        )
    )

    x = x + attn_out
    x = x + self.mlp(
        self.ln_2(x)
    )

    x = x.permute(
        1, 0, 2
    )

    return x


class _RestoreHandle:
    def __init__(
        self,
        layer: nn.Module,
        original_forward
    ):
        self._layer = layer
        self._original = original_forward

    def remove(self):
        self._layer.forward = self._original


def install_attention_hooks(
    model: nn.Module
) -> HookManager:

    manager = HookManager()

    # both sides
    for submodel in [
        model.vision_model,
        model.text_model
    ]:

        for layer in submodel.transformer.resblocks:

            original_forward = layer.forward

            layer.forward = partial(
                _libra_permute_then_forward,
                layer
            )

            manager.add(
                _RestoreHandle(
                    layer,
                    original_forward
                )
            )

    return manager


def install_layernorm_hooks(
    model: nn.Module
) -> HookManager:

    manager = HookManager()

    for name, module in model.named_modules():

        if isinstance(module, nn.LayerNorm):

            handle = module.register_full_backward_hook(
                layernorm_backward_hook
            )

            manager.add(handle)

    return manager


def install_libragrad_hooks(
    model: nn.Module
) -> HookManager:

    manager = HookManager()

    # layernorm hooks
    for submodel in [
        model.vision_model,
        model.text_model
    ]:

        for name, module in submodel.named_modules():

            if isinstance(module, nn.LayerNorm):

                handle = module.register_full_backward_hook(
                    layernorm_backward_hook
                )

                manager.add(handle)

    # attention swaps
    attn_manager = install_attention_hooks(model)

    manager.handles.extend(
        attn_manager.handles
    )

    return manager


def remove_libragrad_hooks(
    manager: HookManager
):
    manager.remove_all()


# activation/grad recorder
class ActivationGradientRecorder:

    def __init__(
        self,
        target_module: nn.Module
    ):
        self.target_module = target_module

        self.activations = None
        self.gradients = None

        self._forward_handle = None
        self._backward_handle = None

    def _forward_hook(
        self,
        module,
        inputs,
        output
    ):

        if isinstance(output, tuple):
            act = output[0]
        else:
            act = output

        self.activations = (
            act.detach().clone()
        )

        act.retain_grad()

    def _backward_hook(
        self,
        module,
        grad_input,
        grad_output
    ):

        if grad_output[0] is not None:
            self.gradients = (
                grad_output[0]
                .detach()
                .clone()
            )

    def install(self):

        self._forward_handle = (
            self.target_module
            .register_forward_hook(
                self._forward_hook
            )
        )

        self._backward_handle = (
            self.target_module
            .register_full_backward_hook(
                self._backward_hook
            )
        )

    def remove(self):

        if self._forward_handle is not None:

            self._forward_handle.remove()
            self._forward_handle = None

        if self._backward_handle is not None:

            self._backward_handle.remove()
            self._backward_handle = None

    def get_saliency(
        self
    ) -> Optional[torch.Tensor]:

        if (
            self.activations is None
            or self.gradients is None
        ):
            return None

        act = self.activations
        grad = self.gradients

        if act.shape != grad.shape:
            return None

        saliency = (
            act.float()
            * grad.float()
        ).abs().sum(dim=-1)

        return saliency[0].detach()

    def is_ready(self) -> bool:
        return (
            self.activations is not None
            and self.gradients is not None
        )

    def reset(self):

        self.activations = None
        self.gradients = None

    def __repr__(self):

        return (
            f"ActivationGradientRecorder("
            f"ready={self.is_ready()}, "
            f"module={type(self.target_module).__name__})"
        )


# quick forward check
def verify_hooks_dont_change_forward(
    model: nn.Module,
    dummy_image: torch.Tensor,
    dummy_text: torch.Tensor,
    atol: float = 1e-4,
) -> bool:

    with torch.no_grad():

        img_before = (
            model
            .get_image_features(dummy_image)
            .clone()
        )

        txt_before = (
            model
            .get_text_features(dummy_text)
            .clone()
        )

    manager = install_libragrad_hooks(model)

    with torch.no_grad():

        img_after = (
            model
            .get_image_features(dummy_image)
            .clone()
        )

        txt_after = (
            model
            .get_text_features(dummy_text)
            .clone()
        )

    remove_libragrad_hooks(manager)

    img_ok = torch.allclose(
        img_before,
        img_after,
        atol=atol
    )

    txt_ok = torch.allclose(
        txt_before,
        txt_after,
        atol=atol
    )

    if img_ok and txt_ok:
        print(
            "[hooks] Forward pass unchanged. Verification PASSED."
        )
    else:
        print(
            f"[hooks] WARNING: Forward changed! "
            f"img_ok={img_ok}, txt_ok={txt_ok}"
        )

    return img_ok and txt_ok
