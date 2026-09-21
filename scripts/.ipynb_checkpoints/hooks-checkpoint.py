"""
scripts/hooks.py

HookManager and LibraGrad-specific backward hooks for CLIP transformers.

Responsibilities:
    - Clean lifecycle management for PyTorch hooks
    - LibraGrad LayerNorm backward correction
    - LibraGrad Attention V-only backward routing via forward method swap
    - Forward activation + gradient recording for saliency extraction

All attention fixes operate by swapping the forward method.
LayerNorm fix operates on the backward pass only.
"""

import torch
import torch.nn as nn
from functools import partial
from typing import List, Optional, Tuple


# ============================================================
# Hook Manager
# ============================================================

class HookManager:
    """
    Manages a collection of PyTorch hook handles and
    forward-method restore handles.
    Ensures clean removal without manual tracking.
    """

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


# ============================================================
# LibraGrad LayerNorm Backward Hook
# ============================================================

def layernorm_backward_hook(
    module: nn.LayerNorm,
    grad_input: Tuple[torch.Tensor, ...],
    grad_output: Tuple[torch.Tensor, ...]
) -> Tuple[torch.Tensor, ...]:
    """
    Full backward hook for nn.LayerNorm.
    Scales gradient to preserve FullGrad-completeness.
    Scale factor: d / (sum(gamma^2) + eps)
    """
    if grad_input[0] is None:
        return grad_input

    gx = grad_input[0]
    d  = gx.shape[-1]

    if hasattr(module, "weight") and module.weight is not None:
        gamma = module.weight.detach()
        scale = d / (torch.sum(gamma ** 2) + 1e-6)
    else:
        scale = 1.0

    gx_scaled = gx * scale
    return (gx_scaled,) + grad_input[1:]


# ============================================================
# LibraGrad Attention Fix
#
# CLIP uses permute_then_forward which calls:
#   x = x + self.attention(self.ln_1(x))
# where the same tensor is Q, K, and V.
#
# register_full_backward_hook on nn.MultiheadAttention is
# unreliable for composite ops in PyTorch 2.x and corrupts
# the backward graph.
#
# Fix: swap the block's forward method to use swap_backward,
# routing gradients only through V during backward while
# keeping the correct forward values.
# ============================================================

def _libra_permute_then_forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    LibraGrad version of permute_then_forward.

    Forward pass: identical to original (correct values preserved).
    Backward pass: gradients flow only through V path, not Q or K.

    Uses swap_backward primitive:
        output = fwd.detach() + (bwd - bwd.detach())
    where:
        fwd = standard attention(ln1, ln1, ln1)   [correct values]
        bwd = V-only attention(detach, detach, ln1) [correct gradients]
    """
    x = x.permute(1, 0, 2)          # [B, T, D] → [T, B, D]

    ln1_out = self.ln_1(x)

    # Forward: standard attention — correct output values
    with torch.no_grad():
        attn_fwd, _ = self.attn(ln1_out, ln1_out, ln1_out)

    # Backward: V-only path — Q and K detached from gradient graph
    attn_bwd, _ = self.attn(
        ln1_out.detach(),            # Q: no gradient
        ln1_out.detach(),            # K: no gradient
        ln1_out,                     # V: gradient flows through here
    )

    # swap_backward: forward value, backward gradient
    attn_out = attn_fwd + (attn_bwd - attn_bwd.detach())

    x = x + attn_out
    x = x + self.mlp(self.ln_2(x))
    x = x.permute(1, 0, 2)          # [T, B, D] → [B, T, D]
    return x


class _RestoreHandle:
    """
    Fake RemovableHandle that restores a module's forward method.
    Used by HookManager.remove_all() to undo forward method swaps.
    """
    def __init__(self, layer: nn.Module, original_forward):
        self._layer    = layer
        self._original = original_forward

    def remove(self):
        self._layer.forward = self._original


def install_attention_hooks(model: nn.Module) -> HookManager:
    """
    Install LibraGrad attention fix on all ResidualAttentionBlocks
    in both vision_model and text_model.

    Swaps permute_then_forward with _libra_permute_then_forward.
    Returns a HookManager whose remove_all() restores originals.
    """
    manager = HookManager()

    for submodel in [model.vision_model, model.text_model]:
        for layer in submodel.transformer.resblocks:
            original_forward = layer.forward
            layer.forward    = partial(_libra_permute_then_forward, layer)
            manager.add(_RestoreHandle(layer, original_forward))

    return manager


# ============================================================
# Install / Remove Helpers
# ============================================================

def install_layernorm_hooks(model: nn.Module) -> HookManager:
    """
    Install LibraGrad backward hooks on all LayerNorm modules.
    """
    manager = HookManager()

    for name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm):
            handle = module.register_full_backward_hook(
                layernorm_backward_hook
            )
            manager.add(handle)

    return manager


def install_libragrad_hooks(model: nn.Module) -> HookManager:
    """
    Install ALL LibraGrad hooks:
        - LayerNorm backward scaling (both vision and text)
        - Attention V-only forward method swap (both vision and text)

    Returns a combined HookManager.
    """
    manager = HookManager()

    # LayerNorm backward hooks
    for submodel in [model.vision_model, model.text_model]:
        for name, module in submodel.named_modules():
            if isinstance(module, nn.LayerNorm):
                handle = module.register_full_backward_hook(
                    layernorm_backward_hook
                )
                manager.add(handle)

    # Attention forward method swaps
    attn_manager = install_attention_hooks(model)
    manager.handles.extend(attn_manager.handles)

    return manager


def remove_libragrad_hooks(manager: HookManager):
    """
    Remove all LibraGrad hooks and restore original forward methods.
    """
    manager.remove_all()


# ============================================================
# Activation + Gradient Recorder
# ============================================================

class ActivationGradientRecorder:
    """
    Records activations (forward) and gradients (backward)
    at a target module for saliency computation.

    Usage:
        recorder = ActivationGradientRecorder(target_layer)
        recorder.install()
        output = model(input)
        loss.backward()
        saliency = recorder.get_saliency()
        recorder.remove()
    """

    def __init__(self, target_module: nn.Module):
        self.target_module  = target_module
        self.activations    = None
        self.gradients      = None
        self._forward_handle  = None
        self._backward_handle = None

    def _forward_hook(self, module, inputs, output):
        if isinstance(output, tuple):
            act = output[0]
        else:
            act = output
        self.activations = act.detach().clone()
        act.retain_grad()

    def _backward_hook(self, module, grad_input, grad_output):
        if grad_output[0] is not None:
            self.gradients = grad_output[0].detach().clone()

    def install(self):
        self._forward_handle = self.target_module.register_forward_hook(
            self._forward_hook
        )
        self._backward_handle = self.target_module.register_full_backward_hook(
            self._backward_hook
        )

    def remove(self):
        if self._forward_handle is not None:
            self._forward_handle.remove()
            self._forward_handle = None
        if self._backward_handle is not None:
            self._backward_handle.remove()
            self._backward_handle = None

    def get_saliency(self) -> Optional[torch.Tensor]:
        """
        Compute saliency as sum(activation * gradient, dim=-1).
        Returns [seq_len] tensor or None if not ready.
        """
        if self.activations is None or self.gradients is None:
            return None

        act  = self.activations
        grad = self.gradients

        if act.shape != grad.shape:
            return None

        saliency = (act.float() * grad.float()).abs().sum(dim=-1)
        return saliency[0].detach()

    def is_ready(self) -> bool:
        return self.activations is not None and self.gradients is not None

    def reset(self):
        self.activations = None
        self.gradients   = None

    def __repr__(self):
        return (
            f"ActivationGradientRecorder("
            f"ready={self.is_ready()}, "
            f"module={type(self.target_module).__name__})"
        )


# ============================================================
# Forward Verification Utility
# ============================================================

def verify_hooks_dont_change_forward(
    model: nn.Module,
    dummy_image: torch.Tensor,
    dummy_text: torch.Tensor,
    atol: float = 1e-4,
) -> bool:
    """
    Verify LibraGrad hooks do not affect the forward pass outputs.
    """
    with torch.no_grad():
        img_before = model.get_image_features(dummy_image).clone()
        txt_before = model.get_text_features(dummy_text).clone()

    manager = install_libragrad_hooks(model)

    with torch.no_grad():
        img_after = model.get_image_features(dummy_image).clone()
        txt_after = model.get_text_features(dummy_text).clone()

    remove_libragrad_hooks(manager)

    img_ok = torch.allclose(img_before, img_after, atol=atol)
    txt_ok = torch.allclose(txt_before, txt_after, atol=atol)

    if img_ok and txt_ok:
        print("[hooks] Forward pass unchanged. Verification PASSED.")
    else:
        print(f"[hooks] WARNING: Forward changed! img_ok={img_ok}, txt_ok={txt_ok}")

    return img_ok and txt_ok