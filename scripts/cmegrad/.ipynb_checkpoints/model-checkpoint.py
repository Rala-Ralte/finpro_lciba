"""
scripts/cmegrad/model.py

LXMERT model loading and LibraGrad patching.

Responsibilities:
    - Load base LXMERT and LibraGrad-patched LXMERT
    - LibraLayerNorm, LibraGELU operators
    - patch_layernorms(), patch_gelu()
    - Forward pass verification
    - VQA answer label loading
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import requests
from transformers import LxmertForQuestionAnswering, LxmertTokenizer
from transformers.activations import GELUActivation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Global LibraGrad mode flag ────────────────────────────────
# Set True to activate LibraGrad corrections in forward pass.
global_libra_mode = True


def set_libra_mode(enabled: bool):
    """Toggle LibraGrad corrections on/off for all patched modules."""
    global global_libra_mode
    global_libra_mode = enabled


# ============================================================
# LibraGrad Primitives
# ============================================================

def swap_backward(forward, backward):
    """Use forward values, route gradient through backward path."""
    return forward.detach() + (backward - backward.detach())


def constant_op(x):
    """Identity in forward, zero gradient in backward."""
    return x.detach()


# ============================================================
# LibraLayerNorm
# ============================================================

class LibraLayerNorm(nn.LayerNorm):
    """
    LayerNorm with constant_op on denominator.
    Prevents gradient imbalance from std-dev division.
    """

    @classmethod
    def from_layernorm(cls, ln: nn.LayerNorm) -> "LibraLayerNorm":
        new = cls(
            ln.normalized_shape,
            eps=ln.eps,
            elementwise_affine=ln.elementwise_affine,
        )
        if ln.elementwise_affine:
            new.weight = ln.weight
            new.bias   = ln.bias
        return new

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not global_libra_mode:
            return super().forward(x)
        mean  = x.mean(dim=-1, keepdim=True)
        var   = x.var(dim=-1, keepdim=True, unbiased=False)
        denom = constant_op(torch.rsqrt(var + self.eps))
        x_n   = (x - mean.detach()) * denom
        if self.elementwise_affine:
            x_n = x_n * self.weight + self.bias
        return x_n


# ============================================================
# LibraGELU
# ============================================================

class LibraGELU(nn.Module):
    """
    GELU with swap_backward on sigmoid gate.
    Forward: exact F.gelu. Backward: detached approximate gate.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not global_libra_mode:
            return F.gelu(x)
        gate = constant_op(torch.sigmoid(1.702 * x))
        return swap_backward(F.gelu(x), x * gate)


# ============================================================
# Patch Functions
# ============================================================

def patch_layernorms(m: nn.Module) -> int:
    """
    Replace all nn.LayerNorm modules with LibraLayerNorm.
    Returns count of replacements.
    """
    n = 0
    for name, module in list(m.named_modules()):
        for attr, child in list(module.named_children()):
            if isinstance(child, nn.LayerNorm):
                setattr(module, attr, LibraLayerNorm.from_layernorm(child))
                n += 1
    return n


def patch_gelu(m: nn.Module) -> int:
    """
    Replace all GELUActivation / nn.GELU modules with LibraGELU.
    Returns count of replacements.
    """
    n = 0
    for name, module in list(m.named_modules()):
        for attr, child in list(module.named_children()):
            if isinstance(child, (GELUActivation, nn.GELU)):
                setattr(module, attr, LibraGELU())
                n += 1
    return n


# ============================================================
# Model Loading
# ============================================================

def load_lxmert(
    model_name: str = "unc-nlp/lxmert-vqa-uncased",
    tokenizer_name: str = "unc-nlp/lxmert-base-uncased",
    hf_cache: str = None,
    device: torch.device = None,
):
    """
    Load base LXMERT model and tokenizer.

    Returns:
        model:     LxmertForQuestionAnswering (eval mode)
        tokenizer: LxmertTokenizer
    """
    import os

    if hf_cache:
        os.environ["HF_HOME"]            = hf_cache
        os.environ["TRANSFORMERS_CACHE"] = hf_cache

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading {model_name}...")
    tokenizer = LxmertTokenizer.from_pretrained(tokenizer_name)
    model     = LxmertForQuestionAnswering.from_pretrained(
        model_name
    ).to(device)
    model.eval()

    print(f"  Parameters:      {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Hidden size:     {model.config.hidden_size}")
    print(f"  Attention heads: {model.config.num_attention_heads}")
    print(f"  Layers:          {model.config.num_hidden_layers}")
    print(f"  Cross-attention: "
          f"{len(model.lxmert.encoder.x_layers)} x LxmertXLayer")

    return model, tokenizer


def load_libra_lxmert(
    model_name: str = "unc-nlp/lxmert-vqa-uncased",
    hf_cache: str = None,
    device: torch.device = None,
    verify: bool = True,
    base_model=None,
    tokenizer=None,
) -> LxmertForQuestionAnswering:
    """
    Load LXMERT with LibraGrad patches applied.

    Applies:
        - 57 LibraLayerNorm replacements
        - 24 LibraGELU replacements

    Args:
        base_model: If provided, uses this model for forward verification.
        verify:     Run forward pass diff check after patching.

    Returns:
        libra_model: patched LxmertForQuestionAnswering (eval mode)
    """
    import os

    if hf_cache:
        os.environ["HF_HOME"]            = hf_cache
        os.environ["TRANSFORMERS_CACHE"] = hf_cache

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    libra_model = LxmertForQuestionAnswering.from_pretrained(
        model_name
    ).to(device)
    libra_model.eval()

    n_ln   = patch_layernorms(libra_model)
    n_gelu = patch_gelu(libra_model)

    print(f"  LibraLayerNorm patches: {n_ln}")
    print(f"  LibraGELU patches:      {n_gelu}")

    if verify and base_model is not None and tokenizer is not None:
        verify_libra_model(base_model, libra_model, tokenizer, device)

    return libra_model


# ============================================================
# VQA Answer Labels
# ============================================================

def load_id2label(model: LxmertForQuestionAnswering) -> dict:
    """
    Load VQA answer vocabulary.

    Tries model.config.id2label first; downloads from
    airsplay/lxmert GitHub if config has fewer than 100 labels.

    Returns:
        id2label: dict {int -> str}
    """
    try:
        id2label = model.config.id2label
        if len(id2label) >= 100:
            print(f"  id2label from config: {len(id2label)} labels")
            return id2label
    except Exception:
        pass

    print("  Downloading VQA answer labels...")
    url      = (
        "https://raw.githubusercontent.com/airsplay/lxmert/"
        "master/data/vqa/trainval_label2ans.json"
    )
    ans_list = requests.get(url, timeout=10).json()
    id2label = {i: ans for i, ans in enumerate(ans_list)}
    print(f"  id2label downloaded: {len(id2label)} labels")
    return id2label


# ============================================================
# Forward Verification
# ============================================================

def verify_libra_model(
    base_model: LxmertForQuestionAnswering,
    libra_model: LxmertForQuestionAnswering,
    tokenizer: LxmertTokenizer,
    device: torch.device,
    question: str = "how many cats are there?",
    feats: torch.Tensor = None,
    boxes: torch.Tensor = None,
    atol: float = 0.01,
) -> bool:
    """
    Verify LibraGrad patches preserve model predictions.
    Uses dummy visual features if feats/boxes not provided.
    """
    global global_libra_mode

    enc = tokenizer(
        question,
        padding="max_length", max_length=20,
        truncation=True, return_tensors="pt",
    ).to(device)

    if feats is None:
        feats = torch.zeros(1, 36, 2048).to(device)
        boxes = torch.zeros(1, 36, 4).to(device)
    else:
        feats = feats.to(device)
        boxes = boxes.to(device)

    kwargs = dict(
        input_ids      = enc["input_ids"],
        attention_mask = enc["attention_mask"],
        token_type_ids = enc["token_type_ids"],
        visual_feats   = feats,
        visual_pos     = boxes,
    )

    global_libra_mode = False
    with torch.no_grad():
        out_std = base_model(**kwargs)

    global_libra_mode = True
    with torch.no_grad():
        out_lib = libra_model(**kwargs)

    diff = (
        out_std.question_answering_score -
        out_lib.question_answering_score
    ).abs().max().item()

    ok = diff < atol
    print(f"  Max logit diff: {diff:.6f}  {'✓' if ok else '✗ WARN'}")

    if not ok:
        print(f"  WARNING: logit drift {diff:.4f} > {atol}")

    return ok