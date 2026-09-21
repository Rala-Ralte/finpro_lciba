"""
scripts/cmegrad/metrics.py

CMAES and entropy metrics for cross-modal attribution evaluation.

CMAES: Cross-Modal Attribution Entropy Score
    H_gap = |H(vis_attr) - H(lang_attr)|
    Lower H_gap = more balanced cross-modal attribution entropy.

CMAS: Cross-Modal Attribution Similarity (concept-level, pending)
    1 - JSD(Q_vis, Q_lang)
"""

import numpy as np
import torch
import torch.nn.functional as F


# ============================================================
# Entropy
# ============================================================

def entropy(x: torch.Tensor) -> float:
    """
    Compute Shannon entropy of softmax-normalized absolute values.

    H(x) = -Σ softmax(|x|) · log(softmax(|x|))

    Args:
        x: 1D tensor of attribution scores

    Returns:
        H: scalar entropy value
    """
    p = F.softmax(x.abs().float(), dim=-1)
    return -(p * (p + 1e-10).log()).sum().item()


def entropy_weight(attr: torch.Tensor):
    """
    Compute entropy weight: w = 1 / (1 + H).

    Lower entropy → higher weight → more trusted modality.

    Returns:
        w: scalar weight in (0, 1]
        H: scalar entropy
    """
    H = entropy(attr)
    return 1.0 / (1.0 + H), H


# ============================================================
# CMAES
# ============================================================

def cmaes(vis_attr: torch.Tensor, text_attr: torch.Tensor) -> dict:
    """
    Compute CMAES metrics for one sample.

    Args:
        vis_attr:  [V] visual attribution scores
        text_attr: [T] text attribution scores

    Returns:
        dict with keys: H_vis, H_lang, H_gap, w_vis, w_lang
    """
    w_vis,  H_vis  = entropy_weight(vis_attr)
    w_lang, H_lang = entropy_weight(text_attr)
    H_gap          = abs(H_vis - H_lang)

    return {
        "H_vis":  H_vis,
        "H_lang": H_lang,
        "H_gap":  H_gap,
        "w_vis":  w_vis,
        "w_lang": w_lang,
    }


# ============================================================
# Aggregate Report
# ============================================================

def report_cmaes(results: dict) -> dict:
    """
    Compute mean ± std for CMAES metrics over collected samples.

    Args:
        results: dict with lists keyed by metric name
                 e.g. {"H_vis": [...], "H_lang": [...], "H_gap": [...]}

    Returns:
        dict with mean and std per metric
    """
    out = {}
    for key, values in results.items():
        arr       = np.array(values)
        out[key]  = {"mean": float(arr.mean()), "std": float(arr.std())}
    return out


def print_cmaes_report(
    cmeg_results: dict,
    baseline_results: dict,
):
    """
    Print formatted CMAES evaluation report.
    """
    def _row(name, d):
        h_vis  = np.array(d["H_vis"])
        h_lang = np.array(d["H_lang"])
        h_gap  = np.array(d["H_gap"])
        print(f"\n  {name}")
        print(f"    H_vis:  {h_vis.mean():.4f} ± {h_vis.std():.4f}")
        print(f"    H_lang: {h_lang.mean():.4f} ± {h_lang.std():.4f}")
        print(f"    H_gap:  {h_gap.mean():.4f} ± {h_gap.std():.4f}")
        return h_gap.mean()

    print("\n" + "=" * 50)
    print("CMAES Evaluation Results")
    print("=" * 50)

    cmeg_gap = _row("CMEGrad", cmeg_results)
    base_gap = _row("Baseline (uniform vision)", baseline_results)

    improvement = (cmeg_gap - base_gap) / base_gap * 100
    mean_w_vis  = np.mean(cmeg_results.get("w_vis", [0]))

    print(f"\n  H_gap improvement: {improvement:+.1f}%")
    print(f"  Mean w_vis:        {mean_w_vis:.4f}")