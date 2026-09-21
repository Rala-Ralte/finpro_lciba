"""
experiments/run_cmegrad.py

500-sample CMAES evaluation for CMEGrad (LXMERT).

Compares CMEGrad against the LibraIxG baseline (uniform vision attribution).

Usage:
    cd /media/crk/vol3/rt/lciba
    python experiments/run_cmegrad.py
    python experiments/run_cmegrad.py --samples 50
    python experiments/run_cmegrad.py --config configs/cmegrad.yaml
"""

import os
import sys
import json
import random
import argparse
import torch
import numpy as np
from tqdm import tqdm

from scripts.utils import load_config, setup_output_dirs, AverageMeter
from scripts.cmegrad.features import (
    build_feature_index,
    load_vqa_annotations,
)
from scripts.cmegrad.model import (
    load_lxmert,
    load_libra_lxmert,
    load_id2label,
)
from scripts.cmegrad.metrics import cmaes, print_cmaes_report
from scripts.cmegrad.attribution import CMEGradLXMERT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Baseline
# ============================================================

def baseline_cmaes(cmegrad_instance, question, image_id):
    """
    LibraGrad IxG text + uniform vision (no bridge, no refinement).
    Used as the comparison baseline for H_gap.
    """
    from scripts.cmegrad.features import load_image_features_fast
    from scripts.cmegrad.metrics import entropy_weight

    feats, boxes = load_image_features_fast(
        image_id,
        cmegrad_instance.feat_index,
        cmegrad_instance.feat_tsv,
    )
    enc = cmegrad_instance.tokenizer(
        question, padding="max_length", max_length=20,
        truncation=True, return_tensors="pt",
    ).to(device)
    feats = feats.to(device)
    boxes = boxes.to(device)

    text_attr, _ = cmegrad_instance._text_attr(enc, feats, boxes)
    vis_attr      = torch.ones(36) / 36    # uniform vision

    _, H_vis  = entropy_weight(vis_attr)
    _, H_lang = entropy_weight(text_attr)
    H_gap     = abs(H_vis - H_lang)

    return H_vis, H_lang, H_gap


# ============================================================
# Main
# ============================================================

def main(args):

    cfg  = load_config(args.config)
    dirs = setup_output_dirs(cfg["outputs"]["base_dir"])

    random.seed(cfg["data"]["seed"])

    print("=" * 60)
    print("CMEGrad CMAES Evaluation")
    print("=" * 60)

    # ── Load model ────────────────────────────────────────────
    model, tokenizer = load_lxmert(
        model_name     = cfg["model"]["lxmert_name"],
        tokenizer_name = cfg["model"]["tokenizer_name"],
        hf_cache       = cfg["model"]["hf_cache"],
        device         = device,
    )
    id2label = load_id2label(model)

    libra_model = load_libra_lxmert(
        model_name  = cfg["model"]["lxmert_name"],
        hf_cache    = cfg["model"]["hf_cache"],
        device      = device,
        verify      = True,
        base_model  = model,
        tokenizer   = tokenizer,
    )

    # ── Load data ─────────────────────────────────────────────
    print(f"\nLoading feature index...")
    feat_index = build_feature_index(cfg["data"]["feat_tsv"])
    print(f"✓ Feature index: {len(feat_index):,} images")

    print(f"\nLoading VQA annotations...")
    questions, _ = load_vqa_annotations(
        cfg["data"]["vqa_q_path"],
        cfg["data"]["vqa_a_path"],
        cfg["data"]["coco_val_dir"],
    )
    print(f"  VQA pairs: {len(questions):,}")

    # Filter to answers in vocabulary
    all_qa  = [q for q in questions if q["answer"] in id2label.values()]
    n       = min(args.samples, len(all_qa))
    eval_qa = random.sample(all_qa, n)
    print(f"  Evaluation set: {len(eval_qa)} QA pairs")

    # ── CMEGrad instance ──────────────────────────────────────
    cmegrad = CMEGradLXMERT(
        model      = libra_model,
        tokenizer  = tokenizer,
        id2label   = id2label,
        feat_index = feat_index,
        feat_tsv   = cfg["data"]["feat_tsv"],
        device     = device,
    )

    # ── Metrics storage ───────────────────────────────────────
    results = {
        "cmeg":     {"H_vis": [], "H_lang": [], "H_gap": [], "w_vis": []},
        "baseline": {"H_vis": [], "H_lang": [], "H_gap": []},
    }
    errors = 0

    # ── Evaluation loop ───────────────────────────────────────
    print(f"\nRunning evaluation on {len(eval_qa)} samples...")

    for qa in tqdm(eval_qa, desc="CMEGrad eval"):
        try:
            _, _, info = cmegrad.generate(qa["question"], qa["image_id"])
            results["cmeg"]["H_vis"].append(info["H_vis"])
            results["cmeg"]["H_lang"].append(info["H_lang"])
            results["cmeg"]["H_gap"].append(info["H_gap"])
            results["cmeg"]["w_vis"].append(info["w_vis"])

            H_vis_b, H_lang_b, H_gap_b = baseline_cmaes(
                cmegrad, qa["question"], qa["image_id"]
            )
            results["baseline"]["H_vis"].append(H_vis_b)
            results["baseline"]["H_lang"].append(H_lang_b)
            results["baseline"]["H_gap"].append(H_gap_b)

        except Exception as e:
            errors += 1
            continue

    print(f"\n  Completed: {len(results['cmeg']['H_vis'])}/{len(eval_qa)}")
    print(f"  Errors:    {errors}")

    # ── Report ────────────────────────────────────────────────
    print_cmaes_report(results["cmeg"], results["baseline"])

    # ── Save ──────────────────────────────────────────────────
    os.makedirs(cfg["outputs"]["results_dir"], exist_ok=True)

    save = {
        "n_samples":    len(results["cmeg"]["H_vis"]),
        "cmeg":         {k: float(np.mean(v))
                         for k, v in results["cmeg"].items()},
        "baseline":     {k: float(np.mean(v))
                         for k, v in results["baseline"].items()},
        "cmeg_std":     {k: float(np.std(v))
                         for k, v in results["cmeg"].items()},
        "baseline_std": {k: float(np.std(v))
                         for k, v in results["baseline"].items()},
    }

    out_path = cfg["outputs"]["results_json"]
    with open(out_path, "w") as f:
        json.dump(save, f, indent=2)

    print(f"\n  Results saved to {out_path}")
    print("✓ run_cmegrad.py complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("CMEGrad CMAES Evaluation")

    parser.add_argument(
        "--config", type=str, default="configs/cmegrad.yaml"
    )
    parser.add_argument(
        "--samples", type=int, default=None
    )

    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.samples is None:
        args.samples = cfg["data"]["samples"]

    main(args)