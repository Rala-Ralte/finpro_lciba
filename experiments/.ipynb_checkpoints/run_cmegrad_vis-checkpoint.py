"""
experiments/run_cmegrad_vis.py

Generate vision + text heatmaps for a random subset of the
500-sample CMAES evaluation set.

Loads the saved cmaes_500.json to reuse the same QA pairs,
re-runs CMEGradLXMERT.generate() on each, and saves side-by-side
vision heatmap + text attribution figures.

Usage:
    cd /media/crk/vol3/rt/lciba
    python experiments/run_cmegrad_vis.py
    python experiments/run_cmegrad_vis.py --n 50
    python experiments/run_cmegrad_vis.py --n 20 --seed 7
"""

import os
import json
import random
import argparse
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image

from scripts.utils import load_config, setup_output_dirs
from scripts.cmegrad.features import (
    build_feature_index,
    load_vqa_annotations,
    load_coco_image,
)
from scripts.cmegrad.model import (
    load_lxmert,
    load_libra_lxmert,
    load_id2label,
)
from scripts.cmegrad.attribution import CMEGradLXMERT
from scripts.cmegrad.visualization import visualize_cmegrad

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):

    cfg  = load_config(args.config)
    dirs = setup_output_dirs(cfg["outputs"]["base_dir"])

    random.seed(args.seed)

    print("=" * 60)
    print(f"CMEGrad Visualization  [{args.n} samples]")
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
        model_name = cfg["model"]["lxmert_name"],
        hf_cache   = cfg["model"]["hf_cache"],
        device     = device,
        verify     = False,
        base_model = model,
        tokenizer  = tokenizer,
    )

    # ── Load data ─────────────────────────────────────────────
    feat_index = build_feature_index(cfg["data"]["feat_tsv"])

    questions, _ = load_vqa_annotations(
        cfg["data"]["vqa_q_path"],
        cfg["data"]["vqa_a_path"],
        cfg["data"]["coco_val_dir"],
    )

    # Use the same seed + filter as run_cmegrad.py so samples match
    random.seed(cfg["data"]["seed"])
    all_qa  = [q for q in questions if q["answer"] in id2label.values()]
    eval_qa = random.sample(all_qa, min(500, len(all_qa)))

    # Now pick args.n from that set with args.seed
    random.seed(args.seed)
    subset = random.sample(eval_qa, min(args.n, len(eval_qa)))

    print(f"  Subset: {len(subset)} samples")

    # ── CMEGrad instance ──────────────────────────────────────
    cmegrad = CMEGradLXMERT(
        model      = libra_model,
        tokenizer  = tokenizer,
        id2label   = id2label,
        feat_index = feat_index,
        feat_tsv   = cfg["data"]["feat_tsv"],
        device     = device,
    )

    # ── Output directory ──────────────────────────────────────
    vis_dir = os.path.join(cfg["outputs"]["base_dir"], "heatmaps")
    os.makedirs(vis_dir, exist_ok=True)

    # ── Visualization loop ────────────────────────────────────
    errors = 0

    for i, qa in enumerate(tqdm(subset, desc="Visualizing")):
        try:
            img_pil = load_coco_image(
                qa["image_id"],
                cfg["data"]["coco_val_dir"],
            )

            lang_r, vis_r, info = cmegrad.generate(
                qa["question"], qa["image_id"]
            )

            # Attach question to info for figure title
            info["question"] = qa["question"]

            save_path = os.path.join(
                vis_dir,
                f"cmeg_{str(qa['image_id']).zfill(12)}_{i:04d}.png",
            )

            visualize_cmegrad(
                img_pil      = img_pil,
                vis_attr_raw = info["vis_attr_raw"],
                text_attr    = lang_r,
                boxes        = info["boxes"],
                tokens       = info["tokens"],
                info         = info,
                save_path    = save_path,
            )

            print(
                f"  [{i+1:03d}] {qa['question'][:45]:45s} "
                f"→ {info['pred_answer']:10s} "
                f"H_gap={info['H_gap']:.3f}"
            )

        except Exception as e:
            errors += 1
            print(f"  [WARNING] Sample {i} failed: {e}")
            continue

    print(f"\n  Saved: {len(subset) - errors}/{len(subset)}")
    print(f"  Errors: {errors}")
    print(f"  Output: {vis_dir}")
    print("✓ run_cmegrad_vis.py complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("CMEGrad Visualization")

    parser.add_argument(
        "--config", type=str, default="configs/cmegrad.yaml",
    )
    parser.add_argument(
        "--n", type=int, default=20,
        help="Number of samples to visualize",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed for subset selection",
    )

    args = parser.parse_args()
    main(args)