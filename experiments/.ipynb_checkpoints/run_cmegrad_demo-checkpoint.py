"""
experiments/run_cmegrad_demo.py

Generate CMEGrad visualization for the 3 demo examples.

Saves heatmap PNGs to outputs/cmegrad/heatmaps/.

Usage:
    cd /media/crk/vol3/rt/lciba
    python experiments/run_cmegrad_demo.py
    python experiments/run_cmegrad_demo.py --config configs/cmegrad.yaml
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # ← must be FIRST, before torch

import argparse
import torch

from scripts.utils import load_config, setup_output_dirs
from scripts.cmegrad.features import build_feature_index, load_coco_image
from scripts.cmegrad.model import load_lxmert, load_libra_lxmert, load_id2label
from scripts.cmegrad.attribution import CMEGradLXMERT
from scripts.cmegrad.visualization import visualize_cmegrad

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
    cfg  = load_config(args.config)
    dirs = setup_output_dirs(cfg["outputs"]["base_dir"])

    print("=" * 60)
    print("CMEGrad Demo Visualization")
    print("=" * 60)

    # Load libra_model only — skip base model to save memory
    libra_model, tokenizer = load_lxmert(
        model_name     = cfg["model"]["lxmert_name"],
        tokenizer_name = cfg["model"]["tokenizer_name"],
        hf_cache       = cfg["model"]["hf_cache"],
        device         = device,
    )
    id2label = load_id2label(libra_model)

    from scripts.cmegrad.model import patch_layernorms, patch_gelu
    n_ln   = patch_layernorms(libra_model)
    n_gelu = patch_gelu(libra_model)
    print(f"  LibraLayerNorm: {n_ln}  LibraGELU: {n_gelu}")
    # ── Load features ─────────────────────────────────────────
    feat_index = build_feature_index(cfg["data"]["feat_tsv"])

    cmegrad = CMEGradLXMERT(
        model      = libra_model,
        tokenizer  = tokenizer,
        id2label   = id2label,
        feat_index = feat_index,
        feat_tsv   = cfg["data"]["feat_tsv"],
        device     = device,
    )

    # ── Demo examples ─────────────────────────────────────────
    examples = cfg["demo"]["examples"]
    vis_cfg  = cfg["visualization"]

    os.makedirs(cfg["outputs"]["heatmaps_dir"], exist_ok=True)

    for ex in examples:
        image_id = ex["image_id"]
        question = ex["question"]

        print(f"\nProcessing: {question}")

        img_pil = load_coco_image(image_id, cfg["data"]["coco_val_dir"])
        lang_r, vis_r, info = cmegrad.generate(question, image_id)

        info["question"] = question

        img_id    = str(image_id).zfill(12)
        save_path = os.path.join(
            cfg["outputs"]["heatmaps_dir"],
            f"cmeg_{img_id}.png",
        )

        visualize_cmegrad(
            img_pil      = img_pil,
            vis_attr_raw = info["vis_attr_raw"],
            text_attr    = lang_r,
            boxes        = info["boxes"],
            tokens       = info["tokens"],
            info         = info,
            save_path    = save_path,
            show         = False,
            sigma        = vis_cfg["sigma"],
            top_k        = vis_cfg["top_k"],
            max_width    = vis_cfg["max_width"],
            max_height   = vis_cfg["max_height"],
            fontsize     = vis_cfg["fontsize"],
            dpi          = vis_cfg["dpi"],
        )

        print(f"  pred:   {info['pred_answer']}")
        print(f"  H_vis:  {info['H_vis']:.4f}  H_lang: {info['H_lang']:.4f}"
              f"  H_gap: {info['H_gap']:.4f}")
        print(f"  ✓ saved → {save_path}")

    print("\n✓ run_cmegrad_demo.py complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("CMEGrad Demo")
    parser.add_argument(
        "--config", type=str, default="configs/cmegrad.yaml"
    )
    args = parser.parse_args()
    main(args)