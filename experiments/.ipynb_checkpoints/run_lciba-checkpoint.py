"""
experiments/run_lciba.py

Run LC-IBA attribution and evaluation.
Option A + B: LibraGrad-corrected backward + prior initialization.

Usage:
    cd /media/crk/vol3/rt/lciba
    python experiments/run_lciba.py
    python experiments/run_lciba.py --samples 50 --progbar
    python experiments/run_lciba.py --no_libragrad_init   # Option A only
    python experiments/run_lciba.py --no_libragrad        # Ablation: no hooks
"""

import os
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from random import sample

from pytorch_grad_cam.metrics.cam_mult_image import (
    DropInConfidence,
    IncreaseInConfidence,
)

from scripts.utils import (
    load_model,
    load_image,
    load_image_from_url,
    load_text,
    decode_tokens,
    pil_to_numpy,
    save_results_csv,
    AverageMeter,
    setup_output_dirs,
    load_config,
    CosSimilarity,
    ImageFeatureExtractor,
    TextFeatureExtractor,
)
from scripts.methods import vision_heatmap_lciba, text_heatmap_lciba
from scripts.visualization import visualize_single
from experiments.run_baseline import get_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Main
# ============================================================

def main(args):

    # ----------------------------------------------------------
    # Setup
    # ----------------------------------------------------------
    cfg  = load_config(args.config)
    dirs = setup_output_dirs(cfg["outputs"]["base_dir"])

    # Determine run label for file naming
    if args.no_libragrad:
        run_label = "lciba_ablation_nograd"
    elif args.no_libragrad_init:
        run_label = "lciba_option_a_only"
    else:
        run_label = "lciba_full"

    print("=" * 60)
    print(f"LC-IBA  [{run_label}]")
    print("=" * 60)
    print(f"  use_libragrad  : {not args.no_libragrad}")
    print(f"  libragrad_init : {not args.no_libragrad_init}")

    # ----------------------------------------------------------
    # Load model
    # ----------------------------------------------------------
    model, preprocess = load_model(
        clip_model_name=cfg["model"]["clip_variant"],
        device=device,
    )

    # ----------------------------------------------------------
    # Load data
    # ----------------------------------------------------------
    print(f"\nLoading data from: {args.data_path}")

    df   = pd.read_csv(args.data_path, sep="\t")
    data = list(df.itertuples(index=False))
    data = sample(data, min(args.samples, len(data)))

    print(f"Samples: {len(data)}")

    # ----------------------------------------------------------
    # Hyperparameters
    # ----------------------------------------------------------
    iba_cfg   = cfg["iba"]
    lciba_cfg = cfg["lciba"]

    prior_strength  = lciba_cfg["prior_strength"]
    use_libragrad   = not args.no_libragrad
    libragrad_init  = not args.no_libragrad_init

    # ----------------------------------------------------------
    # Meters
    # ----------------------------------------------------------
    meters = {k: AverageMeter(k) for k in ["vdrop", "vincr", "tdrop", "tincr"]}

    # ----------------------------------------------------------
    # Evaluation loop
    # ----------------------------------------------------------
    print(f"\nRunning {run_label} ...")

    for i, row in enumerate(tqdm(data, desc=run_label)):
        text       = row[0]
        image_path = row[1]

        try:
            if not isinstance(text, str) or not isinstance(image_path, str):
                continue
            # Load image
            if image_path.startswith("http"):
                image_tensor, pil_image = load_image_from_url(
                    image_path, preprocess, device
                )
            else:
                image_tensor, pil_image = load_image(
                    image_path, preprocess, device
                )

            # Tokenize
            text_ids = load_text(text, device)

            # Compute LC-IBA heatmaps
            vmap = vision_heatmap_lciba(
                text_ids, image_tensor, model,
                layer_idx      = iba_cfg["vlayer"],
                beta           = iba_cfg["vbeta"],
                var            = iba_cfg["vvar"],
                lr             = iba_cfg["vlr"],
                train_steps    = iba_cfg["vsteps"],
                progbar        = args.progbar,
                prior_strength = prior_strength,
            )

            tmap = text_heatmap_lciba(
                text_ids, image_tensor, model,
                layer_idx      = iba_cfg["tlayer"],
                beta           = iba_cfg["tbeta"],
                var            = iba_cfg["tvar"],
                lr             = iba_cfg["tlr"],
                train_steps    = iba_cfg["tsteps"],
                progbar        = args.progbar,
                prior_strength = prior_strength,
            )

            # Metrics
            metrics = get_metrics(
                image_tensor, vmap, text_ids, tmap, model
            )

            for k, v in metrics.items():
                meters[k].update(v)

            # Save heatmap
            if cfg["outputs"]["save_heatmaps"]:
                image_np  = pil_to_numpy(pil_image)
                tokens    = decode_tokens(text_ids)
                save_path = os.path.join(
                    dirs["heatmaps"], f"{run_label}_{i:04d}.png"
                )
                visualize_single(
                    image_np, vmap, tmap, tokens,
                    title     = f"LC-IBA | {text[:50]}",
                    save_path = save_path,
                )

            # CSV
            row_results = {
                "method": run_label,
                "image":  image_path,
                "text":   text,
                **metrics,
            }
            save_results_csv(row_results, cfg["outputs"]["results_csv"])

        except Exception as e:
            print(f"\n[WARNING] Sample {i} failed: {e}")
            continue

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"LC-IBA RESULTS  [{run_label}]")
    print("=" * 60)

    for k, meter in meters.items():
        print(f"  {k:8s}: {meter.avg:.4f}")


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser("LC-IBA")

    parser.add_argument(
        "--config", type=str, default="configs/default.yaml"
    )
    parser.add_argument(
        "--data_path", type=str, default=None
    )
    parser.add_argument(
        "--samples", type=int, default=None
    )
    parser.add_argument(
        "--progbar", action="store_true"
    )
    parser.add_argument(
        "--no_libragrad",
        action="store_true",
        help="Disable LibraGrad hooks (ablation: no Option A or B)",
    )
    parser.add_argument(
        "--no_libragrad_init",
        action="store_true",
        help="Disable prior initialization (Option A only, no Option B)",
    )

    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.data_path is None:
        args.data_path = cfg["data"]["data_path"]
    if args.samples is None:
        args.samples = cfg["data"]["samples"]

    main(args)