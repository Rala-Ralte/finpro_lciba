"""
experiments/run_baseline.py

Run original M2IB attribution and evaluation.
This is the baseline all other methods are compared against.

Usage:
    cd /media/crk/vol3/rt/lciba
    python experiments/run_baseline.py
    python experiments/run_baseline.py --samples 50 --progbar
    python experiments/run_baseline.py --config configs/default.yaml
"""

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
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
from scripts.methods import vision_heatmap_iba, text_heatmap_iba
from scripts.visualization import visualize_single

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Metrics
# ============================================================

def get_metrics(image_feat, vmap, text_ids, tmap, model):
    """
    Compute DropInConfidence and IncreaseInConfidence
    for vision and text attribution maps.

    Args:
        image_feat: [1, 3, 224, 224] image tensor
        vmap:       np.ndarray [224, 224] vision saliency
        text_ids:   [1, seq_len] token tensor
        tmap:       np.ndarray [seq_len] text saliency
        model:      ClipWrapper instance

    Returns:
        dict with keys: vdrop, vincr, tdrop, tincr
    """
    results = {}

    with torch.no_grad():
        vtargets = [CosSimilarity(
            model.get_text_features(text_ids).to(device)
        )]
        ttargets = [CosSimilarity(
            model.get_image_features(image_feat).to(device)
        )]

    # Remove SOT and EOT tokens for text evaluation
    text_ids_eval = text_ids[:, 1:-1]
    tmap_eval     = np.expand_dims(tmap, axis=0)[:, 1:-1]

    # Binarize text map at 50th percentile
    tmap_eval = tmap_eval > np.percentile(tmap_eval, 50)

    vmap_eval = np.expand_dims(vmap, axis=(0, 1))   # [1, 1, 224, 224]

    results["vdrop"] = DropInConfidence()(
        image_feat, vmap_eval, vtargets,
        ImageFeatureExtractor(model)
    )[0][0] * 100

    results["vincr"] = IncreaseInConfidence()(
        image_feat, vmap_eval, vtargets,
        ImageFeatureExtractor(model)
    )[0][0] * 100

    results["tdrop"] = DropInConfidence()(
        text_ids_eval, tmap_eval, ttargets,
        TextFeatureExtractor(model)
    )[0][0] * 100

    results["tincr"] = IncreaseInConfidence()(
        text_ids_eval, tmap_eval, ttargets,
        TextFeatureExtractor(model)
    )[0][0] * 100

    return results


# ============================================================
# Main
# ============================================================

def main(args):

    # ----------------------------------------------------------
    # Setup
    # ----------------------------------------------------------
    cfg  = load_config(args.config)
    dirs = setup_output_dirs(cfg["outputs"]["base_dir"])

    print("=" * 60)
    print("M2IB BASELINE")
    print("=" * 60)

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
    # IBA hyperparameters
    # ----------------------------------------------------------
    iba_cfg = cfg["iba"]

    # ----------------------------------------------------------
    # Meters
    # ----------------------------------------------------------
    meters = {k: AverageMeter(k) for k in ["vdrop", "vincr", "tdrop", "tincr"]}

    # ----------------------------------------------------------
    # Evaluation loop
    # ----------------------------------------------------------
    print("\nRunning M2IB baseline ...")

    for i, row in enumerate(tqdm(data, desc="M2IB")):
        text       = row[0]
        image_path = row[1]

        try:
            if not isinstance(text, str) or not isinstance(image_path, str):
                continue
            if image_path.startswith("http"):
                image_tensor, pil_image = load_image_from_url(
                    image_path, preprocess, device
                )
            else:
                image_tensor, pil_image = load_image(
                    image_path, preprocess, device
                )

            # Tokenize text
            text_ids = load_text(text, device)

            # Compute heatmaps
            vmap = vision_heatmap_iba(
                text_ids, image_tensor, model,
                layer_idx   = iba_cfg["vlayer"],
                beta        = iba_cfg["vbeta"],
                var         = iba_cfg["vvar"],
                lr          = iba_cfg["vlr"],
                train_steps = iba_cfg["vsteps"],
                progbar     = args.progbar,
            )

            tmap = text_heatmap_iba(
                text_ids, image_tensor, model,
                layer_idx   = iba_cfg["tlayer"],
                beta        = iba_cfg["tbeta"],
                var         = iba_cfg["tvar"],
                lr          = iba_cfg["tlr"],
                train_steps = iba_cfg["tsteps"],
                progbar     = args.progbar,
            )

            # Compute metrics
            metrics = get_metrics(
                image_tensor, vmap, text_ids, tmap, model
            )

            # Update meters
            for k, v in metrics.items():
                meters[k].update(v)

            # Save heatmap
            if cfg["outputs"]["save_heatmaps"]:
                image_np  = pil_to_numpy(pil_image)
                tokens    = decode_tokens(text_ids)
                save_path = os.path.join(
                    dirs["heatmaps"], f"m2ib_{i:04d}.png"
                )
                visualize_single(
                    image_np, vmap, tmap, tokens,
                    title     = f"M2IB | {text[:50]}",
                    save_path = save_path,
                )

            # Save row to CSV
            row_results = {
                "method": "m2ib",
                "image":  image_path,
                "text":   text,
                **metrics,
            }
            save_results_csv(row_results, cfg["outputs"]["results_csv"])

        except Exception as e:
            print(f"\n[WARNING] Sample {i} failed: {e}")
            continue

    # ----------------------------------------------------------
    # Print summary
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("M2IB BASELINE RESULTS")
    print("=" * 60)

    for k, meter in meters.items():
        print(f"  {k:8s}: {meter.avg:.4f}")


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser("M2IB Baseline")

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Override data path from config",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Override number of samples",
    )
    parser.add_argument(
        "--progbar",
        action="store_true",
        help="Show per-sample IBA progress bar",
    )

    args = parser.parse_args()

    # Apply CLI overrides
    cfg = load_config(args.config)

    if args.data_path is None:
        args.data_path = cfg["data"]["data_path"]
    if args.samples is None:
        args.samples = cfg["data"]["samples"]

    main(args)