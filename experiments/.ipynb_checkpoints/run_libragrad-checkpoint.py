"""
experiments/run_libragrad.py

Run LibraGrad-only attribution and evaluation.
Option C standalone: no IB optimization, single forward+backward.

Usage:
    cd /media/crk/vol3/rt/lciba
    python experiments/run_libragrad.py
    python experiments/run_libragrad.py --samples 50
"""

import os
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from random import sample

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
)
from scripts.methods import vision_heatmap_libragrad, text_heatmap_libragrad
from scripts.visualization import visualize_single
from experiments.run_baseline import get_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):

    cfg  = load_config(args.config)
    dirs = setup_output_dirs(cfg["outputs"]["base_dir"])

    print("=" * 60)
    print("LIBRAGRAD ONLY  [Option C standalone]")
    print("=" * 60)

    model, preprocess = load_model(
        clip_model_name=cfg["model"]["clip_variant"],
        device=device,
    )

    print(f"\nLoading data from: {args.data_path}")

    df   = pd.read_csv(args.data_path, sep="\t")
    data = list(df.itertuples(index=False))
    data = sample(data, min(args.samples, len(data)))

    print(f"Samples: {len(data)}")

    iba_cfg = cfg["iba"]
    meters  = {k: AverageMeter(k) for k in ["vdrop", "vincr", "tdrop", "tincr"]}

    for i, row in enumerate(tqdm(data, desc="LibraGrad")):
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

            text_ids = load_text(text, device)

            vmap = vision_heatmap_libragrad(
                text_ids, image_tensor, model,
                layer_idx=iba_cfg["vlayer"],
            )

            tmap = text_heatmap_libragrad(
                text_ids, image_tensor, model,
                layer_idx=iba_cfg["tlayer"],
            )

            metrics = get_metrics(
                image_tensor, vmap, text_ids, tmap, model
            )

            for k, v in metrics.items():
                meters[k].update(v)

            if cfg["outputs"]["save_heatmaps"]:
                image_np = (
                    image_tensor[0]
                    .detach()
                    .cpu()
                    .numpy()
                )
                
                image_np = (
                    image_np.transpose(1,2,0)
                )
                
                image_np = (
                    image_np * np.array([
                        0.26862954,
                        0.26130258,
                        0.27577711
                    ])
                    +
                    np.array([
                        0.48145466,
                        0.4578275,
                        0.40821073
                    ])
                )
                
                image_np = np.clip(
                    image_np,
                    0,
                    1
                )
                tokens    = decode_tokens(text_ids)
                save_path = os.path.join(
                    dirs["heatmaps"], f"libragrad_{i:04d}.png"
                )
                visualize_single(
                    image_np, vmap, tmap, tokens,
                    title     = f"LibraGrad | {text[:50]}",
                    save_path = save_path,
                )

            save_results_csv(
                {"method": "libragrad", "image": image_path,
                 "text": text, **metrics},
                cfg["outputs"]["results_csv"],
            )

        except Exception as e:
            print(f"\n[WARNING] Sample {i} failed: {e}")
            continue

    print("\n" + "=" * 60)
    print("LIBRAGRAD RESULTS")
    print("=" * 60)

    for k, meter in meters.items():
        print(f"  {k:8s}: {meter.avg:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("LibraGrad Only")

    parser.add_argument("--config",    type=str, default="configs/default.yaml")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--samples",   type=int, default=None)

    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.data_path is None:
        args.data_path = cfg["data"]["data_path"]
    if args.samples is None:
        args.samples = cfg["data"]["samples"]

    main(args)