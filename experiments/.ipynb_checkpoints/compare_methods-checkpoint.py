"""
experiments/compare_methods.py

Run all four methods on the same samples and produce:
    1. Per-sample comparison grid images
    2. Aggregate metrics CSV
    3. Bar chart comparing all methods

Usage:
    cd /media/crk/vol3/rt/lciba
    python experiments/compare_methods.py
    python experiments/compare_methods.py --samples 50
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
from scripts.methods import VISION_METHODS, TEXT_METHODS
from scripts.visualization import visualize_comparison, plot_metrics_comparison
from experiments.run_baseline import get_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

METRIC_KEYS = ["vdrop", "vincr", "tdrop", "tincr"]


def main(args):

    cfg     = load_config(args.config)
    dirs    = setup_output_dirs(cfg["outputs"]["base_dir"])
    methods = cfg["compare"]["methods"]
    iba_cfg = cfg["iba"]

    print("=" * 60)
    print("METHOD COMPARISON")
    print(f"Methods: {methods}")
    print("=" * 60)

    model, preprocess = load_model(
        clip_model_name=cfg["model"]["clip_variant"],
        device=device,
    )

    df   = pd.read_csv(args.data_path, sep="\t")
    data = list(df.itertuples(index=False))
    data = sample(data, min(args.samples, len(data)))

    print(f"Samples: {len(data)}")

    # Meters per method per metric
    meters = {
        m: {k: AverageMeter(f"{m}_{k}") for k in METRIC_KEYS}
        for m in methods
    }

    for i, row in enumerate(tqdm(data, desc="Comparing")):
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

            text_ids  = load_text(text, device)
            image_np  = pil_to_numpy(pil_image)
            tokens    = decode_tokens(text_ids)

            method_vmaps = {}
            method_tmaps = {}

            for method_name in methods:

                vision_fn = VISION_METHODS[method_name]
                text_fn   = TEXT_METHODS[method_name]

                # Common kwargs — libragrad and fusion ignore extras via **kwargs
                kwargs = dict(
                    layer_idx   = iba_cfg["vlayer"],
                    beta        = iba_cfg["vbeta"],
                    var         = iba_cfg["vvar"],
                    lr          = iba_cfg["vlr"],
                    train_steps = iba_cfg["vsteps"],
                    progbar     = False,
                )

                vmap = vision_fn(text_ids, image_tensor, model, **kwargs)
                tmap = text_fn(
                    text_ids, image_tensor, model,
                    **{**kwargs,
                       "layer_idx":   iba_cfg["tlayer"],
                       "beta":        iba_cfg["tbeta"],
                       "var":         iba_cfg["tvar"],
                       "lr":          iba_cfg["tlr"],
                       "train_steps": iba_cfg["tsteps"],
                    }
                )

                method_vmaps[method_name] = vmap
                method_tmaps[method_name] = tmap

                metrics = get_metrics(
                    image_tensor, vmap, text_ids, tmap, model
                )

                for k, v in metrics.items():
                    meters[method_name][k].update(v)

                save_results_csv(
                    {"method": method_name, "image": image_path,
                     "text": text, **metrics},
                    cfg["outputs"]["results_csv"],
                )

            # Save comparison grid
            if cfg["compare"]["save_comparison_grid"]:
                save_path = os.path.join(
                    dirs["heatmaps"], f"compare_{i:04d}.png"
                )
                visualize_comparison(
                    image_np, method_vmaps, method_tmaps, tokens,
                    title     = text[:60],
                    save_path = save_path,
                )

        except Exception as e:
            print(f"\n[WARNING] Sample {i} failed: {e}")
            continue

    # ----------------------------------------------------------
    # Summary table
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)

    agg = {}
    header = f"{'Method':12s}" + "".join(f"  {k:8s}" for k in METRIC_KEYS)
    print(header)
    print("-" * len(header))

    for method_name in methods:
        row_str = f"{method_name:12s}"
        agg[method_name] = {}
        for k in METRIC_KEYS:
            avg = meters[method_name][k].avg
            agg[method_name][k] = avg
            row_str += f"  {avg:8.4f}"
        print(row_str)

    # ----------------------------------------------------------
    # Bar chart
    # ----------------------------------------------------------
    chart_path = os.path.join(dirs["metrics"], "comparison_chart.png")
    plot_metrics_comparison(
        agg,
        metric_keys = METRIC_KEYS,
        title       = "Method Comparison — All Metrics",
        save_path   = chart_path,
    )
    print(f"\nChart saved to: {chart_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Compare Methods")

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