"""
experiments/benchmark_layers.py

Sweep transformer layer indices to find the optimal
target layer for LC-IBA attribution.

Runs the configured method (default: lciba) across
a range of layer indices and plots metric vs layer.

Usage:
    cd /media/crk/vol3/rt/lciba
    python experiments/benchmark_layers.py
    python experiments/benchmark_layers.py --method m2ib
    python experiments/benchmark_layers.py --layers 6 7 8 9 10 11
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
    AverageMeter,
    setup_output_dirs,
    load_config,
    save_results_csv,
)
from scripts.methods import VISION_METHODS, TEXT_METHODS
from scripts.visualization import plot_layer_benchmark
from experiments.run_baseline import get_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

METRIC_KEYS = ["vdrop", "vincr", "tdrop", "tincr"]


def main(args):

    cfg     = load_config(args.config)
    dirs    = setup_output_dirs(cfg["outputs"]["base_dir"])
    iba_cfg = cfg["iba"]
    bm_cfg  = cfg["benchmark"]

    method_name  = args.method or bm_cfg["method"]
    layer_range  = args.layers or bm_cfg["layer_range"]
    primary_metric = bm_cfg["metric"]

    print("=" * 60)
    print(f"LAYER BENCHMARK  [{method_name}]")
    print(f"Layers : {layer_range}")
    print(f"Metric : {primary_metric}")
    print("=" * 60)

    model, preprocess = load_model(
        clip_model_name=cfg["model"]["clip_variant"],
        device=device,
    )

    df   = pd.read_csv(args.data_path, sep="\t")
    data = list(df.itertuples(index=False))
    data = sample(data, min(args.samples, len(data)))

    print(f"Samples per layer: {len(data)}")

    vision_fn = VISION_METHODS[method_name]
    text_fn   = TEXT_METHODS[method_name]

    # layer_idx -> metric_key -> AverageMeter
    layer_results = {
        layer: {k: AverageMeter(f"L{layer}_{k}") for k in METRIC_KEYS}
        for layer in layer_range
    }

    for layer_idx in layer_range:

        print(f"\n--- Layer {layer_idx} ---")

        for i, row in enumerate(tqdm(data, desc=f"Layer {layer_idx}")):
            text       = row[0]
            image_path = row[1]

            try:
                if not isinstance(text, str) or not isinstance(image_path, str):
                    continue
                if image_path.startswith("http"):
                    image_tensor, _ = load_image_from_url(
                        image_path, preprocess, device
                    )
                else:
                    image_tensor, _ = load_image(
                        image_path, preprocess, device
                    )

                text_ids = load_text(text, device)

                kwargs = dict(
                    layer_idx   = layer_idx,
                    beta        = iba_cfg["vbeta"],
                    var         = iba_cfg["vvar"],
                    lr          = iba_cfg["vlr"],
                    train_steps = iba_cfg["vsteps"],
                    progbar     = False,
                )

                vmap = vision_fn(text_ids, image_tensor, model, **kwargs)
                tmap = text_fn(text_ids, image_tensor, model, **kwargs)

                metrics = get_metrics(
                    image_tensor, vmap, text_ids, tmap, model
                )

                for k, v in metrics.items():
                    layer_results[layer_idx][k].update(v)

                save_results_csv(
                    {"method": method_name, "layer": layer_idx,
                     "image": image_path, "text": text, **metrics},
                    os.path.join(
                        dirs["metrics"],
                        f"benchmark_{method_name}.csv"
                    ),
                )

            except Exception as e:
                print(f"\n[WARNING] Layer {layer_idx} sample {i} failed: {e}")
                continue

        avg = layer_results[layer_idx][primary_metric].avg
        print(f"  Layer {layer_idx}  {primary_metric}: {avg:.4f}")

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"LAYER BENCHMARK RESULTS  [{method_name}]")
    print("=" * 60)

    header = f"{'Layer':6s}" + "".join(f"  {k:8s}" for k in METRIC_KEYS)
    print(header)
    print("-" * len(header))

    for layer_idx in layer_range:
        row_str = f"{layer_idx:6d}"
        for k in METRIC_KEYS:
            row_str += f"  {layer_results[layer_idx][k].avg:8.4f}"
        print(row_str)

    # Best layer
    best_layer = max(
        layer_range,
        key=lambda l: layer_results[l][primary_metric].avg,
    )
    print(f"\nBest layer for {primary_metric}: {best_layer}")

    # ----------------------------------------------------------
    # Plot
    # ----------------------------------------------------------
    plot_data = {
        method_name: {
            l: layer_results[l][primary_metric].avg
            for l in layer_range
        }
    }

    chart_path = os.path.join(
        dirs["metrics"], f"benchmark_{method_name}_{primary_metric}.png"
    )
    plot_layer_benchmark(
        plot_data,
        metric    = primary_metric,
        title     = f"Layer Benchmark — {method_name} — {primary_metric}",
        save_path = chart_path,
    )
    print(f"Chart saved to: {chart_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Layer Benchmark")

    parser.add_argument("--config",    type=str, default="configs/default.yaml")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--samples",   type=int, default=None)
    parser.add_argument("--method",    type=str, default=None,
                        help="Method to benchmark (m2ib/lciba/libragrad/fusion)")
    parser.add_argument("--layers",    type=int, nargs="+", default=None,
                        help="Layer indices to sweep e.g. --layers 6 7 8 9 10 11")

    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.data_path is None:
        args.data_path = cfg["data"]["data_path"]
    if args.samples is None:
        args.samples = cfg["benchmark"]["samples"]

    main(args)