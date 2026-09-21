import os, argparse, time
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from random import sample

# import matplotlib.pyplot as plt


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

# device stuff
# device = "cpu"
# if torch.cuda.is_available():
#     device = "cuda:0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# metric names
# METRIC_KEYS = ["drop", "increase"]
METRIC_KEYS = ["vdrop", "vincr", "tdrop", "tincr"]


def main(args):

    # load config
    cfg = load_config(args.config)
    dirs = setup_output_dirs(cfg["outputs"]["base_dir"])
    iba_cfg = cfg["iba"]
    bm_cfg = cfg["benchmark"]

    # command line wins if provided
    # method_name = bm_cfg["method"]
    method_name = args.method or bm_cfg["method"]

    # layers to run
    # layer_range = list(range(6, 12))
    layer_range = args.layers or bm_cfg["layer_range"]

    primary_metric = bm_cfg["metric"]

    print("=" * 60)
    print("LAYER BENCHMARK  [" + str(method_name) + "]")
    print("Layers : " + str(layer_range))
    print("Metric : " + str(primary_metric))
    print("=" * 60)

    # load clip
    model, preprocess = load_model(
        clip_model_name=cfg["model"]["clip_variant"],
        device=device,
    )

    # data
    df = pd.read_csv(args.data_path, sep="\t")
    data = list(df.itertuples(index=False))

    # data = data[:args.samples]
    data = sample(
        data,
        min(args.samples, len(data))
    )

    print("Samples per layer: " + str(len(data)))

    # get functions
    vision_fn = VISION_METHODS[method_name]
    text_fn = TEXT_METHODS[method_name]

    # layer -> metrics
    # layer_results = {}
    layer_results = {
        layer: {
            k: AverageMeter("L" + str(layer) + "_" + k)
            for k in METRIC_KEYS
        }
        for layer in layer_range
    }

    for layer_idx in layer_range:

        print("\n--- Layer " + str(layer_idx) + " ---")

        for i, row in enumerate(
            tqdm(data, desc="Layer " + str(layer_idx))
        ):

            text = row[0]
            image_path = row[1]

            try:

                if not isinstance(text, str) or not isinstance(image_path, str):
                    continue

                # load image
                if image_path.startswith("http"):
                    image_tensor, _ = load_image_from_url(
                        image_path,
                        preprocess,
                        device,
                    )
                else:
                    image_tensor, _ = load_image(
                        image_path,
                        preprocess,
                        device,
                    )

                text_ids = load_text(text, device)

                # same params for both
                # kwargs = {}
                kwargs = dict(
                    layer_idx=layer_idx,
                    beta=iba_cfg["vbeta"],
                    var=iba_cfg["vvar"],
                    lr=iba_cfg["vlr"],
                    train_steps=iba_cfg["vsteps"],
                    progbar=False,
                )

                # vision
                vmap = vision_fn(
                    text_ids,
                    image_tensor,
                    model,
                    **kwargs
                )

                # text
                tmap = text_fn(
                    text_ids,
                    image_tensor,
                    model,
                    **kwargs
                )

                metrics = get_metrics(
                    image_tensor,
                    vmap,
                    text_ids,
                    tmap,
                    model
                )

                for k, v in metrics.items():
                    layer_results[layer_idx][k].update(v)

                # save row
                save_results_csv(
                    {
                        "method": method_name,
                        "layer": layer_idx,
                        "image": image_path,
                        "text": text,
                        **metrics,
                    },
                    os.path.join(
                        dirs["metrics"],
                        "benchmark_" + str(method_name) + ".csv"
                    ),
                )

            except Exception as e:
                print(
                    "\n[WARNING] Layer "
                    + str(layer_idx)
                    + " sample "
                    + str(i)
                    + " failed: "
                    + str(e)
                )

                # failed.append((layer_idx, i))
                continue

        avg = layer_results[layer_idx][primary_metric].avg

        print(
            "  Layer "
            + str(layer_idx)
            + "  "
            + str(primary_metric)
            + ": "
            + f"{avg:.4f}"
        )

    # summary
    print("\n" + "=" * 60)
    print(
        "LAYER BENCHMARK RESULTS  ["
        + str(method_name)
        + "]"
    )
    print("=" * 60)

    header = (
        f"{'Layer':6s}"
        + "".join(
            f"  {k:8s}"
            for k in METRIC_KEYS
        )
    )

    print(header)
    print("-" * len(header))

    for layer_idx in layer_range:

        row_str = f"{layer_idx:6d}"

        for k in METRIC_KEYS:
            row_str += (
                f"  "
                f"{layer_results[layer_idx][k].avg:8.4f}"
            )

        print(row_str)

    # best one
    # best_layer = layer_range[0]
    best_layer = max(
        layer_range,
        key=lambda l: layer_results[l][primary_metric].avg,
    )

    print(
        "\nBest layer for "
        + str(primary_metric)
        + ": "
        + str(best_layer)
    )

    # plot
    plot_data = {
        method_name: {
            l: layer_results[l][primary_metric].avg
            for l in layer_range
        }
    }

    chart_path = os.path.join(
        dirs["metrics"],
        "benchmark_"
        + str(method_name)
        + "_"
        + str(primary_metric)
        + ".png",
    )

    plot_layer_benchmark(
        plot_data,
        metric=primary_metric,
        title=(
            "Layer Benchmark — "
            + str(method_name)
            + " — "
            + str(primary_metric)
        ),
        save_path=chart_path,
    )

    print("Chart saved to: " + str(chart_path))


if __name__ == "__main__":

    # setup arguments
    # parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser("Layer Benchmark")

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--method",
        type=str,
        default=None,
        help="Method to benchmark (m2ib/lciba/libragrad/fusion)",
    )

    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Layer indices to sweep e.g. --layers 6 7 8 9 10 11",
    )

    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.data_path is None:
        args.data_path = cfg["data"]["data_path"]

    if args.samples is None:
        args.samples = cfg["benchmark"]["samples"]

    main(args)
