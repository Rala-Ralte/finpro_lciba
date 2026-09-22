import os, sys, argparse, time
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from random import sample


# import matplotlib.pyplot as plt
# import random

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

# device
# device = "cpu"
# if torch.cuda.is_available():
#     device = "cuda:0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_metrics(image_feat, vmap, text_ids, tmap, model):

    # metric calculation
    results = {}

    with torch.no_grad():
        # get text target
        # txt_f = model.get_text_features(text_ids)
        vtargets = [
            CosSimilarity(
                model.get_text_features(text_ids).to(device)
            )
        ]

        # image target
        ttargets = [
            CosSimilarity(
                model.get_image_features(image_feat).to(device)
            )
        ]

    # remove first and last tokens
    text_ids_eval = text_ids[:, 1:-1]
    tmap_eval = np.expand_dims(tmap, axis=0)[:, 1:-1]

    # threshold
    # tmap_eval = tmap_eval >= np.median(tmap_eval)
    tmap_eval = tmap_eval > np.percentile(tmap_eval, 50)

    vmap_eval = np.expand_dims(
        vmap,
        axis=(0, 1)
    )

    # vision drop
    results["vdrop"] = DropInConfidence()(
        image_feat,
        vmap_eval,
        vtargets,
        ImageFeatureExtractor(model)
    )[0][0] * 100

    results["vincr"] = IncreaseInConfidence()(
        image_feat,
        vmap_eval,
        vtargets,
        ImageFeatureExtractor(model)
    )[0][0] * 100

    # text drop
    results["tdrop"] = DropInConfidence()(
        text_ids_eval,
        tmap_eval,
        ttargets,
        TextFeatureExtractor(model)
    )[0][0] * 100

    results["tincr"] = IncreaseInConfidence()(
        text_ids_eval,
        tmap_eval,
        ttargets,
        TextFeatureExtractor(model)
    )[0][0] * 100

    return results


def main(args):

    # setup
    cfg = load_config(args.config)
    dirs = setup_output_dirs(cfg["outputs"]["base_dir"])

    print("=" * 60)
    print("M2IB BASELINE")
    print("=" * 60)

    # model load
    # old_model = None
    model, preprocess = load_model(
        clip_model_name=cfg["model"]["clip_variant"],
        device=device,
    )

    # data loading
    print("\nLoading data from: " + str(args.data_path))

    df = pd.read_csv(
        args.data_path,
        sep="\t"
    )

    data = list(df.itertuples(index=False))

    # sample data
    # data = data[:args.samples]
    data = sample(
        data,
        min(args.samples, len(data))
    )

    print("Samples: " + str(len(data)))

    # iba stuff
    iba_cfg = cfg["iba"]

    # meters
    meters = {
        k: AverageMeter(k)
        for k in ["vdrop", "vincr", "tdrop", "tincr"]
    }

    print("\nRunning M2IB baseline ...")

    for i, row in enumerate(
        tqdm(data, desc="M2IB")
    ):

        text = row[0]
        image_path = row[1]

        try:

            if not isinstance(text, str) or not isinstance(image_path, str):
                continue

            # load image
            if image_path.startswith("http"):
                image_tensor, pil_image = load_image_from_url(
                    image_path,
                    preprocess,
                    device
                )
            else:
                image_tensor, pil_image = load_image(
                    image_path,
                    preprocess,
                    device
                )

            # token
            text_ids = load_text(text, device)

            # vision map
            # vmap = None
            vmap = vision_heatmap_iba(
                text_ids,
                image_tensor,
                model,
                layer_idx=iba_cfg["vlayer"],
                beta=iba_cfg["vbeta"],
                var=iba_cfg["vvar"],
                lr=iba_cfg["vlr"],
                train_steps=iba_cfg["vsteps"],
                progbar=args.progbar,
            )

            # text map
            tmap = text_heatmap_iba(
                text_ids,
                image_tensor,
                model,
                layer_idx=iba_cfg["tlayer"],
                beta=iba_cfg["tbeta"],
                var=iba_cfg["tvar"],
                lr=iba_cfg["tlr"],
                train_steps=iba_cfg["tsteps"],
                progbar=args.progbar,
            )

            # metrics
            metrics = get_metrics(
                image_tensor,
                vmap,
                text_ids,
                tmap,
                model
            )

            for k, v in metrics.items():
                meters[k].update(v)

            # save heatmap
            if cfg["outputs"]["save_heatmaps"]:

                image_np = pil_to_numpy(pil_image)
                tokens = decode_tokens(text_ids)

                save_path = os.path.join(
                    dirs["heatmaps"],
                    f"m2ib_{i:04d}.png"
                )

                visualize_single(
                    image_np,
                    vmap,
                    tmap,
                    tokens,
                    title=f"M2IB | {text[:50]}",
                    save_path=save_path,
                )

            # csv stuff
            row_results = {
                "method": "m2ib",
                "image": image_path,
                "text": text,
                **metrics,
            }

            save_results_csv(
                row_results,
                cfg["outputs"]["results_csv"]
            )

        except Exception as e:
            print(
                "\n[WARNING] Sample "
                + str(i)
                + " failed: "
                + str(e)
            )

            # failed.append(i)
            # print(type(e))

            continue

    print("\n" + "=" * 60)
    print("M2IB BASELINE RESULTS")
    print("=" * 60)

    for k, meter in meters.items():
        print(
            f"  {k:8s}: {meter.avg:.4f}"
        )


if __name__ == "__main__":

    # parser setup
    # p = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(
        "M2IB Baseline"
    )

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

    # apply config values
    cfg = load_config(args.config)

    if args.data_path is None:
        args.data_path = cfg["data"]["data_path"]

    if args.samples is None:
        args.samples = cfg["data"]["samples"]

    # run
    main(args)
