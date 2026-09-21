
import os, argparse, time
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from random import sample
# from PIL import Image
# import cv2

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

# dvc = 'cpu'
# if torch.cuda.is_available(): dvc = 'cuda'
dvc = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(p_args):
    c_dat = load_config(p_args.config)
    paths_out = setup_output_dirs(c_dat["outputs"]["base_dir"])

    # setup tag
    if p_args.no_libragrad:
        tag_str = "lciba_ablation_nograd"
    elif p_args.no_libragrad_init:
        tag_str = "lciba_option_a_only"
    else:
        tag_str = "lciba_full"

    print("=" * 60)
    print("LC-IBA  [" + str(tag_str) + "]")
    print("=" * 60)
    print("  use_libragrad  : " + str(not p_args.no_libragrad))
    print("  libragrad_init : " + str(not p_args.no_libragrad_init))

    # load net
    net_m, p_proc = load_model(
        clip_model_name=c_dat["model"]["clip_variant"],
        device=dvc,
    )

    print("\nLoading data from: " + str(p_args.data_path))

    raw_csv = pd.read_csv(p_args.data_path, sep="\t")
    row_list = list(raw_csv.itertuples(index=False))
    # random sample subset
    row_list = sample(row_list, min(p_args.samples, len(row_list)))

    print("Samples: " + str(len(row_list)))

    i_cfg = c_dat["iba"]
    lc_cfg = c_dat["lciba"]

    p_str = lc_cfg["prior_strength"]
    flag_lg = not p_args.no_libragrad
    flag_init = not p_args.no_libragrad_init

    m_dict = {k: AverageMeter(k) for k in ["vdrop", "vincr", "tdrop", "tincr"]}

    print("\nRunning " + str(tag_str) + " ...")

    for count_i, r_item in enumerate(tqdm(row_list, desc=tag_str)):
        t_raw = r_item[0]
        i_path = r_item[1]

        try:
            if not isinstance(t_raw, str) or not isinstance(i_path, str):
                continue
            # load img
            if i_path.startswith("http"):
                tens_im, pil_im = load_image_from_url(i_path, p_proc, dvc)
            else:
                tens_im, pil_im = load_image(i_path, p_proc, dvc)

            t_ids = load_text(t_raw, dvc)

            # run lciba maps
            # t0 = time.time()
            v_hm = vision_heatmap_lciba(
                t_ids,
                tens_im,
                net_m,
                layer_idx=i_cfg["vlayer"],
                beta=i_cfg["vbeta"],
                var=i_cfg["vvar"],
                lr=i_cfg["vlr"],
                train_steps=i_cfg["vsteps"],
                progbar=p_args.progbar,
                prior_strength=p_str,
            )

            t_hm = text_heatmap_lciba(
                t_ids,
                tens_im,
                net_m,
                layer_idx=i_cfg["tlayer"],
                beta=i_cfg["tbeta"],
                var=i_cfg["tvar"],
                lr=i_cfg["tlr"],
                train_steps=i_cfg["tsteps"],
                progbar=p_args.progbar,
                prior_strength=p_str,
            )
            # print("lciba step time:", time.time() - t0)

            res_m = get_metrics(tens_im, v_hm, t_ids, t_hm, net_m)

            for k_m, val_m in res_m.items():
                m_dict[k_m].update(val_m)

            # heatmaps dumping
            if c_dat["outputs"]["save_heatmaps"]:
                arr_im = pil_to_numpy(pil_im)
                sub_toks = decode_tokens(t_ids)
                out_png = os.path.join(
                    paths_out["heatmaps"], f"{tag_str}_{count_i:04d}.png"
                )
                visualize_single(
                    arr_im,
                    v_hm,
                    t_hm,
                    sub_toks,
                    title="LC-IBA | " + str(t_raw[:50]),
                    save_path=out_png,
                )

            # append csv row
            out_row = {
                "method": tag_str,
                "image": i_path,
                "text": t_raw,
                **res_m,
            }
            save_results_csv(out_row, c_dat["outputs"]["results_csv"])

        except Exception as e:
            print(f"\n[WARNING] Sample {count_i} failed: {e}")
            continue

    print("\n" + "=" * 60)
    print("LC-IBA RESULTS  [" + str(tag_str) + "]")
    print("=" * 60)

    for k_m, cur_meter in m_dict.items():
        print(f"  {k_m:8s}: {cur_meter.avg:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser("LC-IBA")

    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--data_path", type=str, default=None)
    p.add_argument("--samples", type=int, default=None)
    p.add_argument("--progbar", action="store_true")
    p.add_argument(
        "--no_libragrad",
        action="store_true",
        help="Disable LibraGrad hooks",
    )
    p.add_argument(
        "--no_libragrad_init",
        action="store_true",
        help="Disable prior initialization",
    )

    opt = p.parse_args()

    c_file = load_config(opt.config)
    if opt.data_path is None:
        opt.data_path = c_file["data"]["data_path"]
    if opt.samples is None:
        opt.samples = c_file["data"]["samples"]

    main(opt)
