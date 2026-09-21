
# runs libragrad script


import os, argparse, time
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from random import sample
# from PIL import Image
# import cv2

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

# dvc = 'cpu'
dvc = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(p_args):
    c_data = load_config(p_args.config)
    paths_out = setup_output_dirs(c_data["outputs"]["base_dir"])

    print("=" * 60)
    print("LIBRAGRAD ONLY  [Option C standalone]")
    print("=" * 60)

    # load clip
    net_m, p_proc = load_model(
        clip_model_name=c_data["model"]["clip_variant"],
        device=dvc,
    )

    print("\nLoading data from: " + str(p_args.data_path))

    # load tsv table
    raw_csv = pd.read_csv(p_args.data_path, sep="\t")
    row_list = list(raw_csv.itertuples(index=False))
    # take subset
    row_list = sample(row_list, min(p_args.samples, len(row_list)))

    print("Samples: " + str(len(row_list)))

    i_cfg = c_data["iba"]
    m_dict = {k: AverageMeter(k) for k in ["vdrop", "vincr", "tdrop", "tincr"]}

    for count_i, r_item in enumerate(tqdm(row_list, desc="LibraGrad")):
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

            # grad maps
            # t0 = time.time()
            v_hm = vision_heatmap_libragrad(
                t_ids,
                tens_im,
                net_m,
                layer_idx=i_cfg["vlayer"],
            )

            t_hm = text_heatmap_libragrad(
                t_ids,
                tens_im,
                net_m,
                layer_idx=i_cfg["tlayer"],
            )
            # print("map time:", time.time() - t0)

            res_m = get_metrics(tens_im, v_hm, t_ids, t_hm, net_m)

            for k_m, val_m in res_m.items():
                m_dict[k_m].update(val_m)

            # heatmap saving logic
            if c_data["outputs"]["save_heatmaps"]:
                # unnormalize manually
                arr_im = tens_im[0].detach().cpu().numpy()
                arr_im = arr_im.transpose(1, 2, 0)
                # apply mean std back
                arr_im = (
                    arr_im
                    * np.array([0.26862954, 0.26130258, 0.27577711])
                    + np.array([0.48145466, 0.4578275, 0.40821073])
                )
                arr_im = np.clip(arr_im, 0, 1)

                sub_toks = decode_tokens(t_ids)
                out_png = os.path.join(
                    paths_out["heatmaps"], f"libragrad_{count_i:04d}.png"
                )
                visualize_single(
                    arr_im,
                    v_hm,
                    t_hm,
                    sub_toks,
                    title="LibraGrad | " + str(t_raw[:50]),
                    save_path=out_png,
                )

            # save to csv file
            save_results_csv(
                {
                    "method": "libragrad",
                    "image": i_path,
                    "text": t_raw,
                    **res_m,
                },
                c_data["outputs"]["results_csv"],
            )

        except Exception as e:
            print(f"\n[WARNING] Sample {count_i} failed: {e}")
            continue

    print("\n" + "=" * 60)
    print("LIBRAGRAD RESULTS")
    print("=" * 60)

    for k_m, cur_meter in m_dict.items():
        print(f"  {k_m:8s}: {cur_meter.avg:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser("LibraGrad Only")

    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--data_path", type=str, default=None)
    p.add_argument("--samples", type=int, default=None)

    opt = p.parse_args()

    c_file = load_config(opt.config)
    if opt.data_path is None:
        opt.data_path = c_file["data"]["data_path"]
    if opt.samples is None:
        opt.samples = c_file["data"]["samples"]

    main(opt)
