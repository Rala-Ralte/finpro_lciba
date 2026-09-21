
# compares all methods and dumps plots/csv


import os, argparse, time
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from random import sample
# from PIL import Image
# import matplotlib.pyplot as plt

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

# dev = "cpu"
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

KEYS_ALL = ["vdrop", "vincr", "tdrop", "tincr"]


def main(p_args):
    c_dat = load_config(p_args.config)
    paths_out = setup_output_dirs(c_dat["outputs"]["base_dir"])
    m_list = c_dat["compare"]["methods"]
    i_cfg = c_dat["iba"]

    print("=" * 60)
    print("METHOD COMPARISON")
    print("Methods: " + str(m_list))
    print("=" * 60)

    # load clip
    net_m, p_proc = load_model(
        clip_model_name=c_dat["model"]["clip_variant"],
        device=dev,
    )

    # read tsv
    raw_df = pd.read_csv(p_args.data_path, sep="\t")
    row_data = list(raw_df.itertuples(index=False))
    # take subset
    row_data = sample(row_data, min(p_args.samples, len(row_data)))

    print("Samples: " + str(len(row_data)))

    # build meter container
    trackers = {
        m: {k: AverageMeter(str(m) + "_" + str(k)) for k in KEYS_ALL}
        for m in m_list
    }

    for count_i, r_item in enumerate(tqdm(row_data, desc="Comparing")):
        t_raw = r_item[0]
        i_path = r_item[1]

        try:
            if not isinstance(t_raw, str) or not isinstance(i_path, str):
                continue
            # fetch img
            if i_path.startswith("http"):
                tens_im, pil_im = load_image_from_url(i_path, p_proc, dev)
            else:
                tens_im, pil_im = load_image(i_path, p_proc, dev)

            t_ids = load_text(t_raw, dev)
            arr_im = pil_to_numpy(pil_im)
            sub_toks = decode_tokens(t_ids)

            dict_v = {}
            dict_t = {}

            for cur_m in m_list:
                fn_v = VISION_METHODS[cur_m]
                fn_t = TEXT_METHODS[cur_m]

                # common args dict
                kw = dict(
                    layer_idx=i_cfg["vlayer"],
                    beta=i_cfg["vbeta"],
                    var=i_cfg["vvar"],
                    lr=i_cfg["vlr"],
                    train_steps=i_cfg["vsteps"],
                    progbar=False,
                )

                # t0 = time.time()
                v_map = fn_v(t_ids, tens_im, net_m, **kw)
                t_map = fn_t(
                    t_ids,
                    tens_im,
                    net_m,
                    **{
                        **kw,
                        "layer_idx": i_cfg["tlayer"],
                        "beta": i_cfg["tbeta"],
                        "var": i_cfg["tvar"],
                        "lr": i_cfg["tlr"],
                        "train_steps": i_cfg["tsteps"],
                    },
                )
                # print("eval time:", time.time() - t0)

                dict_v[cur_m] = v_map
                dict_t[cur_m] = t_map

                sc_dict = get_metrics(tens_im, v_map, t_ids, t_map, net_m)

                for k_k, val_v in sc_dict.items():
                    trackers[cur_m][k_k].update(val_v)

                save_results_csv(
                    {
                        "method": cur_m,
                        "image": i_path,
                        "text": t_raw,
                        **sc_dict,
                    },
                    c_dat["outputs"]["results_csv"],
                )

            # dump grid img
            if c_dat["compare"]["save_comparison_grid"]:
                out_grid = os.path.join(
                    paths_out["heatmaps"], f"compare_{count_i:04d}.png"
                )
                visualize_comparison(
                    arr_im,
                    dict_v,
                    dict_t,
                    sub_toks,
                    title=t_raw[:60],
                    save_path=out_grid,
                )

        except Exception as e:
            print(f"\n[WARNING] Sample {count_i} failed: {e}")
            continue

    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)

    res_tab = {}
    head_txt = f"{'Method':12s}" + "".join(f"  {k:8s}" for k in KEYS_ALL)
    print(head_txt)
    print("-" * len(head_txt))

    for cur_m in m_list:
        row_str = f"{cur_m:12s}"
        res_tab[cur_m] = {}
        for k_k in KEYS_ALL:
            cur_avg = trackers[cur_m][k_k].avg
            res_tab[cur_m][k_k] = cur_avg
            row_str += f"  {cur_avg:8.4f}"
        print(row_str)

    # plot chart
    plot_file = os.path.join(paths_out["metrics"], "comparison_chart.png")
    plot_metrics_comparison(
        res_tab,
        metric_keys=KEYS_ALL,
        title="Method Comparison — All Metrics",
        save_path=plot_file,
    )
    print("\nChart saved to: " + str(plot_file))


if __name__ == "__main__":
    p = argparse.ArgumentParser("Compare Methods")

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
