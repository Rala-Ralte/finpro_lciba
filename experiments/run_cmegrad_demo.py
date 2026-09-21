import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # forced cpu bcoz oom issues

import argparse
# import sys, time
# from PIL import Image
# import matplotlib.pyplot as plt
import torch

from scripts.utils import load_config, setup_output_dirs
from scripts.cmegrad.features import build_feature_index, load_coco_image
from scripts.cmegrad.model import load_lxmert, load_libra_lxmert, load_id2label
from scripts.cmegrad.attribution import CMEGradLXMERT
from scripts.cmegrad.visualization import visualize_cmegrad

# dvc = 'cpu'
# if torch.cuda.is_available(): dvc = 'cuda'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(p_args):
    c_dat = load_config(p_args.config)
    dump_dirs = setup_output_dirs(c_dat["outputs"]["base_dir"])

    print("=" * 60)
    print("CMEGrad Demo Visualization")
    print("=" * 60)

    # only load libra net directly
    m_lib, t_ok = load_lxmert(
        model_name=c_dat["model"]["lxmert_name"],
        tokenizer_name=c_dat["model"]["tokenizer_name"],
        hf_cache=c_dat["model"]["hf_cache"],
        device=device,
    )
    l_map = load_id2label(m_lib)

    # patch imports
    from scripts.cmegrad.model import patch_layernorms, patch_gelu
    c_ln = patch_layernorms(m_lib)
    c_gelu = patch_gelu(m_lib)
    # print("debug ln and gelu count:")
    print("  LibraLayerNorm: " + str(c_ln) + "  LibraGELU: " + str(c_gelu))

    # index build
    idx_f = build_feature_index(c_dat["data"]["feat_tsv"])

    # engine init
    cme_obj = CMEGradLXMERT(
        model=m_lib,
        tokenizer=t_ok,
        id2label=l_map,
        feat_index=idx_f,
        feat_tsv=c_dat["data"]["feat_tsv"],
        device=device,
    )

    # loop examples
    eg_list = c_dat["demo"]["examples"]
    v_opts = c_dat["visualization"]

    # make folder
    # if not os.path.exists(c_dat["outputs"]["heatmaps_dir"]): os.makedirs(...)
    os.makedirs(c_dat["outputs"]["heatmaps_dir"], exist_ok=True)

    for itm in eg_list:
        i_id = itm["image_id"]
        q_str = itm["question"]

        # print("processing now...")
        print("\nProcessing: " + str(q_str))

        # read coco
        raw_pil = load_coco_image(i_id, c_dat["data"]["coco_val_dir"])
        r_l, r_v, meta_d = cme_obj.generate(q_str, i_id)

        meta_d["question"] = q_str

        # zero fill string
        # s_id = "%012d" % i_id
        s_id = str(i_id).zfill(12)
        f_out = os.path.join(
            c_dat["outputs"]["heatmaps_dir"],
            "cmeg_" + str(s_id) + ".png",
        )

        # call plot
        visualize_cmegrad(
            img_pil=raw_pil,
            vis_attr_raw=meta_d["vis_attr_raw"],
            text_attr=r_l,
            boxes=meta_d["boxes"],
            tokens=meta_d["tokens"],
            info=meta_d,
            save_path=f_out,
            show=False,
            sigma=v_opts["sigma"],
            top_k=v_opts["top_k"],
            max_width=v_opts["max_width"],
            max_height=v_opts["max_height"],
            fontsize=v_opts["fontsize"],
            dpi=v_opts["dpi"],
        )

        print("  pred:   " + str(meta_d["pred_answer"]))
        p_txt = (
            "  H_vis:  " + f"{meta_d['H_vis']:.4f}" +
            "  H_lang: " + f"{meta_d['H_lang']:.4f}" +
            "  H_gap: " + f"{meta_d['H_gap']:.4f}"
        )
        print(p_txt)
        print("  ✓ saved -> " + str(f_out))

    print("\n✓ run_cmegrad_demo.py complete")


if __name__ == "__main__":
    p = argparse.ArgumentParser("CMEGrad Demo")
    p.add_argument("--config", type=str, default="configs/cmegrad.yaml")
    args = p.parse_args()
    main(args)

```
