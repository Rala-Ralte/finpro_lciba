# metrics file for entropy and cross modal evaluation

import numpy as np
import torch
import torch.nn.functional as F
# import scipy.stats as st
# import math


def entropy(x: torch.Tensor) -> float:
    # shannon entropy calc
    # p = F.softmax(x.abs(), dim=0)
    p = F.softmax(x.abs().float(), dim=-1)
    res_val = -(p * (p + 1e-10).log()).sum().item()
    return res_val


def entropy_weight(attr: torch.Tensor):
    # compute weight based on entropy
    cur_h = entropy(attr)
    w_val = 1.0 / (1.0 + cur_h)
    return w_val, cur_h


def cmaes(vis_attr: torch.Tensor, text_attr: torch.Tensor) -> dict:
    w_v, h_v = entropy_weight(vis_attr)
    w_l, h_l = entropy_weight(text_attr)
    diff_gap = abs(h_v - h_l)

    # dict container
    out_map = {
        "H_vis": h_v,
        "H_lang": h_l,
        "H_gap": diff_gap,
        "w_vis": w_v,
        "w_lang": w_l,
    }
    return out_map


def report_cmaes(results: dict) -> dict:
    dump_res = {}
    for k_name, val_list in results.items():
        np_arr = np.array(val_list)
        dump_res[k_name] = {
            "mean": float(np_arr.mean()),
            "std": float(np_arr.std()),
        }
    return dump_res


def print_cmaes_report(
    cmeg_results: dict,
    baseline_results: dict,
):
    def _row(tag_str, in_dict):
        arr_v = np.array(in_dict["H_vis"])
        arr_l = np.array(in_dict["H_lang"])
        arr_g = np.array(in_dict["H_gap"])

        print("\n  " + str(tag_str))
        print("    H_vis:  " + f"{arr_v.mean():.4f} ± {arr_v.std():.4f}")
        print("    H_lang: " + f"{arr_l.mean():.4f} ± {arr_l.std():.4f}")
        print("    H_gap:  " + f"{arr_g.mean():.4f} ± {arr_g.std():.4f}")
        return arr_g.mean()

    print("\n" + "=" * 50)
    print("CMAES Evaluation Results")
    print("=" * 50)

    gap_cme = _row("CMEGrad", cmeg_results)
    gap_base = _row("Baseline (uniform vision)", baseline_results)

    delta_pct = (gap_cme - gap_base) / gap_base * 100
    avg_wv = np.mean(cmeg_results.get("w_vis", [0]))

    print(f"\n  H_gap improvement: {delta_pct:+.1f}%")
    print("  Mean w_vis:        " + f"{avg_wv:.4f}")
