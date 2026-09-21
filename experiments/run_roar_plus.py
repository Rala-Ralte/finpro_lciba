import os, sys, json, copy, argparse, time, math
# from math import *
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from random import shuffle, seed as s_d

# import torchvision
# import PIL
import clip as openai_clip

from scripts.utils import (
    load_model,
    load_image,
    load_image_from_url,
    load_text,
    setup_output_dirs,
    load_config,
)
from scripts.methods import VISION_METHODS, TEXT_METHODS

# global stuff
# dvc = 'cpu'
# if torch.cuda.is_available(): dvc = 'cuda:0'
dvc = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# clip mean or somthing
m_vals = torch.tensor([0.48145466, 0.4578275, 0.40821073])
s_vals = torch.tensor([0.26862954, 0.26130258, 0.27577711])
# norm_zero = m_vals - m_vals
norm_zero = torch.zeros(3)  # zero bcoz normalised

P_TOK = 0  # pad


def corrupt_image(x1, a_map, frac, p_sz=32, g_dim=7):
    # crop and corrupt
    tmp_x = x1.clone()

    # m_t = torch.tensor(a_map).float().cuda()
    m_t = torch.tensor(a_map, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    sc = F.adaptive_avg_pool2d(m_t, (g_dim, g_dim)).reshape(-1)

    # how many to take
    # n_c = int(frac * 49)
    n_c = max(1, int(frac * sc.numel()))
    val_dump, top_k_pos = sc.topk(k=n_c)

    # loop
    for p_id in top_k_pos.tolist():
        # calculate row and col
        row_i = p_id // g_dim
        col_i = p_id % g_dim
        y_a, y_b = row_i * p_sz, (row_i + 1) * p_sz
        x_a, x_b = col_i * p_sz, (col_i + 1) * p_sz
        # tmp_x[0, :, y_a:y_b, x_a:x_b] = 0
        for c_idx in range(3):
            tmp_x[0, c_idx, y_a:y_b, x_a:x_b] = norm_zero[c_idx]

    # return image
    return tmp_x


def corrupt_text(t_in, t_attr, pct):
    t_out = t_in.clone()
    arr = t_out[0]

    # finding end
    # stop_id = 49407
    stop_id = int(arr.argmax().item())

    # tokens inside
    tok_list = list(range(1, stop_id))
    # if len(tok_list) == 0:
    if not tok_list or len(tok_list) <= 0:
        return t_out

    # weights
    # w_list = []
    # for j in tok_list: w_list.append(t_attr[j])
    w_list = [t_attr[j] if j < len(t_attr) else 0.0 for j in tok_list]
    w_t = torch.tensor(w_list, dtype=torch.float32)

    k_num = max(1, int(pct * len(tok_list)))
    k_num = min(k_num, len(tok_list))
    _, mx_pts = w_t.topk(k=k_num)

    for item in mx_pts.tolist():
        # arr[tok_list[item]] = 0
        t_out[0, tok_list[item]] = P_TOK

    return t_out


def clip_contrastive_loss(f1, f2, scl):
    # l2 norm
    f1 = f1 / f1.norm(dim=-1, keepdim=True)
    f2 = f2 / f2.norm(dim=-1, keepdim=True)

    # matmul
    # s1 = f1 @ f2.T * scl
    s1 = scl * f1 @ f2.t()
    s2 = s1.t()

    target_y = torch.arange(f1.shape[0], device=f1.device)

    l1 = F.cross_entropy(s1, target_y)
    l2 = F.cross_entropy(s2, target_y)
    # total
    out_l = (l1 + l2) / 2
    return out_l


def get_trainable_params(net_obj, mode_str):
    # check mode
    if mode_str == "full":
        for param in net_obj.parameters():
            param.requires_grad = True
        return [p for p in net_obj.parameters() if p.requires_grad]

    # freeze all first
    for param in net_obj.parameters():
        param.requires_grad = False

    p_arr = []

    # unfreeze needed
    if hasattr(net_obj, "text_projection") and net_obj.text_projection is not None:
        net_obj.text_projection.requires_grad = True
        p_arr.append(net_obj.text_projection)

    if hasattr(net_obj.visual, "proj") and net_obj.visual.proj is not None:
        net_obj.visual.proj.requires_grad = True
        p_arr.append(net_obj.visual.proj)

    # norms
    for layer_m in [net_obj.ln_final, net_obj.visual.ln_post]:
        for param in layer_m.parameters():
            param.requires_grad = True
            p_arr.append(param)

    # temp scale
    if hasattr(net_obj, "logit_scale"):
        net_obj.logit_scale.requires_grad = True
        p_arr.append(net_obj.logit_scale)

    return p_arr


def finetune_and_eval(
    init_weights,
    imgs_tr,
    txts_tr,
    imgs_te,
    txts_te,
    sc_type,
    n_ep,
    step_sz,
    bs,
    v_name,
):
    # load fresh weights
    # m_inst, _ = openai_clip.load("ViT-B/32", device=dvc)
    m_inst, _ = openai_clip.load(v_name, device=dvc, jit=False)
    m_inst = m_inst.float()
    m_inst.load_state_dict(init_weights)
    m_inst.train()

    p_list = get_trainable_params(m_inst, sc_type)
    # opt = torch.optim.SGD(p_list, lr=step_sz)
    opt = torch.optim.Adam(p_list, lr=step_sz)

    tot_tr = len(imgs_tr)
    idx_arr = list(range(tot_tr))

    for ep in range(n_ep):
        shuffle(idx_arr)
        for st in range(0, tot_tr, bs):
            b_ids = idx_arr[st : st + bs]
            if len(b_ids) < 2:
                # skip single item
                continue

            # stack batches
            batch_i = torch.cat([imgs_tr[k] for k in b_ids], dim=0).to(dvc)
            batch_t = torch.cat([txts_tr[k] for k in b_ids], dim=0).to(dvc)

            opt.zero_grad()
            feat_i = m_inst.encode_image(batch_i)
            feat_t = m_inst.encode_text(batch_t)
            cur_loss = clip_contrastive_loss(
                feat_i, feat_t, m_inst.logit_scale.exp()
            )
            cur_loss.backward()
            opt.step()

    # run validation
    m_inst.eval()
    val_res = []
    tot_val = len(imgs_te)

    with torch.no_grad():
        for st in range(0, tot_val, bs):
            en = min(st + bs, tot_val)
            if en - st < 2:
                continue
            batch_i = torch.cat(imgs_te[st:en], dim=0).to(dvc)
            batch_t = torch.cat(txts_te[st:en], dim=0).to(dvc)
            feat_i = m_inst.encode_image(batch_i)
            feat_t = m_inst.encode_text(batch_t)
            l_val = clip_contrastive_loss(
                feat_i, feat_t, m_inst.logit_scale.exp()
            )
            val_res.append(l_val.item())

    # clear up
    del m_inst
    # del opt
    torch.cuda.empty_cache()

    if len(val_res) > 0:
        return float(np.mean(val_res))
    else:
        return float("nan")


def generate_maps_for_split(
    m_name,
    mod_type,
    wrap_m,
    all_i,
    all_t,
    c_iba,
):
    out_maps = []

    # branch for mode
    if mod_type == "image":
        call_fn = VISION_METHODS[m_name]
        l_idx, b_val, vr, rate, iters = (
            c_iba["vlayer"],
            c_iba["vbeta"],
            c_iba["vvar"],
            c_iba["vlr"],
            c_iba["vsteps"],
        )
    else:
        call_fn = TEXT_METHODS[m_name]
        l_idx, b_val, vr, rate, iters = (
            c_iba["tlayer"],
            c_iba["tbeta"],
            c_iba["tvar"],
            c_iba["tlr"],
            c_iba["tsteps"],
        )

    for i_sample, t_sample in zip(all_i, all_t):
        # run forward
        # print("doing one sample...")
        res_m = call_fn(
            t_sample.to(dvc),
            i_sample.to(dvc),
            wrap_m,
            layer_idx=l_idx,
            beta=b_val,
            var=vr,
            lr=rate,
            train_steps=iters,
            progbar=False,
        )
        out_maps.append(res_m)

    return out_maps


def main(p_args):
    conf = load_config(p_args.config)
    paths = setup_output_dirs(conf["outputs"]["base_dir"])
    iba_params = conf["iba"]

    v_str = conf["model"]["clip_variant"]

    # print screen stuff
    print("=" * 60)
    print("ROAR+ (Retrain-based) — CLIP Attribution Evaluation")
    print("  Methods:          " + str(p_args.methods))
    print("  Modalities:       " + str(p_args.modality))
    print("  Corrupt frac:     " + str(p_args.corrupt_fraction))
    print("  Finetune scope:   " + str(p_args.finetune_scope))
    print("  Repeats:          " + str(p_args.repeats))
    print("  Epochs/retrain:   " + str(p_args.epochs))
    print("=" * 60)

    # get models
    wrp, p_proc = load_model(clip_model_name=v_str, device=dvc)

    # save base weights
    b_net, _ = openai_clip.load(v_str, device=dvc, jit=False)
    b_net = b_net.float()
    base_dict = copy.deepcopy(b_net.state_dict())
    del b_net
    torch.cuda.empty_cache()

    # read tsv
    raw_df = pd.read_csv(p_args.data_path, sep="\t")
    row_data = list(raw_df.itertuples(index=False))

    print("\nLoading up to " + str(p_args.samples) + " samples...")
    buf_i, buf_t = [], []
    for r in tqdm(row_data[: p_args.samples * 2], desc="Loading"):
        if len(buf_i) >= p_args.samples:
            break
        s_txt, s_img = r[0], r[1]
        if not isinstance(s_txt, str) or not isinstance(s_img, str):
            continue
        try:
            if s_img.startswith("http"):
                tens_i, _ = load_image_from_url(s_img, p_proc, dvc)
            else:
                tens_i, _ = load_image(s_img, p_proc, dvc)
            tens_t = load_text(s_txt, dvc)
            # save cpu
            buf_i.append(tens_i.cpu())
            buf_t.append(tens_t.cpu())
        except Exception as err:
            # print("fail to read")
            continue

    total_len = len(buf_i)
    print("Loaded " + str(total_len) + " samples")

    # dict init
    # res_dump = {}
    res_dump = {
        m: {mod: [] for mod in p_args.modality}
        for m in p_args.methods
    }

    # repeat test
    for r_idx in range(p_args.repeats):
        print("\n" + "=" * 60)
        print("REPEAT " + str(r_idx + 1) + "/" + str(p_args.repeats))
        print("=" * 60)
        s_d(conf["data"].get("seed", 42) + r_idx)

        # random indices
        arr_idx = list(range(total_len))
        shuffle(arr_idx)
        cut = int(0.8 * total_len)
        tr_ids = arr_idx[:cut]
        va_ids = arr_idx[cut:]

        tr_x = [buf_i[u] for u in tr_ids]
        tr_y = [buf_t[u] for u in tr_ids]
        va_x = [buf_i[u] for u in va_ids]
        va_y = [buf_t[u] for u in va_ids]

        print("  Train: " + str(len(tr_x)) + "  Val: " + str(len(va_x)))

        for md in p_args.modality:
            print("\n  [" + str(md) + "] Computing l_o (original retrain)...")
            base_loss = finetune_and_eval(
                init_weights=base_dict,
                imgs_tr=tr_x,
                txts_tr=tr_y,
                imgs_te=va_x,
                txts_te=va_y,
                sc_type=p_args.finetune_scope,
                n_ep=p_args.epochs,
                step_sz=p_args.lr,
                bs=p_args.batch_size,
                v_name=v_str,
            )
            print("  [" + str(md) + "] l_o = " + f"{base_loss:.4f}")

            for meth in p_args.methods:
                print("  [" + str(md) + "] Method '" + str(meth) + "': generating maps...")
                cur_maps = generate_maps_for_split(
                    meth, md, wrp, tr_x, tr_y, iba_params
                )

                if md == "image":
                    c_x = [
                        corrupt_image(im, mp, p_args.corrupt_fraction)
                        for im, mp in zip(tr_x, cur_maps)
                    ]
                    c_y = tr_y
                else:
                    c_y = [
                        corrupt_text(tx, mp, p_args.corrupt_fraction)
                        for tx, mp in zip(tr_y, cur_maps)
                    ]
                    c_x = tr_x

                print("  [" + str(md) + "] Method '" + str(meth) + "': retrain on corrupted...")
                corrupt_loss = finetune_and_eval(
                    init_weights=base_dict,
                    imgs_tr=c_x,
                    txts_tr=c_y,
                    imgs_te=va_x,
                    txts_te=va_y,
                    sc_type=p_args.finetune_scope,
                    n_ep=p_args.epochs,
                    step_sz=p_args.lr,
                    bs=p_args.batch_size,
                    v_name=v_str,
                )

                if base_loss != 0:
                    diff_sc = (corrupt_loss - base_loss) / base_loss
                else:
                    diff_sc = float("nan")

                res_dump[meth][md].append(diff_sc)

                print(
                    "  [" + str(md) + "] " + str(meth) + ": l_c=" + f"{corrupt_loss:.4f}" +
                    "  score=(l_c-l_o)/l_o=" + f"{diff_sc:+.4f}"
                )

    # calc summary
    print("\n" + "=" * 60)
    print("ROAR+ RESULTS  (mean ± std over repeats)")
    print("Higher score = more faithful attribution")
    print("=" * 60)

    final_tab = {}
    for meth in p_args.methods:
        final_tab[meth] = {}
        print("\n  " + str(meth.upper()))
        for md in p_args.modality:
            arr_s = np.array(res_dump[meth][md])
            m_val = float(arr_s.mean()) if len(arr_s) else float("nan")
            s_val = float(arr_s.std()) if len(arr_s) else float("nan")
            final_tab[meth][md] = {
                "scores": arr_s.tolist(),
                "mean": m_val,
                "std": s_val,
            }
            print(f"    {md:6s}: {m_val:+.4f} ± {s_val:.4f}")

    # table show
    print("\n  COMPARISON (mean ROAR+ score)")
    head_txt = f"  {'Method':12s}" + "".join(f"  {m:>10s}" for m in p_args.modality)
    print(head_txt)
    print("  " + "-" * (len(head_txt) - 2))
    for meth in p_args.methods:
        r_str = f"  {meth:12s}"
        for md in p_args.modality:
            r_str += f"  {final_tab[meth][md]['mean']:>+10.4f}"
        print(r_str)

    # save json
    dict_to_write = {
        "config": {
            "corrupt_fraction": p_args.corrupt_fraction,
            "finetune_scope": p_args.finetune_scope,
            "repeats": p_args.repeats,
            "epochs": p_args.epochs,
            "lr": p_args.lr,
            "batch_size": p_args.batch_size,
            "n_samples": total_len,
        },
        "results": final_tab,
    }

    dir_make = os.path.join(conf["outputs"]["base_dir"], "metrics")
    os.makedirs(dir_make, exist_ok=True)
    f_dest = os.path.join(dir_make, "roar_plus.json")
    with open(f_dest, "w") as fp:
        json.dump(dict_to_write, fp, indent=2)

    print("\n  Results saved to " + str(f_dest))
    print("✓ run_roar_plus.py complete")


if __name__ == "__main__":
    p = argparse.ArgumentParser("ROAR+ for CLIP attribution")

    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--data_path", type=str, default=None)
    p.add_argument("--samples", type=int, default=500)
    p.add_argument(
        "--methods",
        nargs="+",
        default=["m2ib", "lciba", "libragrad", "fusion", "chefer"],
        help="Methods to evaluate (must exist in methods.py registry)",
    )
    p.add_argument(
        "--modality",
        nargs="+",
        default=["image", "text"],
        choices=["image", "text"],
    )
    p.add_argument(
        "--corrupt_fraction",
        type=float,
        default=0.10,
        help="Top fraction of important units to corrupt (default 0.10)",
    )
    p.add_argument(
        "--finetune_scope",
        type=str,
        default="projection",
        choices=["projection", "full"],
        help="What to finetune during retrain (default: projection)",
    )
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=32)

    args = p.parse_args()

    c_file = load_config(args.config)
    if args.data_path is None:
        args.data_path = c_file["data"]["data_path"]

    main(args)

```
