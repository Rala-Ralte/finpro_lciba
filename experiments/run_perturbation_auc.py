import os, sys, json, random, argparse, time
import numpy as np
import torch
from tqdm import tqdm
# import matplotlib.pyplot as plt
# from sklearn.metrics import auc as sk_auc

from scripts.utils import load_config, setup_output_dirs, AverageMeter
from scripts.cmegrad.features import (
    build_feature_index,
    load_vqa_annotations,
    load_image_features_fast,
)
from scripts.cmegrad.model import (
    load_lxmert,
    load_libra_lxmert,
    load_id2label,
)
from scripts.cmegrad.attribution import CMEGradLXMERT

# gpu check
# dev = "cpu"
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# chefer steps from paper table or fig 3
P_STP = [0, 0.25, 0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
# P_STP_OLD = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


class CheferModelUsage:
    def __init__(self, ch_mod, tok, d_v):
        self.model = ch_mod
        self.tokenizer = tok
        self.device = d_v

        self.text_len = None
        self.image_boxes_len = None
        self._enc = None
        self._feats = None
        self._boxes = None

    def set_inputs(self, e_obj, f_obj, b_obj):
        # assign variables here
        self._enc = e_obj
        self._feats = f_obj
        self._boxes = b_obj
        self.text_len = e_obj["input_ids"].shape[1]
        self.image_boxes_len = f_obj.shape[1]

    def forward(self, itm=None):
        # forward pass for generator
        # if itm is not None: print("warn: item not used")
        res = self.model(
            input_ids=self._enc["input_ids"],
            attention_mask=self._enc["attention_mask"],
            token_type_ids=self._enc["token_type_ids"],
            visual_feats=self._feats,
            visual_pos=self._boxes,
            return_dict=True,
            output_attentions=False,
        )
        return res


def load_chefer_lxmert(p_path, m_str, c_dir, d_v):
    # append path if not present
    if p_path not in sys.path:
        sys.path.insert(0, p_path)

    try:
        from lxmert_lrp import LxmertForQuestionAnswering as C_LXMERT
        from ExplanationGenerator import GeneratorOurs
    except ImportError as e:
        # sys.exit(1)
        raise ImportError("import failed check chefer folder scripts/chefer_src/")

    import os
    if c_dir:
        os.environ["HF_HOME"] = c_dir
        os.environ["TRANSFORMERS_CACHE"] = c_dir

    print("Loading Chefer's LRP LXMERT...")
    ch_inst = C_LXMERT.from_pretrained(m_str).to(d_v)
    ch_inst.eval()
    print("  ✓ Chefer LRP LXMERT loaded")

    return ch_inst, GeneratorOurs


def _perturb_image(
    m_eval,
    e_dict,
    v_f,
    v_b,
    attr_s,
    a_labels,
    lbl_map,
    pos_flag: bool,
):
    sc = attr_s.clone().float()
    if pos_flag:
        sc = -sc  # invert scores for pos

    n_box = v_f.shape[1]
    res_l = []

    with torch.no_grad():
        for st in P_STP:
            k_n = int((1 - st) * n_box)
            # if k_n <= 0: k_n = 1
            k_n = max(k_n, 1)

            dump, idxs = sc.topk(k=k_n, dim=-1)
            idxs = idxs.cpu().numpy()

            sub_f = v_f[:, idxs, :]
            sub_b = v_b[:, idxs, :]

            out_m = m_eval(
                input_ids=e_dict["input_ids"],
                attention_mask=e_dict["attention_mask"],
                token_type_ids=e_dict["token_type_ids"],
                visual_feats=sub_f,
                visual_pos=sub_b,
                return_dict=True,
            )

            p_i = out_m.question_answering_score.argmax(-1).item()
            ans_str = a_labels[p_i]
            # get acc
            acc_val = lbl_map.get(ans_str, 0.0)
            res_l.append(acc_val)

    return res_l


def _perturb_text(
    m_eval,
    e_dict,
    v_f,
    v_b,
    t_scores,
    a_labels,
    lbl_map,
    pos_flag: bool,
):
    seq_l = e_dict["input_ids"].shape[1]
    w_sc = t_scores.clone().float()

    # find sep
    arr_toks = e_dict["input_ids"][0].cpu()
    sep_pos = (arr_toks != 0).nonzero(as_tuple=True)[0][-1].item()

    c_sc = w_sc[1:sep_pos]
    len_c = c_sc.shape[0]

    if pos_flag:
        c_sc = -c_sc

    acc_res = []

    with torch.no_grad():
        for s_idx in P_STP:
            take_n = int((1 - s_idx) * len_c)
            take_n = max(take_n, 0)

            if take_n > 0:
                _, sub_idx = c_sc.topk(k=take_n, dim=-1)
                sub_idx = sorted((sub_idx + 1).cpu().numpy().tolist())
            else:
                sub_idx = []

            # cls is 0 and sep
            keep_all = sorted(set([0, sep_pos] + sub_idx))

            sub_ids = e_dict["input_ids"][:, keep_all]
            sub_msk = e_dict["attention_mask"][:, keep_all]
            sub_typ = e_dict["token_type_ids"][:, keep_all]

            out_m = m_eval(
                input_ids=sub_ids,
                attention_mask=sub_msk,
                token_type_ids=sub_typ,
                visual_feats=v_f,
                visual_pos=v_b,
                return_dict=True,
            )

            p_idx = out_m.question_answering_score.argmax(-1).item()
            p_ans = a_labels[p_idx]
            acc_res.append(lbl_map.get(p_ans, 0.0))

    return acc_res


def auc(val_arr: list) -> float:
    # return sk_auc(P_STP, val_arr)
    # trapz formula
    ans = float(np.trapz(val_arr, x=P_STP) / (P_STP[-1] - P_STP[0]))
    return ans


def _reset_chefer_attn_state(ch_m):
    # reset all to none
    for m in ch_m.modules():
        if hasattr(m, "attn"):
            m.attn = None
        if hasattr(m, "attn_cam"):
            m.attn_cam = None
        if hasattr(m, "attn_gradients"):
            m.attn_gradients = None


def get_chefer_attrs(
    c_wrap,
    g_cls,
    e_dict,
    v_f,
    v_b,
):
    # clear grads / attn first
    _reset_chefer_attn_state(c_wrap.model)

    c_wrap.set_inputs(e_dict, v_f, v_b)
    gen_obj = g_cls(c_wrap)

    # get relevance matrices
    R_t_t, R_t_i = gen_obj.generate_ours(
        input=None,
        use_lrp=True,
        normalize_self_attention=True,
        apply_self_in_rule_10=True,
    )

    v_rel = R_t_i[0].detach().cpu().float()
    t_rel = R_t_t[0].detach().cpu().float()

    def _norm(x):
        # minmax
        return (x - x.min()) / (x.max() - x.min() + 1e-10)

    return _norm(v_rel), _norm(t_rel)


def build_vqa_label_dicts(path_str):
    import json
    from collections import Counter

    with open(path_str) as fp:
        raw_vqa = json.load(fp)

    out_d = {}
    for itm in raw_vqa["annotations"]:
        q_id = itm["question_id"]
        ans_arr = [a["answer"] for a in itm["answers"]]
        cnts = Counter(ans_arr)
        out_d[q_id] = {
            a_k: min(a_v / 3.0, 1.0) for a_k, a_v in cnts.items()
        }
    return out_d


def model_correct_at_step0(m_obj, e_dict, v_f, v_b, vqa_ans, l_map):
    # test step 0
    with torch.no_grad():
        o = m_obj(
            input_ids=e_dict["input_ids"],
            attention_mask=e_dict["attention_mask"],
            token_type_ids=e_dict["token_type_ids"],
            visual_feats=v_f,
            visual_pos=v_b,
            return_dict=True,
        )
    best_id = o.question_answering_score.argmax(-1).item()
    b_ans = vqa_ans[best_id]
    return l_map.get(b_ans, 0.0)


def main(opt):
    cfg_all = load_config(opt.config)
    paths_out = setup_output_dirs(cfg_all["outputs"]["base_dir"])

    random.seed(cfg_all["data"]["seed"])
    np.random.seed(cfg_all["data"]["seed"])

    print("=" * 60)
    print("Perturbation AUC (MIF/LIF) — CMEGrad vs Chefer")
    str_mode = "LIF (positive)" if opt.positive_pert else "MIF (negative)"
    print("Perturbation type: " + str(str_mode))
    print("Modalities: " + str(opt.modality))
    print("Methods: " + str(opt.methods))
    print("=" * 60)

    # chefer dir path
    c_path = opt.chefer_src_path or os.environ.get(
        "CHEFER_LXMERT_SRC",
        os.path.join(os.getcwd(), "scripts", "chefer_src"),
    )

    # models load
    base_m, tok_m = load_lxmert(
        model_name=cfg_all["model"]["lxmert_name"],
        tokenizer_name=cfg_all["model"]["tokenizer_name"],
        hf_cache=cfg_all["model"]["hf_cache"],
        device=dev,
    )
    map_labels = load_id2label(base_m)
    v_answers = list(map_labels.values())

    l_m = None
    ch_wrapper = None
    gen_cls = None

    if "cmegrad" in opt.methods:
        l_m = load_libra_lxmert(
            model_name=cfg_all["model"]["lxmert_name"],
            hf_cache=cfg_all["model"]["hf_cache"],
            device=dev,
            verify=False,
        )
        cme_engine = CMEGradLXMERT(
            model=l_m,
            tokenizer=tok_m,
            id2label=map_labels,
            feat_index=None,
            feat_tsv=cfg_all["data"]["feat_tsv"],
            device=dev,
        )

    if "chefer" in opt.methods:
        ch_m_inst, gen_cls = load_chefer_lxmert(
            p_path=c_path,
            model_name=cfg_all["model"]["lxmert_name"],
            hf_cache=cfg_all["model"]["hf_cache"],
            d_v=dev,
        )
        ch_wrapper = CheferModelUsage(ch_m_inst, tok_m, dev)

    print("\nLoading feature index...")
    f_idx = build_feature_index(cfg_all["data"]["feat_tsv"])

    print("Loading VQA annotations...")
    q_data, _ = load_vqa_annotations(
        cfg_all["data"]["vqa_q_path"],
        cfg_all["data"]["vqa_a_path"],
        cfg_all["data"]["coco_val_dir"],
    )

    print("Building VQA soft-accuracy label dicts...")
    all_label_dicts = build_vqa_label_dicts(cfg_all["data"]["vqa_a_path"])

    v_set = set(map_labels.values())
    filter_qa = [item for item in q_data if item["answer"] in v_set]
    tot_eval = min(opt.samples, len(filter_qa))
    batch_qa = random.sample(filter_qa, tot_eval)
    print("Evaluation set: " + str(len(batch_qa)) + " QA pairs")

    res_tracker = {
        m: {
            "image": [0.0] * len(P_STP),
            "text": [0.0] * len(P_STP),
            "count": 0,
        }
        for m in opt.methods
    }

    if "random" in opt.methods:
        res_tracker["random"] = {
            "image": [0.0] * len(P_STP),
            "text": [0.0] * len(P_STP),
            "count": 0,
        }

    err_cnt = 0
    bad_step0 = 0

    print("\nRunning perturbation evaluation...")

    for curr_qa in tqdm(batch_qa, desc="Perturb eval"):
        try:
            # load features fast
            cur_f, cur_b = load_image_features_fast(
                curr_qa["image_id"], f_idx, cfg_all["data"]["feat_tsv"]
            )
            cur_f = cur_f.to(dev)
            cur_b = cur_b.to(dev)

            enc_out = tok_m(
                curr_qa["question"],
                padding="max_length",
                max_length=20,
                truncation=True,
                return_tensors="pt",
            ).to(dev)

            corr_ans = curr_qa["answer"]
            q_lbls = all_label_dicts.get(curr_qa["question_id"], {corr_ans: 1.0})

            # check base correct
            acc_0 = model_correct_at_step0(
                base_m, enc_out, cur_f, cur_b, v_answers, q_lbls
            )
            if acc_0 <= 0.0:
                bad_step0 += 1
                continue

            # cmegrad
            if "cmegrad" in opt.methods:
                try:
                    v_cm, _ = cme_engine._vision_attr(enc_out, cur_f, cur_b)
                    t_cm, _ = cme_engine._text_attr(enc_out, cur_f, cur_b)

                    T_len = enc_out["input_ids"].shape[1]
                    V_len = v_cm.shape[0]
                    w_v, H_v = cme_engine._ew(v_cm)
                    w_l, H_l = cme_engine._ew(t_cm)
                    bridge_x = cme_engine._bridge_matrix(
                        enc_out, cur_f, cur_b, T_len, V_len
                    )
                    t_cm, v_cm = cme_engine._refine(
                        t_cm, v_cm, w_l, w_v, bridge_x
                    )

                    def _norm(v_in):
                        return (v_in - v_in.min()) / (v_in.max() - v_in.min() + 1e-10)

                    v_cm = _norm(v_cm.cpu().float())
                    t_cm = _norm(t_cm.cpu().float())

                    if "image" in opt.modality:
                        scores_list = _perturb_image(
                            base_m,
                            enc_out,
                            cur_f,
                            cur_b,
                            v_cm,
                            v_answers,
                            q_lbls,
                            pos_flag=opt.positive_pert,
                        )
                        for k_i, val in enumerate(scores_list):
                            res_tracker["cmegrad"]["image"][k_i] += val

                    if "text" in opt.modality:
                        scores_list = _perturb_text(
                            base_m,
                            enc_out,
                            cur_f,
                            cur_b,
                            t_cm,
                            v_answers,
                            q_lbls,
                            pos_flag=opt.positive_pert,
                        )
                        for k_i, val in enumerate(scores_list):
                            res_tracker["cmegrad"]["text"][k_i] += val

                    res_tracker["cmegrad"]["count"] += 1

                except Exception as e:
                    err_cnt += 1
                    if opt.verbose:
                        import traceback
                        print(f"\n[CMEGrad WARNING] {curr_qa['question'][:40]}")
                        traceback.print_exc()
                finally:
                    l_m.zero_grad()
                    torch.cuda.empty_cache()

            # chefer
            if "chefer" in opt.methods:
                try:
                    v_ch, t_ch = get_chefer_attrs(
                        ch_wrapper, gen_cls, enc_out, cur_f, cur_b
                    )

                    if "image" in opt.modality:
                        scores_list = _perturb_image(
                            base_m,
                            enc_out,
                            cur_f,
                            cur_b,
                            v_ch,
                            v_answers,
                            q_lbls,
                            pos_flag=opt.positive_pert,
                        )
                        for k_i, val in enumerate(scores_list):
                            res_tracker["chefer"]["image"][k_i] += val

                    if "text" in opt.modality:
                        scores_list = _perturb_text(
                            base_m,
                            enc_out,
                            cur_f,
                            cur_b,
                            t_ch,
                            v_answers,
                            q_lbls,
                            pos_flag=opt.positive_pert,
                        )
                        for k_i, val in enumerate(scores_list):
                            res_tracker["chefer"]["text"][k_i] += val

                    res_tracker["chefer"]["count"] += 1

                except Exception as e:
                    err_cnt += 1
                    if opt.verbose:
                        import traceback
                        print(f"\n[Chefer WARNING] {curr_qa['question'][:40]}")
                        traceback.print_exc()
                finally:
                    ch_m_inst.zero_grad()
                    torch.cuda.empty_cache()

            # rand
            if "random" in opt.methods:
                try:
                    v_rnd = torch.rand(36)
                    t_rnd = torch.rand(enc_out["input_ids"].shape[1])

                    if "image" in opt.modality:
                        scores_list = _perturb_image(
                            base_m,
                            enc_out,
                            cur_f,
                            cur_b,
                            v_rnd,
                            v_answers,
                            q_lbls,
                            pos_flag=opt.positive_pert,
                        )
                        for k_i, val in enumerate(scores_list):
                            res_tracker["random"]["image"][k_i] += val

                    if "text" in opt.modality:
                        scores_list = _perturb_text(
                            base_m,
                            enc_out,
                            cur_f,
                            cur_b,
                            t_rnd,
                            v_answers,
                            q_lbls,
                            pos_flag=opt.positive_pert,
                        )
                        for k_i, val in enumerate(scores_list):
                            res_tracker["random"]["text"][k_i] += val

                    res_tracker["random"]["count"] += 1

                except Exception as e:
                    err_cnt += 1
                    if opt.verbose:
                        print(f"\n[Random WARNING] {curr_qa['question'][:40]} -> {e}")

        except Exception as e:
            err_cnt += 1
            if opt.verbose:
                import traceback
                print(f"\n[LOAD WARNING] {curr_qa['question'][:40]}")
                traceback.print_exc()
            continue

    # finish loop
    print("\nCompleted: " + str(res_tracker[opt.methods[0]]["count"]) + "/" + str(len(batch_qa)))
    print("Errors:    " + str(err_cnt))
    print("Skipped (model wrong at step 0): " + str(bad_step0))

    print("\n" + "=" * 60)
    print("PERTURBATION AUC  [" + str(str_mode) + "]")
    print("=" * 60)

    final_map = {}

    for meth in opt.methods:
        tot_cnt = res_tracker[meth]["count"]
        if tot_cnt == 0:
            continue

        final_map[meth] = {}
        print("\n  " + str(meth.upper()))

        for md in ["image", "text"]:
            if md not in opt.modality:
                continue

            avg_vals = [v / tot_cnt for v in res_tracker[meth][md]]
            cur_auc = auc(avg_vals)

            final_map[meth][md] = {
                "per_step": avg_vals,
                "auc": cur_auc,
            }

            s_line = "  ".join(f"{a:.3f}" for a in avg_vals)
            print(f"    {md:5s} steps: {s_line}")
            print(f"    {md:5s} AUC:   {cur_auc:.4f}")

    if len(opt.methods) >= 2:
        print("\n  AUC COMPARISON")
        print(f"  {'Method':12s}  {'Image AUC':>10s}  {'Text AUC':>10s}")
        print("  " + "-" * 36)
        for meth in opt.methods:
            if meth not in final_map:
                continue
            img_score = final_map[meth].get("image", {}).get("auc", float("nan"))
            txt_score = final_map[meth].get("text", {}).get("auc", float("nan"))
            print(f"  {meth:12s}  {img_score:>10.4f}  {txt_score:>10.4f}")

    # save output
    dump_data = {
        "pert_type": str_mode,
        "n_samples": res_tracker[opt.methods[0]]["count"],
        "pert_steps": P_STP,
        "results": final_map,
    }

    out_folder = cfg_all["outputs"].get(
        "results_dir",
        os.path.join(cfg_all["outputs"]["base_dir"], "results"),
    )
    os.makedirs(out_folder, exist_ok=True)
    suffix = "lif" if opt.positive_pert else "mif"
    save_file = os.path.join(out_folder, f"perturbation_auc_{suffix}.json")

    with open(save_file, "w") as fp:
        json.dump(dump_data, fp, indent=2)

    print("\n  Results saved to " + str(save_file))
    print("✓ run_perturbation_auc.py complete")


if __name__ == "__main__":
    p = argparse.ArgumentParser("Perturbation AUC — CMEGrad vs Chefer")

    p.add_argument("--config", type=str, default="configs/cmegrad.yaml")
    p.add_argument("--samples", type=int, default=500)
    p.add_argument(
        "--methods",
        nargs="+",
        default=["cmegrad", "chefer", "random"],
        choices=["cmegrad", "chefer", "random"],
        help="Methods to evaluate",
    )
    p.add_argument(
        "--modality",
        nargs="+",
        default=["image", "text"],
        choices=["image", "text"],
        help="Modalities to perturb",
    )
    p.add_argument(
        "--positive_pert",
        action="store_true",
        help="LIF: remove least important first (default: MIF)",
    )
    p.add_argument(
        "--chefer_src_path",
        type=str,
        default=None,
        help="Path to Chefer lxmert/src/ directory.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-sample warnings",
    )

    opt = p.parse_args()
    main(opt)

```
