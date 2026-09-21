import os, sys, json, random, argparse, time
import torch
import numpy as np
from tqdm import tqdm

# import pandas as pd
# import matplotlib.pyplot as plt
# from copy import deepcopy

from scripts.utils import load_config, setup_output_dirs, AverageMeter
from scripts.cmegrad.features import build_feature_index, load_vqa_annotations
from scripts.cmegrad.model import load_lxmert, load_libra_lxmert, load_id2label
from scripts.cmegrad.metrics import cmaes, print_cmaes_report
from scripts.cmegrad.attribution import CMEGradLXMERT

# device stuff
# device = "cpu"
# if torch.cuda.is_available():
#     device = "cuda:0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def baseline_cmaes(cmeg, ques, img_id):

    # baseline thing
    # no bridge or refinement here
    from scripts.cmegrad.features import load_image_features_fast
    from scripts.cmegrad.metrics import entropy_weight

    feats, boxes = load_image_features_fast(
        img_id,
        cmeg.feat_index,
        cmeg.feat_tsv,
    )

    # tokenize question
    enc = cmeg.tokenizer(
        ques,
        padding="max_length",
        max_length=20,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    feats = feats.to(device)
    boxes = boxes.to(device)

    text_attr, _ = cmeg._text_attr(enc, feats, boxes)

    # uniform vision
    # vis_attr = torch.ones(36) / float(36)
    vis_attr = torch.ones(36) / 36

    _, H_vis = entropy_weight(vis_attr)
    _, H_lang = entropy_weight(text_attr)

    # gap
    H_gap = abs(H_vis - H_lang)

    return H_vis, H_lang, H_gap


def main(args):

    cfg = load_config(args.config)
    dirs = setup_output_dirs(cfg["outputs"]["base_dir"])

    random.seed(cfg["data"]["seed"])

    print("=" * 60)
    print("CMEGrad CMAES Evaluation")
    print("=" * 60)

    # load model
    # old_model = None
    model, tokenizer = load_lxmert(
        model_name=cfg["model"]["lxmert_name"],
        tokenizer_name=cfg["model"]["tokenizer_name"],
        hf_cache=cfg["model"]["hf_cache"],
        device=device,
    )

    id2label = load_id2label(model)

    # load libra patched model
    libra_model = load_libra_lxmert(
        model_name=cfg["model"]["lxmert_name"],
        hf_cache=cfg["model"]["hf_cache"],
        device=device,
        verify=True,
        base_model=model,
        tokenizer=tokenizer,
    )

    # data
    print("\nLoading feature index...")
    feat_index = build_feature_index(cfg["data"]["feat_tsv"])
    print("Feature index: " + str(len(feat_index)) + " images")

    print("\nLoading VQA annotations...")
    questions, _ = load_vqa_annotations(
        cfg["data"]["vqa_q_path"],
        cfg["data"]["vqa_a_path"],
        cfg["data"]["coco_val_dir"],
    )

    print("  VQA pairs: " + str(len(questions)))

    # filter stuff
    all_qa = [
        q for q in questions
        if q["answer"] in id2label.values()
    ]

    n = min(args.samples, len(all_qa))

    # eval_qa = all_qa[:n]
    eval_qa = random.sample(all_qa, n)

    print("  Evaluation set: " + str(len(eval_qa)) + " QA pairs")

    # cme object
    cmegrad = CMEGradLXMERT(
        model=libra_model,
        tokenizer=tokenizer,
        id2label=id2label,
        feat_index=feat_index,
        feat_tsv=cfg["data"]["feat_tsv"],
        device=device,
    )

    # results
    # results = {}
    results = {
        "cmeg": {
            "H_vis": [],
            "H_lang": [],
            "H_gap": [],
            "w_vis": [],
        },
        "baseline": {
            "H_vis": [],
            "H_lang": [],
            "H_gap": [],
        },
    }

    errors = 0

    print("\nRunning evaluation on " + str(len(eval_qa)) + " samples...")

    for qa in tqdm(eval_qa, desc="CMEGrad eval"):
        try:

            # do cmeg
            _, _, info = cmegrad.generate(
                qa["question"],
                qa["image_id"],
            )

            results["cmeg"]["H_vis"].append(info["H_vis"])
            results["cmeg"]["H_lang"].append(info["H_lang"])
            results["cmeg"]["H_gap"].append(info["H_gap"])
            results["cmeg"]["w_vis"].append(info["w_vis"])

            # baseline
            H_vis_b, H_lang_b, H_gap_b = baseline_cmaes(
                cmegrad,
                qa["question"],
                qa["image_id"],
            )

            results["baseline"]["H_vis"].append(H_vis_b)
            results["baseline"]["H_lang"].append(H_lang_b)
            results["baseline"]["H_gap"].append(H_gap_b)

        except Exception as e:
            errors += 1
            # print("sample failed", e)
            # failed.append(qa)
            continue

    print(
        "\n  Completed: "
        + str(len(results["cmeg"]["H_vis"]))
        + "/"
        + str(len(eval_qa))
    )

    print("  Errors:    " + str(errors))

    # report
    print_cmaes_report(
        results["cmeg"],
        results["baseline"],
    )

    # save things
    os.makedirs(
        cfg["outputs"]["results_dir"],
        exist_ok=True,
    )

    # save = {}
    save = {
        "n_samples": len(results["cmeg"]["H_vis"]),

        "cmeg": {
            k: float(np.mean(v))
            for k, v in results["cmeg"].items()
        },

        "baseline": {
            k: float(np.mean(v))
            for k, v in results["baseline"].items()
        },

        "cmeg_std": {
            k: float(np.std(v))
            for k, v in results["cmeg"].items()
        },

        "baseline_std": {
            k: float(np.std(v))
            for k, v in results["baseline"].items()
        },
    }

    out_path = cfg["outputs"]["results_json"]

    with open(out_path, "w") as f:
        json.dump(save, f, indent=2)

    print("\n  Results saved to " + str(out_path))
    print("run_cmegrad.py complete")


if __name__ == "__main__":

    # parser
    # p = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(
        "CMEGrad CMAES Evaluation"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/cmegrad.yaml",
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    # get config again
    cfg = load_config(args.config)

    if args.samples is N
```
