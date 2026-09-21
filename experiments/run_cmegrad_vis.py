import os, json, random, argparse, time
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image

from scripts.utils import load_config, setup_output_dirs
from scripts.cmegrad.features import build_feature_index, load_vqa_annotations, load_coco_image
from scripts.cmegrad.model import load_lxmert, load_libra_lxmert, load_id2label
from scripts.cmegrad.attribution import CMEGradLXMERT
from scripts.cmegrad.visualization import visualize_cmegrad

# device thing
# dev = "cpu"
# if torch.cuda.is_available():
#     dev = "cuda"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# old stuff
# OUT = "heatmaps"
# debug_list = []
# import matplotlib.pyplot as plt


def main(args):
    cfg = load_config(args.config)
    dirs = setup_output_dirs(cfg["outputs"]["base_dir"])

    random.seed(args.seed)

    print("=" * 60)
    print(f"CMEGrad Visualization  [{args.n} samples]")
    print("=" * 60)

    # load model
    # model, tokenizer = load_lxmert(cfg["model"]["lxmert_name"], ...)
    model, tokenizer = load_lxmert(
        model_name=cfg["model"]["lxmert_name"],
        tokenizer_name=cfg["model"]["tokenizer_name"],
        hf_cache=cfg["model"]["hf_cache"],
        device=device,
    )

    # labels
    id2label = load_id2label(model)

    # libra version
    libra_model = load_libra_lxmert(
        model_name=cfg["model"]["lxmert_name"],
        hf_cache=cfg["model"]["hf_cache"],
        device=device,
        verify=False,
        base_model=model,
        tokenizer=tokenizer,
    )

    # feat stuff
    feat_index = build_feature_index(cfg["data"]["feat_tsv"])

    questions, _ = load_vqa_annotations(
        cfg["data"]["vqa_q_path"],
        cfg["data"]["vqa_a_path"],
        cfg["data"]["coco_val_dir"],
    )

    # same random process as the other script
    random.seed(cfg["data"]["seed"])
    all_qa = [q for q in questions if q["answer"] in id2label.values()]

    # x = list(all_qa)
    # random.shuffle(x)
    # eval_qa = x[:500]
    eval_qa = random.sample(all_qa, min(500, len(all_qa)))

    # select smaller group
    random.seed(args.seed)
    subset = random.sample(eval_qa, min(args.n, len(eval_qa)))

    print("  Subset: " + str(len(subset)) + " samples")

    # make object
    # cme = None
    cmegrad = CMEGradLXMERT(
        model=libra_model,
        tokenizer=tokenizer,
        id2label=id2label,
        feat_index=feat_index,
        feat_tsv=cfg["data"]["feat_tsv"],
        device=device,
    )

    vis_dir = os.path.join(cfg["outputs"]["base_dir"], "heatmaps")
    os.makedirs(vis_dir, exist_ok=True)

    errors = 0
    # old_errors = []

    for i, qa in enumerate(tqdm(subset, desc="Visualizing")):
        try:
            img_pil = load_coco_image(
                qa["image_id"],
                cfg["data"]["coco_val_dir"],
            )

            # print("generating", i)
            lang_r, vis_r, info = cmegrad.generate(
                qa["question"], qa["image_id"]
            )

            # keep question around
            info["question"] = qa["question"]

            # name = "cmeg_" + str(qa["image_id"]) + ".png"
            save_path = os.path.join(
                vis_dir,
                f"cmeg_{str(qa['image_id']).zfill(12)}_{i:04d}.png",
            )

            visualize_cmegrad(
                img_pil=img_pil,
                vis_attr_raw=info["vis_attr_raw"],
                text_attr=lang_r,
                boxes=info["boxes"],
                tokens=info["tokens"],
                info=info,
                save_path=save_path,
            )

            print(
                f"  [{i+1:03d}] {qa['question'][:45]:45s} "
                f"-> {info['pred_answer']:10s} "
                f"H_gap={info['H_gap']:.3f}"
            )

        except Exception as e:
            errors += 1
            # errors.append(i)
            # print(type(e))
            print(f"  [WARNING] Sample {i} failed: {e}")
            continue

    print("\n  Saved: " + str(len(subset) - errors) + "/" + str(len(subset)))
    print("  Errors: " + str(errors))
    print("  Output: " + str(vis_dir))
    print("done - run_cmegrad_vis.py")


if __name__ == "__main__":
    # parser setup
    # p = argparse.ArgumentParser()
    parser = argparse.ArgumentParser("CMEGrad Visualization")

    parser.add_argument("--config", type=str, default="configs/cmegrad.yaml")

    parser.add_argument(
        "--n", type=int, default=20,
        help="Number of samples to visualize",
    )

    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed for subset selection",
    )

    args = parser.parse_args()

    # if args.n < 1:
    #     raise ValueError("n must be positive")
    main(args)
