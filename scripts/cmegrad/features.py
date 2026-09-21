# handles tsv byte offsets and feature unpacking
#touching buffer decoding breaks box normalization

import os, base64
import numpy as np
import torch
from PIL import Image

F_NAMES = [
    "img_id", "img_h", "img_w", "objects_id", "objects_conf",
    "attrs_id", "attrs_conf", "num_boxes", "boxes", "features",
]


def build_feature_index(p_tsv: str) -> dict:
    idx_file = p_tsv + ".index.npy"

    if os.path.exists(idx_file):
        print("  Loading cached feature index...")
        return dict(np.load(idx_file, allow_pickle=True).item())

    print("  Building feature index (one-time, ~30s)...")
    idx_map = {}
    with open(p_tsv, "rb") as fp:
        while True:
            cur_off = fp.tell()
            row_b = fp.readline()
            if not row_b:
                break
            tag_str = row_b.split(b"\t")[0].decode("utf-8")
            idx_map[tag_str] = cur_off

    np.save(idx_file, idx_map)
    print("  Index built: " + f"{len(idx_map):,}" + " images")
    return idx_map


def load_image_features_fast(
    id_num: int,
    idx_map: dict,
    p_tsv: str,
):
    key_name = "COCO_val2014_" + str(id_num).zfill(12)

    if key_name not in idx_map:
        raise KeyError("image_id " + str(id_num) + " not found in feature index")

    with open(p_tsv, "rb") as fp:
        fp.seek(idx_map[key_name])
        raw_str = fp.readline().decode("utf-8")

    dict_row = dict(zip(F_NAMES, raw_str.strip().split("\t")))
    b_count = int(dict_row["num_boxes"])

    arr_f = np.frombuffer(
        base64.b64decode(dict_row["features"]),
        dtype=np.float32,
    ).reshape(b_count, 2048).copy()

    arr_b = np.frombuffer(
        base64.b64decode(dict_row["boxes"]),
        dtype=np.float32,
    ).reshape(b_count, 4).copy()

    dim_h = float(dict_row["img_h"])
    dim_w = float(dict_row["img_w"])
    arr_b[:, [0, 2]] /= dim_w
    arr_b[:, [1, 3]] /= dim_h

    return (
        torch.tensor(arr_f).unsqueeze(0),
        torch.tensor(arr_b).unsqueeze(0),
    )


def load_coco_image(id_num: int, dir_val: str) -> Image.Image:
    f_path = os.path.join(dir_val, str(id_num).zfill(12) + ".jpg")
    return Image.open(f_path).convert("RGB")


def load_vqa_annotations(
    p_q: str,
    p_a: str,
    dir_val: str,
):
    import json
    from collections import defaultdict

    with open(p_q) as fp:
        raw_q = json.load(fp)
    with open(p_a) as fp:
        raw_a = json.load(fp)

    valid_ids = {
        int(f.split(".")[0])
        for f in os.listdir(dir_val)
        if f.endswith(".jpg")
    }

    ans_map = {
        item["question_id"]: item["multiple_choice_answer"]
        for item in raw_a["annotations"]
    }

    q_list = [
        {
            "image_id": q_it["image_id"],
            "question_id": q_it["question_id"],
            "question": q_it["question"],
            "answer": ans_map.get(q_it["question_id"], ""),
        }
        for q_it in raw_q["questions"]
        if q_it["image_id"] in valid_ids
    ]

    grouped_qa = defaultdict(list)
    for q_entry in q_list:
        grouped_qa[q_entry["image_id"]].append(q_entry)

    return q_list, grouped_qa

