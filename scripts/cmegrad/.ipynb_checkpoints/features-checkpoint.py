"""
scripts/cmegrad/features.py

FRCNN bottom-up feature loading for LXMERT.

Responsibilities:
    - O(1) byte-offset indexed TSV access
    - Normalized bounding box loading
    - Feature index build + cache
    - COCO image loading
"""

import os
import base64
import numpy as np
import torch
from PIL import Image


FIELDNAMES = [
    "img_id", "img_h", "img_w", "objects_id", "objects_conf",
    "attrs_id", "attrs_conf", "num_boxes", "boxes", "features",
]


# ============================================================
# Feature Index
# ============================================================

def build_feature_index(tsv_path: str) -> dict:
    """
    Build or load a byte-offset index for the TSV feature file.
    Index maps image_id string → byte offset for O(1) seek.

    Cached at tsv_path + ".index.npy".
    """
    index_path = tsv_path + ".index.npy"

    if os.path.exists(index_path):
        print("  Loading cached feature index...")
        return dict(np.load(index_path, allow_pickle=True).item())

    print("  Building feature index (one-time, ~30s)...")
    index = {}
    with open(tsv_path, "rb") as f:
        while True:
            offset = f.tell()
            line   = f.readline()
            if not line:
                break
            img_id        = line.split(b"\t")[0].decode("utf-8")
            index[img_id] = offset

    np.save(index_path, index)
    print(f"  Index built: {len(index):,} images")
    return index


# ============================================================
# Feature Loading
# ============================================================

def load_image_features_fast(
    image_id: int,
    index: dict,
    tsv_path: str,
):
    """
    Load FRCNN bottom-up features for a single image.

    Args:
        image_id: COCO image id (int)
        index:    Byte-offset index from build_feature_index()
        tsv_path: Path to val2014_obj36.tsv

    Returns:
        feats: torch.Tensor [1, num_boxes, 2048]
        boxes: torch.Tensor [1, num_boxes, 4]  (normalized x1y1x2y2)
    """
    target_id = f"COCO_val2014_{str(image_id).zfill(12)}"

    if target_id not in index:
        raise KeyError(f"image_id {image_id} not found in feature index")

    with open(tsv_path, "rb") as f:
        f.seek(index[target_id])
        line = f.readline().decode("utf-8")

    row       = dict(zip(FIELDNAMES, line.strip().split("\t")))
    num_boxes = int(row["num_boxes"])

    feats = np.frombuffer(
        base64.b64decode(row["features"]),
        dtype=np.float32,
    ).reshape(num_boxes, 2048).copy()

    boxes = np.frombuffer(
        base64.b64decode(row["boxes"]),
        dtype=np.float32,
    ).reshape(num_boxes, 4).copy()

    h, w             = float(row["img_h"]), float(row["img_w"])
    boxes[:, [0, 2]] /= w
    boxes[:, [1, 3]] /= h

    return (
        torch.tensor(feats).unsqueeze(0),
        torch.tensor(boxes).unsqueeze(0),
    )


# ============================================================
# COCO Image Loading
# ============================================================

def load_coco_image(image_id: int, coco_val_dir: str) -> Image.Image:
    """
    Load a COCO val2017 image by ID.

    Args:
        image_id:     COCO image id (int)
        coco_val_dir: Path to val2017/ directory

    Returns:
        PIL.Image in RGB mode
    """
    path = os.path.join(coco_val_dir, f"{str(image_id).zfill(12)}.jpg")
    return Image.open(path).convert("RGB")


# ============================================================
# VQA Annotation Loading
# ============================================================

def load_vqa_annotations(
    vqa_q_path: str,
    vqa_a_path: str,
    coco_val_dir: str,
):
    """
    Load VQA v2 annotations filtered to COCO val2017 images.

    Args:
        vqa_q_path:   Path to v2_OpenEnded_mscoco_val2014_questions.json
        vqa_a_path:   Path to v2_mscoco_val2014_annotations.json
        coco_val_dir: Path to val2017/ directory

    Returns:
        questions: list of dicts with keys:
                   image_id, question_id, question, answer
        qa_by_image: defaultdict(list) keyed by image_id
    """
    import json
    from collections import defaultdict

    with open(vqa_q_path) as f:
        vqa_q = json.load(f)
    with open(vqa_a_path) as f:
        vqa_a = json.load(f)

    val2017_ids = {
        int(fname.split(".")[0])
        for fname in os.listdir(coco_val_dir)
        if fname.endswith(".jpg")
    }

    answers = {
        a["question_id"]: a["multiple_choice_answer"]
        for a in vqa_a["annotations"]
    }

    questions = [
        {
            "image_id":    q["image_id"],
            "question_id": q["question_id"],
            "question":    q["question"],
            "answer":      answers.get(q["question_id"], ""),
        }
        for q in vqa_q["questions"]
        if q["image_id"] in val2017_ids
    ]

    qa_by_image = defaultdict(list)
    for q in questions:
        qa_by_image[q["image_id"]].append(q)

    return questions, qa_by_image