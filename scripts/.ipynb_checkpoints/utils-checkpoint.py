"""
scripts/utils.py

Shared utilities for LC-IBA experiment.

Based on scripts/utils.py from M2IB:
    https://github.com/YingWANGG/M2IB

Changes vs original:
    [ADD] load_image()         - unified image loading + CLIP preprocessing
    [ADD] load_text()          - unified text tokenization
    [ADD] load_model()         - CLIP + ClipWrapper loader
    [ADD] save_results_csv()   - append results row to CSV
    [ADD] AverageMeter         - running mean tracker for metrics
    [ADD] setup_output_dirs()  - creates outputs/ subdirectories
"""

import os
import csv
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Original M2IB Utilities (unchanged)
# ============================================================

def normalize(x: np.ndarray) -> np.ndarray:
    """
    Min-max normalize array to [0, 1].

    Handles edge case where max == min (returns zeros).
    """
    x_min = x.min()
    x_max = x.max()

    if x_max - x_min < 1e-8:
        return np.zeros_like(x)

    return (x - x_min) / (x_max - x_min)


class mySequential(nn.Sequential):
    """
    nn.Sequential that forwards tuple inputs correctly.
    Used to chain original_layer + InformationBottleneck.
    """

    def forward(self, *input, **kwargs):
        for module in self._modules.values():
            if type(input) == tuple:
                input = module(*input)
            else:
                input = module(input)
        return input


def replace_layer(
    model: nn.Module,
    target: nn.Module,
    replacement: nn.Module,
):
    """
    Replace a given module within a parent module.
    Used to inject InformationBottleneck into the transformer.

    Args:
        model:       Parent nn.Module to search within
        target:      The module to replace
        replacement: The module to insert in its place

    Raises:
        RuntimeError if target is not found in model
    """

    def replace_in(
        model: nn.Module,
        target: nn.Module,
        replacement: nn.Module,
    ) -> bool:
        for name, submodule in model.named_children():
            if submodule == target:
                if isinstance(model, nn.ModuleList):
                    model[int(name)] = replacement
                elif isinstance(model, nn.Sequential):
                    model[int(name)] = replacement
                else:
                    model.__setattr__(name, replacement)
                return True
            elif len(list(submodule.named_children())) > 0:
                if replace_in(submodule, target, replacement):
                    return True
        return False

    if not replace_in(model, target, replacement):
        raise RuntimeError(
            f"Cannot substitute layer: "
            f"{target.__class__.__name__} is not a child of "
            f"{model.__class__.__name__}"
        )


class CosSimilarity:
    """
    Target function for pytorch-grad-cam metrics.
    Computes cosine similarity between model output
    and stored reference features.
    """

    def __init__(self, features: torch.Tensor):
        self.features = features

    def __call__(self, model_output: torch.Tensor) -> torch.Tensor:
        cos = torch.nn.CosineSimilarity()
        return cos(model_output, self.features)


class ImageFeatureExtractor(torch.nn.Module):
    """Wraps ClipWrapper.get_image_features for grad-cam metrics."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.get_image_features(x)


class TextFeatureExtractor(torch.nn.Module):
    """Wraps ClipWrapper.get_text_features for grad-cam metrics."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.get_text_features(x)


def image_transform(t, height=7, width=7):
    """Transformation for CAM (image)."""
    if t.size(1) == 1:
        t = t.permute(1, 0, 2)
    result = t[:, 1:, :].reshape(t.size(0), height, width, t.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def text_transform(t):
    """Transformation for CAM (text)."""
    if t.size(1) == 1:
        t = t.permute(1, 0, 2)
    result = t[:, :, :].reshape(t.size(0), 1, -1, t.size(2))
    return result


# ============================================================
# [ADD] Model Loading
# ============================================================

def load_model(
    clip_model_name: str = "ViT-B/32",
    device: torch.device = None,
):
    """
    Load OpenAI CLIP and wrap with ClipWrapper.

    Args:
        clip_model_name: CLIP model variant.
                         Options: ViT-B/32, ViT-B/16, ViT-L/14
        device:          torch.device. Defaults to cuda if available.

    Returns:
        model:      ClipWrapper instance
        preprocess: CLIP image preprocessing transform
    """
    import clip
    from scripts.clip_wrapper import ClipWrapper

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_model, preprocess = clip.load(
        clip_model_name,
        device=device,
        jit=False,
    )
    base_model = base_model.float() 
    model = ClipWrapper(base_model).to(device)

    print(f"[utils] Loaded CLIP {clip_model_name} on {device}")
    print(
        f"[utils] Vision blocks: "
        f"{len(model.vision_model.transformer.resblocks)}"
    )
    print(
        f"[utils] Text blocks:   "
        f"{len(model.text_model.transformer.resblocks)}"
    )

    return model, preprocess


# ============================================================
# [ADD] Image Loading
# ============================================================

def load_image(
    image_path: str,
    preprocess,
    device: torch.device = None,
):
    """
    Load an image from disk and apply CLIP preprocessing.

    Args:
        image_path: Path to image file
        preprocess: CLIP preprocess transform from clip.load()
        device:     Target device

    Returns:
        image_tensor: [1, 3, 224, 224] tensor on device
        pil_image:    Original PIL Image (for visualization)
    """
    from PIL import Image

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pil_image    = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(pil_image).unsqueeze(0).to(device)

    return image_tensor, pil_image


def load_image_from_url(
    url: str,
    preprocess,
    device: torch.device = None,
    timeout: int = 5,
):
    """
    Load an image from a URL and apply CLIP preprocessing.

    Args:
        url:        Image URL
        preprocess: CLIP preprocess transform
        device:     Target device
        timeout:    Request timeout in seconds

    Returns:
        image_tensor: [1, 3, 224, 224] tensor on device
        pil_image:    Original PIL Image
    """
    import requests
    from PIL import Image
    from io import BytesIO

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    response     = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    pil_image    = Image.open(BytesIO(response.content)).convert("RGB")
    image_tensor = preprocess(pil_image).unsqueeze(0).to(device)

    return image_tensor, pil_image


def pil_to_numpy(pil_image, size: int = 224) -> np.ndarray:
    """
    Convert PIL image to [H, W, 3] float32 numpy array in [0, 1].
    Resizes to size x size for consistent visualization.

    Args:
        pil_image: PIL.Image
        size:      Output spatial size

    Returns:
        np.ndarray [size, size, 3] in [0, 1]
    """
    from PIL import Image

    pil_image = pil_image.convert("RGB").resize(
        (size, size), Image.BILINEAR
    )
    return np.array(pil_image).astype(np.float32) / 255.0


# ============================================================
# [ADD] Text Loading
# ============================================================

def load_text(
    text: str,
    device: torch.device = None,
):
    """
    Tokenize text using OpenAI CLIP tokenizer.

    Args:
        text:   Input string
        device: Target device

    Returns:
        text_tensor: [1, 77] token tensor on device
    """
    import clip

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return clip.tokenize([text]).to(device)


def decode_tokens(
    text_tensor: torch.Tensor,
    skip_special: bool = False,
) -> list:
    """
    Decode CLIP token IDs back to string tokens.

    Args:
        text_tensor:  [1, seq_len] token tensor
        skip_special: If True, skip SOT (49406) and EOT (49407)

    Returns:
        List of token strings
    """
    from clip.simple_tokenizer import SimpleTokenizer

    tokenizer = SimpleTokenizer()
    ids       = text_tensor[0].cpu().tolist()

    tokens = []
    for id_ in ids:
        if id_ == 0:
            break
        if skip_special and id_ in (49406, 49407):
            continue
        tokens.append(tokenizer.decode([id_]))

    return tokens


# ============================================================
# [ADD] Results CSV
# ============================================================

def save_results_csv(
    results: dict,
    output_path: str,
):
    """
    Append a results row to a CSV file.
    Creates the file with headers if it does not exist.

    Args:
        results:     dict of {column_name: value}
        output_path: Path to CSV file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = output_path.exists()

    with open(output_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results.keys()))

        if not file_exists:
            writer.writeheader()

        writer.writerow(results)


# ============================================================
# [ADD] AverageMeter
# ============================================================

class AverageMeter:
    """
    Tracks running mean of a metric across samples.

    Usage:
        meter = AverageMeter("vdrop")
        for batch in batches:
            meter.update(value)
        print(meter.avg)
    """

    def __init__(self, name: str = ""):
        self.name  = name
        self.reset()

    def reset(self):
        self.val   = 0.0
        self.avg   = 0.0
        self.sum   = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count

    def __repr__(self):
        return f"AverageMeter({self.name}: avg={self.avg:.4f}, n={self.count})"


# ============================================================
# [ADD] Output Directory Setup
# ============================================================

def setup_output_dirs(base: str = "outputs") -> dict:
    """
    Create standard output directory structure.

    Args:
        base: Root output directory path

    Returns:
        dict of {name: path} for each subdirectory
    """
    dirs = {
        "heatmaps":    os.path.join(base, "heatmaps"),
        "metrics":     os.path.join(base, "metrics"),
        "logs":        os.path.join(base, "logs"),
        "checkpoints": os.path.join(base, "checkpoints"),
    }

    for name, path in dirs.items():
        os.makedirs(path, exist_ok=True)

    return dirs


# ============================================================
# [ADD] Config Loader
# ============================================================

def load_config(config_path: str = "configs/default.yaml") -> dict:
    """
    Load experiment configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        dict of configuration values
    """
    import yaml

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config