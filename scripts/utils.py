import os, csv, torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# dev = "cpu"
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize(x_arr: np.ndarray) -> np.ndarray:
    lo_v = x_arr.min()
    hi_v = x_arr.max()

    if hi_v - lo_v < 1e-8:
        return np.zeros_like(x_arr)

    return (x_arr - lo_v) / (hi_v - lo_v)


class mySequential(nn.Sequential):
    def forward(self, *arg_in, **kwargs):
        for sub_m in self._modules.values():
            if type(arg_in) == tuple:
                arg_in = sub_m(*arg_in)
            else:
                arg_in = sub_m(arg_in)
        return arg_in


def replace_layer(
    root_m: nn.Module,
    old_m: nn.Module,
    new_m: nn.Module,
):
    def replace_in(
        curr_m: nn.Module,
        tgt_m: nn.Module,
        sub_m: nn.Module,
    ) -> bool:
        for str_k, ch_mod in curr_m.named_children():
            if ch_mod == tgt_m:
                if isinstance(curr_m, nn.ModuleList):
                    curr_m[int(str_k)] = sub_m
                elif isinstance(curr_m, nn.Sequential):
                    curr_m[int(str_k)] = sub_m
                else:
                    curr_m.__setattr__(str_k, sub_m)
                return True
            elif len(list(ch_mod.named_children())) > 0:
                if replace_in(ch_mod, tgt_m, sub_m):
                    return True
        return False

    if not replace_in(root_m, old_m, new_m):
        raise RuntimeError("failed to swap module: " + str(old_m.__class__.__name__))


class CosSimilarity:
    def __init__(self, ref_f: torch.Tensor):
        self.features = ref_f

    def __call__(self, pred_t: torch.Tensor) -> torch.Tensor:
        fn_c = torch.nn.CosineSimilarity()
        return fn_c(pred_t, self.features)


class ImageFeatureExtractor(torch.nn.Module):
    def __init__(self, net_wrap):
        super().__init__()
        self.model = net_wrap

    def __call__(self, im_t: torch.Tensor) -> torch.Tensor:
        return self.model.get_image_features(im_t)


class TextFeatureExtractor(torch.nn.Module):
    def __init__(self, net_wrap):
        super().__init__()
        self.model = net_wrap

    def __call__(self, tx_t: torch.Tensor) -> torch.Tensor:
        return self.model.get_text_features(tx_t)


def image_transform(t_in, height=7, width=7):
    if t_in.size(1) == 1:
        t_in = t_in.permute(1, 0, 2)
    res_t = t_in[:, 1:, :].reshape(t_in.size(0), height, width, t_in.size(2))
    res_t = res_t.transpose(2, 3).transpose(1, 2)
    return res_t


def text_transform(t_in):
    if t_in.size(1) == 1:
        t_in = t_in.permute(1, 0, 2)
    res_t = t_in[:, :, :].reshape(t_in.size(0), 1, -1, t_in.size(2))
    return res_t


def load_model(
    m_name: str = "ViT-B/32",
    d_v: torch.device = None,
):
    import clip
    from scripts.clip_wrapper import ClipWrapper

    if d_v is None:
        d_v = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_net, tr_proc = clip.load(
        m_name,
        device=d_v,
        jit=False,
    )
    base_net = base_net.float()
    net_wrap = ClipWrapper(base_net).to(d_v)

    print("[utils] Loaded CLIP " + str(m_name) + " on " + str(d_v))
    print(
        "[utils] Vision blocks: "
        + str(len(net_wrap.vision_model.transformer.resblocks))
    )
    print(
        "[utils] Text blocks:   "
        + str(len(net_wrap.text_model.transformer.resblocks))
    )

    return net_wrap, tr_proc


def load_image(
    p_img: str,
    tr_proc,
    d_v: torch.device = None,
):
    from PIL import Image

    if d_v is None:
        d_v = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pil_im = Image.open(p_img).convert("RGB")
    tens_im = tr_proc(pil_im).unsqueeze(0).to(d_v)

    return tens_im, pil_im


def load_image_from_url(
    str_url: str,
    tr_proc,
    d_v: torch.device = None,
    t_out: int = 5,
):
    import requests
    from PIL import Image
    from io import BytesIO

    if d_v is None:
        d_v = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hdrs = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    req_res = requests.get(str_url, headers=hdrs, timeout=t_out)
    req_res.raise_for_status()
    pil_im = Image.open(BytesIO(req_res.content)).convert("RGB")
    tens_im = tr_proc(pil_im).unsqueeze(0).to(d_v)

    return tens_im, pil_im


def pil_to_numpy(pil_im, side_sz: int = 224) -> np.ndarray:
    from PIL import Image

    pil_im = pil_im.convert("RGB").resize(
        (side_sz, side_sz), Image.BILINEAR
    )
    return np.array(pil_im).astype(np.float32) / 255.0


def load_text(
    s_txt: str,
    d_v: torch.device = None,
):
    import clip

    if d_v is None:
        d_v = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return clip.tokenize([s_txt]).to(d_v)


def decode_tokens(
    tok_tensor: torch.Tensor,
    no_specials: bool = False,
) -> list:
    from clip.simple_tokenizer import SimpleTokenizer

    tok_engine = SimpleTokenizer()
    arr_ids = tok_tensor[0].cpu().tolist()

    out_toks = []
    for cur_id in arr_ids:
        if cur_id == 0:
            break
        if no_specials and cur_id in (49406, 49407):
            continue
        out_toks.append(tok_engine.decode([cur_id]))

    return out_toks


def save_results_csv(
    dict_vals: dict,
    dst_file: str,
):
    path_obj = Path(dst_file)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    is_present = path_obj.exists()

    with open(path_obj, "a", newline="") as fp:
        csv_w = csv.DictWriter(fp, fieldnames=list(dict_vals.keys()))

        if not is_present:
            csv_w.writeheader()

        csv_w.writerow(dict_vals)


class AverageMeter:
    def __init__(self, str_lbl: str = ""):
        self.name = str_lbl
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, num_val: float, num_n: int = 1):
        self.val = num_val
        self.sum += num_val * num_n
        self.count += num_n
        self.avg = self.sum / self.count

    def __repr__(self):
        return f"AverageMeter({self.name}: avg={self.avg:.4f}, n={self.count})"


def setup_output_dirs(root_base: str = "outputs") -> dict:
    sub_map = {
        "heatmaps": os.path.join(root_base, "heatmaps"),
        "metrics": os.path.join(root_base, "metrics"),
        "logs": os.path.join(root_base, "logs"),
        "checkpoints": os.path.join(root_base, "checkpoints"),
    }

    for _, p_dir in sub_map.items():
        os.makedirs(p_dir, exist_ok=True)

    return sub_map


def load_config(p_cfg: str = "configs/default.yaml") -> dict:
    import yaml

    with open(p_cfg, "r") as fp:
        raw_dict = yaml.safe_load(fp)

    return raw_dict

