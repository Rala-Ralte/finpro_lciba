# patches layers with modified ops


import torch
import torch.nn as nn
import torch.nn.functional as F
import requests
from transformers import LxmertForQuestionAnswering, LxmertTokenizer
from transformers.activations import GELUActivation

# dev = "cpu"
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# global flag
global_libra_mode = True


def set_libra_mode(val_flag: bool):
    global global_libra_mode
    global_libra_mode = val_flag


def swap_backward(f_out, b_out):
    # stop grad on f_out and graft b_out grad
    return f_out.detach() + (b_out - b_out.detach())


def constant_op(val_in):
    # zero grad passthrough
    return val_in.detach()


class LibraLayerNorm(nn.LayerNorm):
    @classmethod
    def from_layernorm(cls, old_ln: nn.LayerNorm) -> "LibraLayerNorm":
        new_obj = cls(
            old_ln.normalized_shape,
            eps=old_ln.eps,
            elementwise_affine=old_ln.elementwise_affine,
        )
        if old_ln.elementwise_affine:
            new_obj.weight = old_ln.weight
            new_obj.bias = old_ln.bias
        return new_obj

    def forward(self, t_in: torch.Tensor) -> torch.Tensor:
        if not global_libra_mode:
            return super().forward(t_in)
        m_vec = t_in.mean(dim=-1, keepdim=True)
        v_vec = t_in.var(dim=-1, keepdim=True, unbiased=False)
        d_val = constant_op(torch.rsqrt(v_vec + self.eps))
        norm_t = (t_in - m_vec.detach()) * d_val
        if self.elementwise_affine:
            norm_t = norm_t * self.weight + self.bias
        return norm_t


class LibraGELU(nn.Module):
    def forward(self, t_in: torch.Tensor) -> torch.Tensor:
        if not global_libra_mode:
            return F.gelu(t_in)
        # 1.702 approx constant
        sig_gate = constant_op(torch.sigmoid(1.702 * t_in))
        return swap_backward(F.gelu(t_in), t_in * sig_gate)


def patch_layernorms(m_net: nn.Module) -> int:
    cnt_ln = 0
    for str_name, mod_inst in list(m_net.named_modules()):
        for sub_attr, ch_obj in list(mod_inst.named_children()):
            if isinstance(ch_obj, nn.LayerNorm):
                setattr(mod_inst, sub_attr, LibraLayerNorm.from_layernorm(ch_obj))
                cnt_ln += 1
    return cnt_ln


def patch_gelu(m_net: nn.Module) -> int:
    cnt_g = 0
    for str_name, mod_inst in list(m_net.named_modules()):
        for sub_attr, ch_obj in list(mod_inst.named_children()):
            if isinstance(ch_obj, (GELUActivation, nn.GELU)):
                setattr(mod_inst, sub_attr, LibraGELU())
                cnt_g += 1
    return cnt_g


def load_lxmert(
    m_name: str = "unc-nlp/lxmert-vqa-uncased",
    t_name: str = "unc-nlp/lxmert-base-uncased",
    c_dir: str = None,
    d_target: torch.device = None,
):
    import os

    if c_dir:
        os.environ["HF_HOME"] = c_dir
        os.environ["TRANSFORMERS_CACHE"] = c_dir

    if d_target is None:
        d_target = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading " + str(m_name) + "...")
    tok_obj = LxmertTokenizer.from_pretrained(t_name)
    lx_mod = LxmertForQuestionAnswering.from_pretrained(m_name).to(d_target)
    lx_mod.eval()

    # print config info
    print("  Parameters:      " + f"{sum(p.numel() for p in lx_mod.parameters()):,}")
    print("  Hidden size:      " + str(lx_mod.config.hidden_size))
    print("  Attention heads: " + str(lx_mod.config.num_attention_heads))
    print("  Layers:           " + str(lx_mod.config.num_hidden_layers))
    print(
        "  Cross-attention: "
        + str(len(lx_mod.lxmert.encoder.x_layers))
        + " x LxmertXLayer"
    )

    return lx_mod, tok_obj


def load_libra_lxmert(
    m_name: str = "unc-nlp/lxmert-vqa-uncased",
    c_dir: str = None,
    d_target: torch.device = None,
    verify: bool = True,
    base_model=None,
    tokenizer=None,
) -> LxmertForQuestionAnswering:
    import os

    if c_dir:
        os.environ["HF_HOME"] = c_dir
        os.environ["TRANSFORMERS_CACHE"] = c_dir

    if d_target is None:
        d_target = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lib_mod = LxmertForQuestionAnswering.from_pretrained(m_name).to(d_target)
    lib_mod.eval()

    num_ln = patch_layernorms(lib_mod)
    num_glu = patch_gelu(lib_mod)

    print("  LibraLayerNorm patches: " + str(num_ln))
    print("  LibraGELU patches:      " + str(num_glu))

    if verify and base_model is not None and tokenizer is not None:
        verify_libra_model(base_model, lib_mod, tokenizer, d_target)

    return lib_mod


def load_id2label(net_obj: LxmertForQuestionAnswering) -> dict:
    try:
        dict_lbls = net_obj.config.id2label
        if len(dict_lbls) >= 100:
            print("  id2label from config: " + str(len(dict_lbls)) + " labels")
            return dict_lbls
    except Exception:
        pass

    print("  Downloading VQA answer labels...")
    raw_url = (
        "https://raw.githubusercontent.com/airsplay/lxmert/"
        "master/data/vqa/trainval_label2ans.json"
    )
    arr_ans = requests.get(raw_url, timeout=10).json()
    dict_lbls = {idx: txt for idx, txt in enumerate(arr_ans)}
    print("  id2label downloaded: " + str(len(dict_lbls)) + " labels")
    return dict_lbls


def verify_libra_model(
    m_std: LxmertForQuestionAnswering,
    m_lib: LxmertForQuestionAnswering,
    tok_obj: LxmertTokenizer,
    d_target: torch.device,
    q_txt: str = "how many cats are there?",
    f_t: torch.Tensor = None,
    b_t: torch.Tensor = None,
    tol_thr: float = 0.01,
) -> bool:
    global global_libra_mode

    enc_d = tok_obj(
        q_txt,
        padding="max_length",
        max_length=20,
        truncation=True,
        return_tensors="pt",
    ).to(d_target)

    if f_t is None:
        f_t = torch.zeros(1, 36, 2048).to(d_target)
        b_t = torch.zeros(1, 36, 4).to(d_target)
    else:
        f_t = f_t.to(d_target)
        b_t = b_t.to(d_target)

    kw_in = dict(
        input_ids=enc_d["input_ids"],
        attention_mask=enc_d["attention_mask"],
        token_type_ids=enc_d["token_type_ids"],
        visual_feats=f_t,
        visual_pos=b_t,
    )

    global_libra_mode = False
    with torch.no_grad():
        out_a = m_std(**kw_in)

    global_libra_mode = True
    with torch.no_grad():
        out_b = m_lib(**kw_in)

    max_delta = (
        out_a.question_answering_score - out_b.question_answering_score
    ).abs().max().item()

    is_valid = max_delta < tol_thr
    status_str = "✓" if is_valid else "✗ WARN"
    print("  Max logit diff: " + f"{max_delta:.6f}" + "  " + status_str)

    if not is_valid:
        print("  WARNING: logit drift " + f"{max_delta:.4f}" + " > " + str(tol_thr))

    return is_valid
