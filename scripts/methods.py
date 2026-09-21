import numpy as np
import torch
from scripts.chefer import vision_heatmap_chefer, text_heatmap_chefer

from scripts.iba import IBAInterpreter, Estimator
from scripts.libragrad import (
    compute_libragrad_vision_saliency,
    compute_libragrad_text_saliency,
    compute_fusion_map,
    prepare_vision_prior,
    prepare_text_prior,
    compute_libragrad_both,
)
from scripts.utils import normalize

# dvc = 'cpu'
# if torch.cuda.is_available(): dvc = 'cuda:0'
dvc = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_feature_map(net_obj, idx_l, in_t):
    with torch.no_grad():
        out_dict = net_obj(in_t, output_hidden_states=True)
        # skip initial embed block
        t_feats = out_dict["hidden_states"][idx_l + 1]
    return t_feats


def extract_bert_layer(net_obj, idx_l):
    for _, mod_1 in net_obj.named_children():
        for str_k, mod_2 in mod_1.named_children():
            if str_k in ("layers", "resblocks"):
                for block_id, sub_mod in mod_2.named_children():
                    if block_id == str(idx_l):
                        return sub_mod
    raise ValueError("layer " + str(idx_l) + " not found in architecture")


def get_compression_estimator(s_var, target_l, f_maps):
    est_inst = Estimator(target_l)
    est_inst.M = torch.zeros_like(f_maps)
    est_inst.S = s_var * np.ones(f_maps.shape)
    est_inst.N = 1
    return est_inst


def text_heatmap_iba(
    t_tokens,
    img_t,
    m_wrap,
    l_idx,
    beta_val,
    var_val,
    step_sz=1.0,
    n_iters=10,
    show_bar=True,
):
    h_states = extract_feature_map(m_wrap.text_model, l_idx, t_tokens)
    t_layer = extract_bert_layer(m_wrap.text_model, l_idx)
    est_obj = get_compression_estimator(var_val, t_layer, h_states)

    iba_run = IBAInterpreter(
        model=m_wrap,
        estim=est_obj,
        beta=beta_val,
        lr=step_sz,
        steps=n_iters,
        progbar=show_bar,
        use_libragrad=False,
        libragrad_init=False,
    )
    return iba_run.text_heatmap(t_tokens, img_t)


def vision_heatmap_iba(
    t_tokens,
    img_t,
    m_wrap,
    l_idx,
    beta_val,
    var_val,
    step_sz=1.0,
    n_iters=10,
    show_bar=True,
):
    h_states = extract_feature_map(m_wrap.vision_model, l_idx, img_t)
    v_layer = extract_bert_layer(m_wrap.vision_model, l_idx)
    est_obj = get_compression_estimator(var_val, v_layer, h_states)

    iba_run = IBAInterpreter(
        model=m_wrap,
        estim=est_obj,
        beta=beta_val,
        lr=step_sz,
        steps=n_iters,
        progbar=show_bar,
        use_libragrad=False,
        libragrad_init=False,
    )
    return iba_run.vision_heatmap(t_tokens, img_t)


def text_heatmap_lciba(
    t_tokens,
    img_t,
    m_wrap,
    l_idx,
    beta_val,
    var_val,
    step_sz=1.0,
    n_iters=10,
    show_bar=True,
    prior_strength=3.0,
):
    t_layer = extract_bert_layer(m_wrap.text_model, l_idx)

    raw_tmap = compute_libragrad_text_saliency(
        model=m_wrap,
        image=img_t,
        text=t_tokens,
        target_layer=t_layer,
        use_libragrad=True,
    )

    h_states = extract_feature_map(m_wrap.text_model, l_idx, t_tokens)
    dim_h = h_states.shape[-1]

    t_prior = prepare_text_prior(raw_tmap, hidden_dim=dim_h)
    est_obj = get_compression_estimator(var_val, t_layer, h_states)

    iba_run = IBAInterpreter(
        model=m_wrap,
        estim=est_obj,
        beta=beta_val,
        lr=step_sz,
        steps=n_iters,
        progbar=show_bar,
        use_libragrad=False,
        libragrad_init=True,
    )

    iba_run.bottleneck.libragrad_initialize_alpha(
        t_prior, strength=prior_strength
    )
    return iba_run.text_heatmap(t_tokens, img_t, prior=t_prior)


def vision_heatmap_lciba(
    t_tokens,
    img_t,
    m_wrap,
    l_idx,
    beta_val,
    var_val,
    step_sz=1.0,
    n_iters=10,
    show_bar=True,
    prior_strength=3.0,
):
    v_layer = extract_bert_layer(m_wrap.vision_model, l_idx)

    raw_vmap = compute_libragrad_vision_saliency(
        model=m_wrap,
        image=img_t,
        text=t_tokens,
        target_layer=v_layer,
        use_libragrad=True,
    )

    h_states = extract_feature_map(m_wrap.vision_model, l_idx, img_t)
    dim_h = h_states.shape[-1]

    n_patches = h_states.shape[1] - 1
    sz_grid = int(n_patches**0.5)

    v_prior = prepare_vision_prior(
        raw_vmap,
        hidden_dim=dim_h,
        grid_size=sz_grid,
    )

    est_obj = get_compression_estimator(var_val, v_layer, h_states)

    iba_run = IBAInterpreter(
        model=m_wrap,
        estim=est_obj,
        beta=beta_val,
        lr=step_sz,
        steps=n_iters,
        progbar=show_bar,
        use_libragrad=True,
        libragrad_init=False,
    )

    return iba_run.vision_heatmap(t_tokens, img_t, prior=None)


def text_heatmap_libragrad(
    t_tokens,
    img_t,
    m_wrap,
    l_idx,
    **kwargs,
):
    t_layer = extract_bert_layer(m_wrap.text_model, l_idx)

    return compute_libragrad_text_saliency(
        model=m_wrap,
        image=img_t,
        text=t_tokens,
        target_layer=t_layer,
        use_libragrad=True,
    )


def vision_heatmap_libragrad(
    t_tokens,
    img_t,
    m_wrap,
    l_idx,
    **kwargs,
):
    v_layer = extract_bert_layer(m_wrap.vision_model, l_idx)

    return compute_libragrad_vision_saliency(
        model=m_wrap,
        image=img_t,
        text=t_tokens,
        target_layer=v_layer,
        use_libragrad=True,
    )


def text_heatmap_fusion(
    t_tokens,
    img_t,
    m_wrap,
    l_idx,
    beta_val,
    var_val,
    step_sz=1.0,
    n_iters=10,
    show_bar=True,
):
    m_iba = text_heatmap_iba(
        t_tokens,
        img_t,
        m_wrap,
        l_idx,
        beta_val,
        var_val,
        step_sz,
        n_iters,
        show_bar,
    )
    m_lg = text_heatmap_libragrad(t_tokens, img_t, m_wrap, l_idx)

    cut_len = min(len(m_iba), len(m_lg))

    return compute_fusion_map(
        m_iba[:cut_len],
        m_lg[:cut_len],
    )


def vision_heatmap_fusion(
    t_tokens,
    img_t,
    m_wrap,
    l_idx,
    beta_val,
    var_val,
    step_sz=1.0,
    n_iters=10,
    show_bar=True,
):
    m_iba = vision_heatmap_iba(
        t_tokens,
        img_t,
        m_wrap,
        l_idx,
        beta_val,
        var_val,
        step_sz,
        n_iters,
        show_bar,
    )
    m_lg = vision_heatmap_libragrad(t_tokens, img_t, m_wrap, l_idx)

    return compute_fusion_map(m_iba, m_lg)


VISION_METHODS = {
    "m2ib": vision_heatmap_iba,
    "lciba": vision_heatmap_lciba,
    "libragrad": vision_heatmap_libragrad,
    "fusion": vision_heatmap_fusion,
    "chefer": vision_heatmap_chefer,
}

TEXT_METHODS = {
    "m2ib": text_heatmap_iba,
    "lciba": text_heatmap_lciba,
    "libragrad": text_heatmap_libragrad,
    "fusion": text_heatmap_fusion,
    "chefer": text_heatmap_chefer,
}
