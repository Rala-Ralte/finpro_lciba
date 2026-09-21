import os
import numpy as np
import matplotlib
# force agg
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.text import Text
from PIL import Image


def _to_display_image(arr_in: np.ndarray) -> np.ndarray:
    t_img = arr_in.astype(np.float32)
    if t_img.max() > 1.0:
        t_img = t_img / 255.0
    return np.clip(t_img, 0.0, 1.0)


def _reverse_clip_normalization(raw_chw: np.ndarray) -> np.ndarray:
    m_arr = np.array([0.48145466, 0.4578275, 0.40821073])
    s_arr = np.array([0.26862954, 0.26130258, 0.40821073])

    tmp_t = raw_chw.transpose(1, 2, 0)
    tmp_t = tmp_t * s_arr + m_arr
    return np.clip(tmp_t, 0.0, 1.0)


def _generate_token_colors(
    vals_arr: np.ndarray,
    min_a: float = 0.15,
    max_a: float = 0.85,
) -> list:
    val_lo = vals_arr.min()
    val_hi = vals_arr.max()

    if val_hi - val_lo < 1e-8:
        scaled_s = np.zeros_like(vals_arr)
    else:
        scaled_s = (vals_arr - val_lo) / (val_hi - val_lo)

    res_cols = []
    for sc in scaled_s:
        cur_a = min_a + sc * (max_a - min_a)
        res_cols.append((0.2, 0.4, 1.0, cur_a))

    return res_cols


def overlay_vision_heatmap(img_arr, hm_arr, alpha_v=0.5, cmap_name="jet"):
    import matplotlib.cm as cm
    from scripts.utils import normalize
    img_arr = _to_display_image(img_arr)
    hm_arr = np.clip(normalize(hm_arr), 0.0, 1.0)
    col_map = cm.get_cmap(cmap_name)(hm_arr)[:, :, :3]
    out_blend = (1 - alpha_v) * img_arr + alpha_v * col_map
    return np.clip(out_blend, 0.0, 1.0)


class _TokenText(Text):
    def __init__(self, pos_x, pos_y, str_txt, bg_c, *args, **kwargs):
        super().__init__(pos_x, pos_y, str_txt, *args, **kwargs)
        self.bgcolor = bg_c

    def draw(self, p_render, *args, **kwargs):
        c_r, c_g, c_b, c_a = self.bgcolor
        b_box = dict(
            facecolor=(c_r, c_g, c_b),
            edgecolor=(c_r, c_g, c_b),
            boxstyle="round,pad=0.01",
            alpha=c_a,
        )
        self.set_bbox(b_box)
        super().draw(p_render, *args, **kwargs)


def _plot_tokens(
    ax_obj,
    t_strs: list,
    c_list: list,
    lim_w: float,
    lim_h: float,
    f_sz: int = 12,
):
    cur_x = 0.0
    cur_y = lim_h
    gap_w = f_sz * 0.2

    for count_i, (t_tok, c_val) in enumerate(zip(t_strs, c_list)):
        elem_t = _TokenText(
            cur_x, cur_y * 0.7, t_tok, c_val,
            fontsize=f_sz,
        )
        ax_obj.add_artist(elem_t)

        w_token = f_sz * 0.4 * len(t_tok)

        if count_i + 1 < len(t_strs):
            w_next = f_sz * 0.4 * len(t_strs[count_i + 1]) + gap_w
            if cur_x + w_token + w_next >= lim_w:
                cur_x = 0.0
                cur_y -= 1.0
            else:
                cur_x += w_token + gap_w
        else:
            cur_x += w_token


def visualize_single(
    im_data: np.ndarray,
    v_map: np.ndarray,
    t_map: np.ndarray,
    t_list: list,
    t_title: str = None,
    dest_path: str = None,
    flag_show: bool = False,
    b_boxes: list = None,
) -> plt.Figure:
    sub_toks = [t.split("<")[0] for t in t_list[1:-1]]
    sub_tmap = t_map[1:-1] if len(t_map) > 2 else t_map

    fig_obj, sub_axes = plt.subplots(1, 2, figsize=(10, 4))

    # vision plot
    v_blend = overlay_vision_heatmap(im_data, v_map)
    sub_axes[0].imshow(v_blend)

    if b_boxes:
        for x_b, y_b, w_b, h_b in b_boxes:
            box_patch = mpatches.Rectangle(
                (x_b, y_b), w_b, h_b,
                linewidth=2, edgecolor="red", facecolor="none",
            )
            sub_axes[0].add_patch(box_patch)

    sub_axes[0].axis("off")
    sub_axes[0].set_title("Vision Saliency", fontsize=10)

    # text plot
    pal = _generate_token_colors(sub_tmap)
    _plot_tokens(
        sub_axes[1], sub_toks, pal,
        lim_w=100, lim_h=4, f_sz=13,
    )
    sub_axes[1].set_xlim(0, 100)
    sub_axes[1].set_ylim(-4, 4)
    sub_axes[1].axis("off")
    sub_axes[1].set_title("Text Attribution", fontsize=10)

    if t_title:
        fig_obj.suptitle(t_title, fontsize=11, y=1.01)

    plt.tight_layout()

    if dest_path:
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        plt.savefig(dest_path, bbox_inches="tight", dpi=150)

    if flag_show:
        plt.show()

    return fig_obj


def visualize_comparison(
    im_data: np.ndarray,
    v_maps_dict: dict,
    t_maps_dict: dict,
    t_list: list,
    t_title: str = None,
    dest_path: str = None,
    flag_show: bool = False,
) -> plt.Figure:
    m_keys = list(v_maps_dict.keys())
    cnt_m = len(m_keys)

    sub_toks = [t.split("<")[0] for t in t_list[1:-1]]

    fig_obj, sub_axes = plt.subplots(cnt_m, 2, figsize=(12, 3.5 * cnt_m))

    if cnt_m == 1:
        sub_axes = [sub_axes]

    for count_i, m_name in enumerate(m_keys):
        cur_v = v_maps_dict[m_name]
        cur_t = t_maps_dict[m_name]

        sub_t = cur_t[1:-1] if len(cur_t) > 2 else cur_t

        v_blend = overlay_vision_heatmap(im_data, cur_v)
        sub_axes[count_i][0].imshow(v_blend)
        sub_axes[count_i][0].axis("off")
        sub_axes[count_i][0].set_title(str(m_name) + " — Vision", fontsize=9)

        pal = _generate_token_colors(sub_t)
        _plot_tokens(
            sub_axes[count_i][1], sub_toks, pal,
            lim_w=100, lim_h=4, f_sz=12,
        )
        sub_axes[count_i][1].set_xlim(0, 100)
        sub_axes[count_i][1].set_ylim(-4, 4)
        sub_axes[count_i][1].axis("off")
        sub_axes[count_i][1].set_title(str(m_name) + " — Text", fontsize=9)

    if t_title:
        fig_obj.suptitle(t_title, fontsize=12, y=1.01)

    plt.tight_layout()

    if dest_path:
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        plt.savefig(dest_path, bbox_inches="tight", dpi=150)

    if flag_show:
        plt.show()

    plt.close(fig_obj)
    return fig_obj


def plot_metrics_comparison(
    data_metrics: dict,
    m_keys: list = None,
    t_title: str = "Method Comparison",
    dest_path: str = None,
    flag_show: bool = False,
) -> plt.Figure:
    if m_keys is None:
        m_keys = ["vdrop", "vincr", "tdrop", "tincr"]

    arr_methods = list(data_metrics.keys())
    tot_metrics = len(m_keys)
    tot_methods = len(arr_methods)

    x_coords = np.arange(tot_metrics)
    b_width = 0.8 / tot_methods
    palette_c = plt.cm.tab10(np.linspace(0, 0.5, tot_methods))

    fig_obj, ax_sub = plt.subplots(figsize=(10, 5))

    for idx_m, (cur_meth, col_v) in enumerate(zip(arr_methods, palette_c)):
        bar_vals = [data_metrics[cur_meth].get(k_m, 0.0) for k_m in m_keys]
        ax_sub.bar(
            x_coords + idx_m * b_width - (tot_methods - 1) * b_width / 2,
            bar_vals,
            b_width,
            label=cur_meth,
            color=col_v,
            alpha=0.85,
        )

    ax_sub.set_xticks(x_coords)
    ax_sub.set_xticklabels(m_keys, fontsize=11)
    ax_sub.set_ylabel("Score", fontsize=11)
    ax_sub.set_title(t_title, fontsize=12)
    ax_sub.legend(fontsize=10)
    ax_sub.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if dest_path:
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        plt.savefig(dest_path, bbox_inches="tight", dpi=150)

    if flag_show:
        plt.show()

    plt.close(fig_obj)
    return fig_obj


def plot_layer_benchmark(
    res_layers: dict,
    target_metric: str = "vdrop",
    t_title: str = "Layer Benchmark",
    dest_path: str = None,
    flag_show: bool = False,
) -> plt.Figure:
    fig_obj, ax_sub = plt.subplots(figsize=(10, 5))

    for cur_meth, dict_l in res_layers.items():
        keys_sorted = sorted(dict_l.keys())
        plot_pts = [dict_l[l_k] for l_k in keys_sorted]
        ax_sub.plot(keys_sorted, plot_pts, marker="o", label=cur_meth)

    ax_sub.set_xlabel("Layer Index", fontsize=11)
    ax_sub.set_ylabel(target_metric, fontsize=11)
    ax_sub.set_title(t_title, fontsize=12)
    ax_sub.legend(fontsize=10)
    ax_sub.grid(alpha=0.3)

    plt.tight_layout()

    if dest_path:
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        plt.savefig(dest_path, bbox_inches="tight", dpi=150)

    if flag_show:
        plt.show()

    plt.close(fig_obj)
    return fig_obj
