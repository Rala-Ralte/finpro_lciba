
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# font params
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.size"] = 12

# method labels
m_names = ["LC-IBA", "M2IB", "LibraGrad", "Fusion"]
im_vals = [0.0830, 0.1239, 0.1094, 0.0911]
im_stds = [0.0637, 0.0450, 0.0465, 0.0742]
tx_vals = [0.1008, 0.0713, 0.1060, 0.0902]
tx_stds = [0.0531, 0.0507, 0.0594, 0.0551]

idx_x = np.arange(len(m_names))
w_bar = 0.36

fig_p, ax_sub = plt.subplots(figsize=(6.5, 4.4))

bar_1 = ax_sub.bar(
    idx_x - w_bar / 2,
    im_vals,
    w_bar,
    label="Image",
    color="#2980b9",
    yerr=im_stds,
    capsize=4,
    alpha=0.9,
)
bar_2 = ax_sub.bar(
    idx_x + w_bar / 2,
    tx_vals,
    w_bar,
    label="Text",
    color="#e67e22",
    yerr=tx_stds,
    capsize=4,
    alpha=0.9,
)

ax_sub.set_ylabel(r"ROAR+ score $(l_c - l_o)/l_o$")
ax_sub.set_title("ROAR+ Faithfulness (higher is better)")
ax_sub.set_xticks(idx_x)
ax_sub.set_xticklabels(m_names)
ax_sub.legend(frameon=False)
ax_sub.grid(True, axis="y", alpha=0.3)
ax_sub.axhline(0, color="black", lw=0.8)

fig_p.tight_layout()
fig_p.savefig("roar_plus_barplot.pdf", bbox_inches="tight")
print("  saved roar_plus_barplot.pdf")
plt.close(fig_p)
print("Done.")

