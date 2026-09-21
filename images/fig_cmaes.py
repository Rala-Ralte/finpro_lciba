
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 12

# ── Final results (N=5000) ───────────────────────────────────
methods = ["CMEGrad", "Baseline\n(uniform vision)"]
H_vis  = [2.4897, 3.5835]
H_lang = [0.1065, 0.1065]
H_gap  = [2.4240, 3.4770]
H_vis_err  = [1.2119, 0.0000]
H_gap_err  = [1.2066, 0.3206]

x = np.arange(len(methods))
width = 0.25

fig, ax = plt.subplots(figsize=(6.5, 4.4))

b1 = ax.bar(x - width, H_vis,  width, label=r"$H(\mathrm{vision})$",
            color="#2980b9", yerr=H_vis_err, capsize=4, alpha=0.9)
b2 = ax.bar(x,         H_lang, width, label=r"$H(\mathrm{language})$",
            color="#27ae60", alpha=0.9)
b3 = ax.bar(x + width, H_gap,  width, label=r"$H_{\mathrm{gap}}$",
            color="#c0392b", yerr=H_gap_err, capsize=4, alpha=0.9)

ax.axhline(np.log(36), ls="--", color="gray", lw=1, alpha=0.7)
ax.text(1.32, np.log(36) + 0.04, r"$\ln(36)$", fontsize=10, color="gray")

ax.set_ylabel("Entropy (nats)")
ax.set_title("CMAES: Cross-Modal Attribution Entropy")
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.legend(frameon=False, loc="upper left")
ax.grid(True, axis="y", alpha=0.3)

# annotate improvement
ax.annotate("-30.3%", xy=(1 + width, H_gap[1]), xytext=(0 + width, H_gap[0] + 0.5),
            fontsize=11, color="#c0392b", fontweight="bold",
            ha="center")

fig.tight_layout()
fig.savefig("cmaes_barplot.pdf", bbox_inches="tight")
print("  saved cmaes_barplot.pdf")
plt.close(fig)
print("Done.")
