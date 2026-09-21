
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42      # editable text in PDF
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 12

PERT_STEPS = [0, 0.25, 0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]

# ── Final results (N=4691) ───────────────────────────────────
# Text modality, MIF
mif = {
    "CMEGrad":  [0.965, 0.665, 0.583, 0.471, 0.357, 0.263, 0.102, 0.083, 0.083],
    "Chefer":   [0.965, 0.733, 0.511, 0.272, 0.201, 0.158, 0.090, 0.083, 0.083],
    "Random":   [0.965, 0.709, 0.499, 0.254, 0.187, 0.144, 0.087, 0.083, 0.083],
}

# Text modality, LIF
lif = {
    "CMEGrad":  [0.965, 0.608, 0.429, 0.253, 0.197, 0.154, 0.089, 0.083, 0.083],
    "Chefer":   [0.965, 0.736, 0.527, 0.272, 0.192, 0.152, 0.090, 0.083, 0.083],
    "Random":   [0.965, 0.706, 0.494, 0.265, 0.195, 0.150, 0.089, 0.083, 0.083],
}

styles = {
    "CMEGrad": dict(color="#c0392b", marker="o", linewidth=2.2, markersize=6),
    "Chefer":  dict(color="#2980b9", marker="s", linewidth=2.0, markersize=5,
                    linestyle="--"),
    "Random":  dict(color="#7f8c8d", marker="^", linewidth=1.8, markersize=5,
                    linestyle=":"),
}

def make_plot(data, title, ylabel, outname):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for method, vals in data.items():
        ax.plot(PERT_STEPS, vals, label=method, **styles[method])
    ax.set_xlabel("Fraction of tokens removed")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    fig.tight_layout()
    fig.savefig(outname, bbox_inches="tight")
    print(f"  saved {outname}")
    plt.close(fig)

make_plot(
    mif,
    "Text Perturbation — MIF (Most Important First)",
    "VQA soft accuracy",
    "perturbation_mif.pdf",
)

make_plot(
    lif,
    "Text Perturbation — LIF (Least Important First)",
    "VQA soft accuracy",
    "perturbation_lif.pdf",
)

print("Done.")
