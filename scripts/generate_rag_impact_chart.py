from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/home/sheng/cyber-ft")
OUT_DIR = ROOT / "docs/ieee_acsac_2026/IEEE_ACSAC_2026__Sun_/figures"
OUT_PNG = OUT_DIR / "rag_impact_chart_v2.png"


MODELS = [
    "DeepSeek",
    "Phi-4-mini",
    "Granite 4.1",
    "Gemma 4",
]
PANELS = [
    ("Mapped CVEs (n=903)", np.array([27.0, 18.5, 64.3, 70.4]), np.array([85.8, 92.7, 88.4, 90.7])),
    ("Unmapped CVEs (n=97)", np.array([13.4, 12.4, 56.7, 61.9]), np.array([47.4, 74.2, 59.8, 86.6])),
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    x = np.arange(len(MODELS))
    width = 0.22

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.2), dpi=300, sharey=True)
    handles = None

    for ax, (title, zero_shot, with_rag) in zip(axes, PANELS):
        gains = with_rag - zero_shot
        zero_bars = ax.bar(
            x - width / 2,
            zero_shot,
            width,
            label="Zero-shot",
            color="#8fb9e8",
            edgecolor="#4a4a4a",
            linewidth=0.25,
        )
        rag_bars = ax.bar(
            x + width / 2,
            with_rag,
            width,
            label="With RAG",
            color="#7fbf00",
            edgecolor="#4a4a4a",
            linewidth=0.25,
        )
        handles = (zero_bars, rag_bars)

        for bars in (zero_bars, rag_bars):
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 1.2,
                    f"{height:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    fontweight="bold",
                )

        for i, gain in enumerate(gains):
            y = max(zero_shot[i], with_rag[i]) + 6.0
            ax.text(
                x[i],
                y,
                f"+{gain:.1f}",
                ha="center",
                va="bottom",
                fontsize=6,
                color="#087f5b",
                fontweight="bold",
            )

        ax.set_title(title, fontsize=8, pad=4)
        ax.set_ylim(0, 104)
        ax.set_yticks(np.arange(0, 101, 20))
        ax.set_xticks(x)
        ax.set_xticklabels(MODELS)
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.45)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.6)
        ax.spines["bottom"].set_linewidth(0.6)
        ax.tick_params(axis="both", width=0.5, length=2.5)

    axes[0].set_ylabel("Strict accuracy (%)")
    if handles is not None:
        fig.legend(
            handles,
            ["Zero-shot", "With RAG"],
            loc="upper center",
            bbox_to_anchor=(0.5, 1.03),
            ncol=2,
            frameon=False,
            handlelength=1.3,
            columnspacing=1.6,
        )

    fig.tight_layout(pad=0.2, w_pad=0.8, rect=[0, 0, 1, 0.94])
    fig.savefig(OUT_PNG, bbox_inches="tight")
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
