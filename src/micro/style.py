"""Figure design tokens (docs/DESIGN_SYSTEM.md). Never hard-code these values."""

from __future__ import annotations

TOKENS = {
    "ink": "#111318",
    "grid": "#D6DAE1",
    "baseline": "#6B7280",
    "model": "#3B6EA5",
    "microprice": "#8A5A00",
    "pass": "#2E7D32",
    "fail": "#B3261E",
    "muted": "#6B7280",
}

FIGSIZE = (6.5, 4.2)
DPI = 160


def apply_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": TOKENS["ink"],
            "axes.labelcolor": TOKENS["ink"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": TOKENS["grid"],
            "grid.linewidth": 0.6,
            "grid.alpha": 0.7,
            "text.color": TOKENS["ink"],
            "xtick.color": TOKENS["ink"],
            "ytick.color": TOKENS["ink"],
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "legend.frameon": False,
            "lines.linewidth": 1.4,
            "savefig.dpi": DPI,
        }
    )
