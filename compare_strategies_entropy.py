"""
compare_strategies_entropy.py
------------------------------
Computes H/H_max and z_entropy for every strategy and scatter plots them.

Usage
-----
    python compare_strategies_entropy.py [options]

    -g, --grid_size   Grid side length (default: 4)
    -l, --seq_length  Sequence length  (default: 6)
    -n, --num_seq     Sequences per strategy (default: 500)
    -s, --save        Output path (default: entropy_scatter.png)
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

sys.path.insert(0, str(Path(__file__).parent))

from calculate_entropy import calculate_entropy

# ── All strategy names ────────────────────────────────────────────────────────
STRATEGIES = [
    "random",
    "rowwise",
    "rowwise_sequential",
    "center_out_radial",
    "center_out_spiral",
    "neighbor_first",
    "nearest_first",
    "farthest_first",
    "checkerboard",
    "knights_move",
    "snake",
    "perimeter_crawl",
    "islands",
    "random_walk",
    "diagonal_sweep",
    "weighted_center_bias",
    "hilbert_curve",
]

DISPLAY_NAMES = {
    "random"               : "Random",
    "rowwise"              : "Row-wise",
    "rowwise_sequential"   : "Row-wise\nSequential",
    "center_out_radial"    : "Center-Out\nRadial",
    "center_out_spiral"    : "Center-Out\nSpiral",
    "neighbor_first"       : "Neighbor\nFirst",
    "nearest_first"        : "Nearest\nFirst",
    "farthest_first"       : "Farthest\nFirst",
    "checkerboard"         : "Checkerboard",
    "knights_move"         : "Knight's\nMove",
    "snake"                : "Snake",
    "perimeter_crawl"      : "Perimeter\nCrawl",
    "islands"              : "Islands",
    "random_walk"          : "Random\nWalk",
    "diagonal_sweep"       : "Diagonal\nSweep",
    "weighted_center_bias" : "Weighted\nCenter Bias",
    "hilbert_curve"        : "Hilbert\nCurve",
}

# Colour by broad category
CATEGORY_COLORS = {
    "random"               : "#e8e8e8",
    "rowwise"              : "#4fc3f7",
    "rowwise_sequential"   : "#0288d1",
    "center_out_radial"    : "#ff8a65",
    "center_out_spiral"    : "#f4511e",
    "neighbor_first"       : "#81c784",
    "nearest_first"        : "#388e3c",
    "farthest_first"       : "#1b5e20",
    "checkerboard"         : "#ce93d8",
    "knights_move"         : "#7b1fa2",
    "snake"                : "#4dd0e1",
    "perimeter_crawl"      : "#00838f",
    "islands"              : "#ffb74d",
    "random_walk"          : "#a5d6a7",
    "diagonal_sweep"       : "#fff176",
    "weighted_center_bias" : "#f48fb1",
    "hilbert_curve"        : "#ef5350",
}


def run(
    grid_size: int  = 4,
    seq_length: int = 6,
    num_seq: int    = 1000,
    save_path: str  = "plots/spatial_entropy_comparison.png",
) -> None:

    print(f"\nComputing entropy for {len(STRATEGIES)} strategies "
          f"[grid={grid_size}×{grid_size}, seq_length={seq_length}, "
          f"num_seq={num_seq}]\n")

    h_norm_vals, z_vals, names, colors = [], [], [], []

    for strat in STRATEGIES:
        result = calculate_entropy(
            grid_size=grid_size,
            seq_length=seq_length,
            num_seq=num_seq,
            strategy_name=strat,
            verbose=False
        )
        h_norm_vals.append(result["h_normalized"])
        z_vals.append(result["z_entropy"])
        names.append(strat)
        colors.append(CATEGORY_COLORS[strat])
        print(f"  {strat:<26}  H/H_max={result['h_normalized']:.4f}  "
              f"z={result['z_entropy']:+.2f}")

    h_norm_vals = np.array(h_norm_vals)
    z_vals      = np.array(z_vals)

    # ── Figure ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 8), facecolor="#0d0d1a")
    ax.set_facecolor("#0d0d1a")

    # Reference lines
    ax.axhline(0, color="#555577", linewidth=0.8, linestyle="--", zorder=1)
    ax.axvline(h_norm_vals.mean(), color="#555577", linewidth=0.8,
               linestyle=":", zorder=1, label=f"mean H/H_max = {h_norm_vals.mean():.3f}")

    # Scatter points
    scatter = ax.scatter(
        h_norm_vals, z_vals,
        c=colors,
        s=160,
        edgecolors="white",
        linewidths=0.6,
        zorder=3,
        alpha=0.93,
    )

    # Labels — offset to avoid overlap with the dot
    for i, strat in enumerate(names):
        x, y = h_norm_vals[i], z_vals[i]

        # Nudge labels away from the point
        x_off = 0.007
        y_off = (z_vals.max() - z_vals.min()) * 0.018
        ha = "left"

        # Push left for rightmost points
        if x > np.percentile(h_norm_vals, 80):
            x_off = -0.007
            ha = "right"

        label = DISPLAY_NAMES[strat]
        ax.text(
            x + x_off, y + y_off,
            label,
            fontsize=7,
            ha=ha, va="bottom",
            color="white",
            linespacing=1.2,
            path_effects=[
                pe.withStroke(linewidth=2.5, foreground="#0d0d1a")
            ],
            zorder=4,
        )

    # Axis formatting
    ax.set_xlabel("H / H_max  (normalised transition entropy)",
                  color="white", fontsize=11, labelpad=8)
    ax.set_ylabel("z_entropy  (standard deviations below shuffle null)",
                  color="white", fontsize=11, labelpad=8)
    ax.tick_params(colors="white", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#555577")

    ax.set_xlim(h_norm_vals.min() - 0.06, h_norm_vals.max() + 0.09)
    y_pad = (z_vals.max() - z_vals.min()) * 0.12
    ax.set_ylim(z_vals.min() - y_pad, z_vals.max() + y_pad)

    ax.grid(color="#1e1e3a", linewidth=0.5, zorder=0)
    ax.legend(fontsize=8, facecolor="#1a1a2e", edgecolor="#555577",
              labelcolor="white", loc="upper left")

    # Quadrant annotations
    x_mid = h_norm_vals.mean()
    y_mid = 0
    y_top = z_vals.max() + y_pad * 0.5
    y_bot = z_vals.min() - y_pad * 0.5

    ax.text(x_mid + 0.01, y_top, "high H, low structure vs null →",
            color="#888899", fontsize=7, ha="left", va="top", style="italic")
    ax.text(x_mid - 0.01, y_top, "← low H, low structure vs null",
            color="#888899", fontsize=7, ha="right", va="top", style="italic")
    ax.text(x_mid + 0.01, y_bot, "high H, high structure vs null →",
            color="#888899", fontsize=7, ha="left", va="bottom", style="italic")
    ax.text(x_mid - 0.01, y_bot, "← low H, high structure vs null",
            color="#888899", fontsize=7, ha="right", va="bottom", style="italic")

    fig.suptitle(
        f"Transition Entropy: H/H_max  vs  z_entropy\n"
        f"{grid_size}×{grid_size} grid · seq_length={seq_length} · "
        f"{num_seq} sequences per strategy · ",
        color="white", fontsize=12, y=0.98,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\nSaved → {out.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid_size",  "-g", type=int, default=4)
    parser.add_argument("--seq_length", "-l", type=int, default=6)
    parser.add_argument("--num_seq",    "-n", type=int, default=500)
    parser.add_argument("--save",       "-s", type=str,
                        default="plots/spatial_entropy_comparison.png")
    args = parser.parse_args()

    run(grid_size=args.grid_size, seq_length=args.seq_length,
        num_seq=args.num_seq, save_path=args.save)
