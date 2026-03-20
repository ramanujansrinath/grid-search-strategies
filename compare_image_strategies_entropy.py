"""
compare_image_strategies_entropy.py
------------------------------------
Computes H/H_max and z_entropy for all image-based strategies across every
image found in the images/ folder, plus Random and Hilbert Curve as
non-image baselines.  Scatter-plots H/H_max (x) vs z_entropy (y).

Points are colour-coded by image.  The four image-based strategies are
distinguished by marker shape.  The two non-image baselines (Random,
Hilbert Curve) are plotted in white/grey with distinct markers and labelled
directly on the plot.

Usage
-----
    python compare_image_strategies_entropy.py [options]

    -g, --grid_size   Grid side length (default: 4)
    -l, --seq_length  Sequence length  (default: 6)
    -n, --num_seq     Sequences per strategy/image (default: 500)
    -s, --save        Output path (default: entropy_scatter_image.png)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.lines as mlines

sys.path.insert(0, str(Path(__file__).parent))

from calculate_entropy import calculate_entropy

# ── Strategy definitions ──────────────────────────────────────────────────────

IMAGE_STRATEGIES = [
    "image_salience",
    "image_contrast",
    "image_color_concentration",
    "image_texture",
]

BASELINE_STRATEGIES = [
    "random",
]

# Hilbert Curve is computed for reference but shown in the title, not plotted
REFERENCE_STRATEGY = "hilbert_curve"

DISPLAY_NAMES = {
    "image_salience"            : "Salience",
    "image_contrast"            : "Contrast",
    "image_color_concentration" : "Color Concentration",
    "image_texture"             : "Texture",
    "random"                    : "Random",
    "hilbert_curve"             : "Hilbert Curve",
}

# Marker shapes per image strategy (baselines get their own fixed markers)
STRATEGY_MARKERS = {
    "image_salience"            : "o",   # circle
    "image_contrast"            : "s",   # square
    "image_color_concentration" : "^",   # triangle up
    "image_texture"             : "D",   # diamond
}

BASELINE_MARKERS = {
    "random" : "*",
}

BASELINE_COLORS = {
    "random" : "#e8e8e8",
}

# Distinct colours for up to 12 images; cycles if more are present
IMAGE_PALETTE = [
    "#4fc3f7", "#81c784", "#ffb74d", "#f48fb1",
    "#ce93d8", "#fff176", "#4dd0e1", "#ff8a65",
    "#a5d6a7", "#80cbc4", "#ffe082", "#ef9a9a",
]


def _find_images(images_dir: Path) -> list[Path]:
    """Return sorted list of img_*.jpg files in images_dir."""
    imgs = sorted(images_dir.glob("img_*.jpg"),
                  key=lambda p: int(p.stem.split("_")[1]))
    if not imgs:
        raise FileNotFoundError(
            f"No images found in {images_dir}.  "
            f"Expected files named img_1.jpg, img_2.jpg, … "
            f"in the images/ subfolder."
        )
    return imgs


def run(
    grid_size: int  = 4,
    seq_length: int = 6,
    num_seq: int    = 500,
    save_path: str  = "plots/image_entropy_comparison.png",
) -> None:

    images_dir = Path(__file__).parent / "images"
    image_paths = _find_images(images_dir)
    n_images    = len(image_paths)

    print(f"\nFound {n_images} image(s) in {images_dir}")
    print(f"Grid={grid_size}×{grid_size}  seq_length={seq_length}  "
          f"num_seq={num_seq}\n")

    # ── Collect results ───────────────────────────────────────────────────
    # Each entry: (strategy, img_index or None, h_norm, z, display_label)
    records = []

    # Hilbert Curve: computed for reference only, not plotted
    print(f"  [hilbert_curve] computing (reference only) …", end=" ", flush=True)
    hilbert_result = calculate_entropy(
        grid_size=grid_size, seq_length=seq_length,
        num_seq=num_seq, strategy_name=REFERENCE_STRATEGY, verbose=False,
    )
    hilbert_h = hilbert_result["h_normalized"]
    hilbert_z = hilbert_result["z_entropy"]
    print(f"H/H_max={hilbert_h:.4f}  z={hilbert_z:+.2f}  (will appear in title)")
    print()

    # Non-image baselines plotted (random only)
    for strat in BASELINE_STRATEGIES:
        print(f"  [{strat}] computing …", end=" ", flush=True)
        result = calculate_entropy(
            grid_size=grid_size, seq_length=seq_length,
            num_seq=num_seq, strategy_name=strat, verbose=False,
        )
        records.append({
            "strategy"  : strat,
            "img_index" : None,
            "h_norm"    : result["h_normalized"],
            "z"         : result["z_entropy"],
        })
        print(f"H/H_max={result['h_normalized']:.4f}  "
              f"z={result['z_entropy']:+.2f}")

    print()

    # Image-based strategies × images
    for img_path in image_paths:
        img_index = int(img_path.stem.split("_")[1])
        print(f"  Image {img_index} ({img_path.name})")
        for strat in IMAGE_STRATEGIES:
            print(f"    [{strat}] …", end=" ", flush=True)
            result = calculate_entropy(
                grid_size=grid_size, seq_length=seq_length,
                num_seq=num_seq, strategy_name=strat,
                img_index=img_index, verbose=False,
            )
            records.append({
                "strategy"  : strat,
                "img_index" : img_index,
                "h_norm"    : result["h_normalized"],
                "z"         : result["z_entropy"],
            })
            print(f"H/H_max={result['h_normalized']:.4f}  "
                  f"z={result['z_entropy']:+.2f}")
        print()

    # ── Build arrays ──────────────────────────────────────────────────────
    h_all = np.array([r["h_norm"] for r in records])
    z_all = np.array([r["z"]      for r in records])

    # ── Figure ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 8), facecolor="#0d0d1a")
    ax.set_facecolor("#0d0d1a")

    # Reference lines
    ax.axhline(0, color="#555577", linewidth=0.8, linestyle="--", zorder=1)
    ax.axvline(h_all.mean(), color="#555577", linewidth=0.8, linestyle=":",
               zorder=1)

    # ── Plot image-based strategies ───────────────────────────────────────
    for r in records:
        if r["img_index"] is None:
            continue
        idx   = r["img_index"] - 1          # 0-based for palette
        color = IMAGE_PALETTE[idx % len(IMAGE_PALETTE)]
        marker = STRATEGY_MARKERS[r["strategy"]]
        ax.scatter(
            r["h_norm"], r["z"],
            c=color, marker=marker,
            s=120, edgecolors="white", linewidths=0.5,
            zorder=3, alpha=0.90,
        )

    # ── Plot non-image baselines ──────────────────────────────────────────
    for r in records:
        if r["img_index"] is not None:
            continue
        strat  = r["strategy"]
        color  = BASELINE_COLORS[strat]
        marker = BASELINE_MARKERS[strat]
        ax.scatter(
            r["h_norm"], r["z"],
            c=color, marker=marker,
            s=260, edgecolors="white", linewidths=1.2,
            zorder=5, alpha=1.0,
        )
        # Direct label for baselines
        x_off = -0.007 if r["h_norm"] > h_all.mean() else 0.007
        ha    = "right" if r["h_norm"] > h_all.mean() else "left"
        y_pad_label = (z_all.max() - z_all.min()) * 0.025
        ax.text(
            r["h_norm"] + x_off,
            r["z"] + y_pad_label,
            DISPLAY_NAMES[strat],
            fontsize=9, fontweight="bold",
            ha=ha, va="bottom", color=color,
            path_effects=[pe.withStroke(linewidth=2.5, foreground="#0d0d1a")],
            zorder=6,
        )

    # ── Legend: strategy shapes ───────────────────────────────────────────
    shape_handles = [
        mlines.Line2D([], [], marker=STRATEGY_MARKERS[s], color="w",
                      markerfacecolor="#aaaaaa", markersize=8,
                      linestyle="None", label=DISPLAY_NAMES[s])
        for s in IMAGE_STRATEGIES
    ]
    shape_handles += [
        mlines.Line2D([], [], marker=BASELINE_MARKERS[s], color="w",
                      markerfacecolor=BASELINE_COLORS[s], markersize=10,
                      linestyle="None", label=DISPLAY_NAMES[s])
        for s in BASELINE_STRATEGIES
    ]
    strategy_legend = ax.legend(
        handles=shape_handles,
        title="Strategy", title_fontsize=8,
        fontsize=8, facecolor="#1a1a2e", edgecolor="#555577",
        labelcolor="white", loc="upper left",
    )
    strategy_legend.get_title().set_color("white")
    ax.add_artist(strategy_legend)

    # ── Legend: image colours ─────────────────────────────────────────────
    image_handles = [
        mlines.Line2D([], [], marker="o", color="w",
                      markerfacecolor=IMAGE_PALETTE[i % len(IMAGE_PALETTE)],
                      markersize=8, linestyle="None",
                      label=f"img_{i + 1}.png")
        for i in range(n_images)
    ]
    image_legend = ax.legend(
        handles=image_handles,
        title="Image", title_fontsize=8,
        fontsize=8, facecolor="#1a1a2e", edgecolor="#555577",
        labelcolor="white", loc="upper right",
    )
    image_legend.get_title().set_color("white")
    ax.add_artist(image_legend)

    # ── Axis formatting ───────────────────────────────────────────────────
    ax.set_xlabel("H / H_max  (normalised transition entropy)",
                  color="white", fontsize=11, labelpad=8)
    ax.set_ylabel("z_entropy  (standard deviations from shuffle null)",
                  color="white", fontsize=11, labelpad=8)
    ax.tick_params(colors="white", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#555577")

    x_pad = (h_all.max() - h_all.min()) * 0.10
    y_pad = (z_all.max() - z_all.min()) * 0.14
    ax.set_xlim(h_all.min() - x_pad, h_all.max() + x_pad)
    ax.set_ylim(z_all.min() - y_pad, z_all.max() + y_pad)
    ax.grid(color="#1e1e3a", linewidth=0.5, zorder=0)

    # ── Quadrant labels ───────────────────────────────────────────────────
    x_mid = h_all.mean()
    y_top = z_all.max() + y_pad * 0.6
    y_bot = z_all.min() - y_pad * 0.6
    kw = dict(color="#888899", fontsize=7, va="top", style="italic")
    ax.text(x_mid + 0.002, y_top, "high H, low structure vs null →",
            ha="left",  **kw)
    ax.text(x_mid - 0.002, y_top, "← low H, low structure vs null",
            ha="right", **kw)
    kw["va"] = "bottom"
    ax.text(x_mid + 0.002, y_bot, "high H, high structure vs null →",
            ha="left",  **kw)
    ax.text(x_mid - 0.002, y_bot, "← low H, high structure vs null",
            ha="right", **kw)

    # ── Title ─────────────────────────────────────────────────────────────
    strat_str = ", ".join(DISPLAY_NAMES[s] for s in IMAGE_STRATEGIES)
    hilbert_z_sign = "+" if hilbert_z >= 0 else ""
    fig.suptitle(
        f"Image Strategy Entropy: H/H_max  vs  z_entropy\n"
        f"Strategies: {strat_str}  +  Random baseline\n"
        f"{grid_size}×{grid_size} grid · seq_length={seq_length} · "
        f"{num_seq} sequences · {n_images} image(s)\n"
        f"Reference — Hilbert Curve: H/H_max={hilbert_h:.4f}  "
        f"z={hilbert_z_sign}{hilbert_z:.2f}",
        color="white", fontsize=11, y=0.995,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Saved → {out.resolve()}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid_size",  "-g", type=int, default=4)
    parser.add_argument("--seq_length", "-l", type=int, default=6)
    parser.add_argument("--num_seq",    "-n", type=int, default=500)
    parser.add_argument("--save",       "-s", type=str,
                        default="plots/image_entropy_comparison.png")
    args = parser.parse_args()

    run(grid_size=args.grid_size, seq_length=args.seq_length,
        num_seq=args.num_seq,    save_path=args.save)
