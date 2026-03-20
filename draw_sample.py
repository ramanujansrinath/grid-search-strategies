"""
draw_sample.py
--------------
Visualisation utility for all selection strategies.

Usage
-----
    from draw_sample import drawSample

    # Non-image strategy  →  saves to  plots/neighbor_first.png
    drawSample(grid_size=4, seq_length=6, num_seq=100,
               strategy_name="neighbor_first", seed=42)

    # Image-based strategy  →  saves to  plots/image_salience_img1.png
    #                          and        plots/image_salience_img1_diagnostics.png
    drawSample(grid_size=4, seq_length=6, num_seq=100,
               strategy_name="image_salience",
               img_index=1, seed=42)

Image-based strategies are detected automatically via inspect.signature:
if a strategy's generate_sequences function has an img_index parameter,
it is treated as image-based.  Calling drawSample without img_index for
such a strategy raises a ValueError with a descriptive message.
"""

from __future__ import annotations

import importlib
import inspect
import math
import random
import sys
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import numpy as np

# ── Make sure strategies/ and utils/ are importable ──────────────────────────
_HERE = Path(__file__).parent
for _d in (_HERE / "strategies", _HERE / "utils"):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

# ── Tunable visual constants ──────────────────────────────────────────────────
CMAP_NAME       = "plasma"      # matplotlib colormap: purple → yellow
UNVISITED_COLOR = "#1a1a2e"     # dark navy for unvisited cells
VISITED_ALPHA   = 0.92          # opacity of colored cells
LABEL_FONTSIZE  = 5.5           # font size for order labels inside cells
TITLE_FONTSIZE  = 5.0           # font size for each subplot title
SUPTITLE_SIZE   = 11            # font size for the overall figure title
CELL_LINEWIDTH  = 0.4           # border linewidth for each cell rectangle
SUBPLOT_PAD     = 0.35          # padding between subplots (fraction of axis size)
FIG_DPI         = 150           # dots-per-inch for saved / displayed figure
# ─────────────────────────────────────────────────────────────────────────────


def _load_strategy(strategy_name: str):
    """Import and return the strategy module by name."""
    name = strategy_name.lower().strip()
    if not name.startswith("strategy_"):
        name = "strategy_" + name
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        raise ValueError(f"Cannot find strategy module '{name}'.")


def _requires_image(mod) -> bool:
    """Return True if the strategy's generate_sequences expects img_index."""
    return "img_index" in inspect.signature(mod.generate_sequences).parameters


def _call_generate_sequences(
    mod,
    strategy_name: str,
    grid_size: int,
    seq_length: int,
    n_seq: int,
    img_index: Optional[int],
) -> tuple:
    """
    Call mod.generate_sequences.  Always returns (sequences, diagnostics_or_None).

    For image-based strategies, requests diagnostics and returns
    (sequences, diagnostics_dict).  For spatial strategies returns
    (sequences, None).  Raises ValueError if img_index is needed but None.
    """
    if _requires_image(mod):
        if img_index is None:
            raise ValueError(
                f"Strategy '{strategy_name}' is image-based and requires "
                f"img_index to be provided.  Pass img_index=<n> where n is "
                f"the 1-based index of the image in the images/ folder "
                f"(e.g. img_index=1 loads images/img_1.png)."
            )
        sequences, diagnostics = mod.generate_sequences(
            grid_size=grid_size, seq_length=seq_length,
            n_seq=n_seq, img_index=img_index,
            return_diagnostics=True,
        )
        return sequences, diagnostics
    sequences = mod.generate_sequences(
        grid_size=grid_size, seq_length=seq_length, n_seq=n_seq,
    )
    return sequences, None


def _subplot_grid(num_seq: int):
    """Return (n_rows, n_cols) for a near-square subplot layout."""
    n_cols = math.ceil(math.sqrt(num_seq))
    n_rows = math.ceil(num_seq / n_cols)
    return n_rows, n_cols


def _draw_sequence(
    ax: plt.Axes,
    seq: List[int],
    grid_size: int,
    seq_idx: int,
    cmap,
    norm,
) -> None:
    """Draw one sequence into a single Axes."""
    N = grid_size
    ax.set_xlim(0, N)
    ax.set_ylim(0, N)
    ax.set_aspect("equal")
    ax.axis("off")

    visit_order = {box: rank for rank, box in enumerate(seq, start=1)}

    for box in range(1, N * N + 1):
        row = (box - 1) // N
        col = (box - 1) % N
        x   = col
        y   = (N - 1) - row

        if box in visit_order:
            rank      = visit_order[box]
            facecolor = cmap(norm(rank))
            ax.add_patch(patches.FancyBboxPatch(
                (x + 0.04, y + 0.04), 0.92, 0.92,
                boxstyle="round,pad=0.03",
                linewidth=CELL_LINEWIDTH,
                edgecolor="white",
                facecolor=(*facecolor[:3], VISITED_ALPHA),
            ))
            ax.text(
                x + 0.5, y + 0.5, str(rank),
                ha="center", va="center",
                fontsize=LABEL_FONTSIZE, fontweight="bold",
                color="white" if facecolor[0] < 0.7 else "#111111",
            )
        else:
            ax.add_patch(patches.FancyBboxPatch(
                (x + 0.04, y + 0.04), 0.92, 0.92,
                boxstyle="round,pad=0.03",
                linewidth=CELL_LINEWIDTH,
                edgecolor="#444466",
                facecolor=UNVISITED_COLOR,
            ))

    ax.set_title(f"#{seq_idx + 1}", fontsize=TITLE_FONTSIZE,
                 pad=1.5, color="#cccccc")


def _draw_diagnostics(
    diag: dict,
    grid_size: int,
    strategy_name: str,
    img_index: int,
    save_path: str,
) -> None:
    """
    Draw a 3-panel diagnostic figure for an image-based strategy:
      Panel 1 — resized input image (RGB)
      Panel 2 — full P×P pixel-level metric map, with N×N grid lines overlaid
      Panel 3 — N×N noise-free weight grid with per-cell value labels

    Always saved to save_path (caller is responsible for the full path).
    """
    N            = grid_size
    img_rgb      = diag["img_rgb"]
    metric       = diag["metric"]
    weights_flat = diag["no_noise_weights"]
    weights_grid = weights_flat.reshape(N, N)

    BG = "#0d0d1a"
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), facecolor=BG)
    for ax in axes:
        ax.set_facecolor(BG)

    display_name = strategy_name.replace("strategy_", "").replace("_", " ").title()

    # ── Panel 1: Input image ──────────────────────────────────────────────
    axes[0].imshow(img_rgb)
    axes[0].set_title(f"Input image  (img_{img_index})",
                      color="white", fontsize=9, pad=6)
    axes[0].axis("off")

    # ── Panel 2: P×P metric map with N×N grid overlay ────────────────────
    ax = axes[1]
    P        = metric.shape[0]
    has_neg  = metric.min() < 0
    cmap_met = "RdBu_r" if has_neg else "inferno"
    vabs     = float(np.abs(metric).max())
    vmin_met = -vabs if has_neg else float(metric.min())

    im1 = ax.imshow(metric, cmap=cmap_met, vmin=vmin_met, vmax=vabs,
                    interpolation="nearest")
    # Overlay N×N grid lines
    cell_px = P / N
    for i in range(1, N):
        ax.axhline(i * cell_px - 0.5, color="white", linewidth=0.7, alpha=0.55)
        ax.axvline(i * cell_px - 0.5, color="white", linewidth=0.7, alpha=0.55)
    cb1 = fig.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
    cb1.ax.yaxis.set_tick_params(color="white", labelsize=7)
    plt.setp(cb1.ax.yaxis.get_ticklabels(), color="white")
    cb1.outline.set_edgecolor("#555577")
    ax.set_title(f"Pixel metric  ({P}×{P})", color="white", fontsize=9, pad=6)
    ax.axis("off")

    # ── Panel 3: N×N noise-free weight grid ──────────────────────────────
    ax = axes[2]
    im2 = ax.imshow(weights_grid, cmap="viridis", vmin=0.1, vmax=0.9,
                    interpolation="nearest",
                    extent=[-0.5, N - 0.5, N - 0.5, -0.5])
    # Grid lines
    for i in range(N + 1):
        ax.axhline(i - 0.5, color="white", linewidth=0.5, alpha=0.4)
        ax.axvline(i - 0.5, color="white", linewidth=0.5, alpha=0.4)
    # Per-cell labels: box number (small) + weight value (large)
    for row in range(N):
        for col in range(N):
            w   = weights_grid[row, col]
            box = row * N + col + 1
            txt_color = "white" if w < 0.6 else "#111111"
            ax.text(col, row - 0.12, str(box),
                    ha="center", va="center", fontsize=6,
                    color=txt_color, alpha=0.75)
            ax.text(col, row + 0.22, f"{w:.2f}",
                    ha="center", va="center", fontsize=8,
                    color=txt_color, fontweight="bold")
    cb2 = fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
    cb2.ax.yaxis.set_tick_params(color="white", labelsize=7)
    plt.setp(cb2.ax.yaxis.get_ticklabels(), color="white")
    cb2.outline.set_edgecolor("#555577")
    ax.set_title(f"Noise-free weights  ({N}×{N})",
                 color="white", fontsize=9, pad=6)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#555577")

    fig.suptitle(
        f"Diagnostics: {display_name}  ·  img {img_index}  ·  {N}×{N} grid",
        color="white", fontsize=11, y=1.01,
    )
    plt.tight_layout()

    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Diagnostics saved → {out.resolve()}")

    plt.close(fig)


def drawSample(
    grid_size: int = 4,
    seq_length: int = 6,
    num_seq: int = 100,
    strategy_name: str = "random",
    img_index: Optional[int] = None,
    seed: Optional[int] = None,
) -> None:
    """
    Generate *num_seq* sequences using the named strategy, then plot each
    one as a grid in a tiled figure.

    Figures are always auto-saved to the plots/ folder:
      plots/<strategy_name>.png                       (spatial strategies)
      plots/<strategy_name>_img<n>.png                (image strategies, main figure)
      plots/<strategy_name>_img<n>_diagnostics.png    (image strategies, diagnostic figure)

    Parameters
    ----------
    grid_size     : side length of the square grid (e.g. 4 for a 4×4 grid).
    seq_length    : number of boxes selected per sequence.
    num_seq       : total number of sequences (= number of subplot panels).
    strategy_name : name of the strategy module to use.
    img_index     : 1-based image index for image-based strategies
                    (e.g. 1 → images/img_1.png).  Required for strategies
                    whose generate_sequences has an img_index parameter;
                    raises ValueError if omitted for those strategies.
    seed          : optional RNG seed for reproducibility.
    """
    if seed is not None:
        random.seed(seed)

    mod                    = _load_strategy(strategy_name)
    sequences, diagnostics = _call_generate_sequences(
        mod, strategy_name, grid_size, seq_length, num_seq, img_index
    )

    cmap = plt.get_cmap(CMAP_NAME)
    norm = mcolors.Normalize(vmin=1, vmax=seq_length)

    n_rows, n_cols = _subplot_grid(num_seq)
    cell_size = 0.55 + 0.12 * grid_size
    fig_w = n_cols * cell_size + 1.2
    fig_h = n_rows * cell_size + 0.7

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(fig_w, fig_h), facecolor="#0d0d1a",
    )
    axes_flat = np.array(axes).flatten()

    for i, ax in enumerate(axes_flat):
        if i < num_seq:
            _draw_sequence(ax, sequences[i], grid_size, i, cmap, norm)
            ax.set_facecolor("#0d0d1a")
        else:
            ax.axis("off")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes_flat, shrink=0.6, aspect=30,
                        pad=0.01, fraction=0.015)
    cbar.set_label("Visit order", color="white", fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="white", labelsize=7)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
    cbar.outline.set_edgecolor("#555577")

    # ── Auto-save paths ───────────────────────────────────────────────────
    stem      = strategy_name.replace("strategy_", "")
    img_suffix = f"_img{img_index}" if img_index is not None else ""
    plots_dir = _HERE / "plots"
    main_path = plots_dir / f"{stem}{img_suffix}.png"
    diag_path = plots_dir / f"{stem}{img_suffix}_diagnostics.png"

    display_name = stem.replace("_", " ").title()
    img_label    = f" · img {img_index}" if img_index is not None else ""
    fig.suptitle(
        f"Strategy: {display_name}{img_label}   "
        f"[{grid_size}×{grid_size} grid · seq length {seq_length} · {num_seq} sequences]",
        fontsize=SUPTITLE_SIZE, color="white", y=0.995,
    )

    plt.tight_layout(pad=SUBPLOT_PAD)

    main_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(main_path, dpi=FIG_DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Saved → {main_path.resolve()}")

    plt.close(fig)

    # ── Diagnostic figure for image-based strategies ──────────────────────
    if diagnostics is not None:
        _draw_diagnostics(diagnostics, grid_size, strategy_name,
                          img_index, str(diag_path))


# ── CLI convenience ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualise selection strategies.")
    parser.add_argument("strategy",              type=str, help="Strategy name")
    parser.add_argument("--grid_size",  "-g",    type=int, default=4)
    parser.add_argument("--seq_length", "-l",    type=int, default=6)
    parser.add_argument("--num_seq",    "-n",    type=int, default=100)
    parser.add_argument("--img_index",  "-i",    type=int, default=None,
                        help="Image index for image-based strategies "
                             "(e.g. 1 → images/img_1.png)")
    parser.add_argument("--seed",                type=int, default=None)
    args = parser.parse_args()

    drawSample(
        grid_size=args.grid_size,
        seq_length=args.seq_length,
        num_seq=args.num_seq,
        strategy_name=args.strategy,
        img_index=args.img_index,
        seed=args.seed,
    )
