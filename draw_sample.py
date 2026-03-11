"""
draw_sample.py
--------------
Visualisation utility for all selection strategies.

Usage
-----
    from draw_sample import drawSample

    drawSample(
        grid_size     = 4,
        seq_length    = 6,
        num_seq       = 100,
        strategy_name = "neighbor_first",   # bare name, no "strategy_" prefix needed
        save_path     = "output.png",        # optional; if None, plt.show() is called
        seed          = 42,                  # optional RNG seed for reproducibility
    )

Strategy names accepted (case-insensitive, "strategy_" prefix optional)
------------------------------------------------------------------------
    random, rowwise, rowwise_sequential, center_out_radial,
    center_out_spiral, neighbor_first, nearest_first, farthest_first,
    checkerboard, knights_move, snake, perimeter_crawl, islands,
    random_walk, diagonal_sweep, weighted_center_bias, hilbert_curve
"""

from __future__ import annotations

import importlib
import math
import random
import sys
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import numpy as np

# ── Make sure the strategies directory is importable ─────────────────────────
_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# ── Tunable visual constants ──────────────────────────────────────────────────
CMAP_NAME       = "plasma"      # matplotlib colormap: blue-ish → yellow
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
        raise ValueError(
            f"Cannot find strategy module '{name}'. "
            f"Available strategies: random, rowwise, rowwise_sequential, "
            f"center_out_radial, center_out_spiral, neighbor_first, "
            f"nearest_first, farthest_first, checkerboard, knights_move, "
            f"snake, perimeter_crawl, islands, random_walk, diagonal_sweep, "
            f"weighted_center_bias, hilbert_curve"
        )


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

    # ── Draw all cells ────────────────────────────────────────────────────
    visit_order = {}   # box → 1-based order it was visited
    for rank, box in enumerate(seq, start=1):
        visit_order[box] = rank

    for box in range(1, N * N + 1):
        row = (box - 1) // N          # 0-indexed, top = 0
        col = (box - 1) % N

        # Matplotlib y-axis: row 0 at TOP → invert y
        x = col
        y = (N - 1) - row

        if box in visit_order:
            rank = visit_order[box]
            facecolor = cmap(norm(rank))
            rect = patches.FancyBboxPatch(
                (x + 0.04, y + 0.04),
                0.92, 0.92,
                boxstyle="round,pad=0.03",
                linewidth=CELL_LINEWIDTH,
                edgecolor="white",
                facecolor=(*facecolor[:3], VISITED_ALPHA),
            )
            ax.add_patch(rect)

            # Order label
            ax.text(
                x + 0.5, y + 0.5, str(rank),
                ha="center", va="center",
                fontsize=LABEL_FONTSIZE,
                fontweight="bold",
                color="white" if facecolor[0] < 0.7 else "#111111",
            )
        else:
            rect = patches.FancyBboxPatch(
                (x + 0.04, y + 0.04),
                0.92, 0.92,
                boxstyle="round,pad=0.03",
                linewidth=CELL_LINEWIDTH,
                edgecolor="#444466",
                facecolor=UNVISITED_COLOR,
            )
            ax.add_patch(rect)

    # Sequence index (small, top-left)
    ax.set_title(f"#{seq_idx + 1}", fontsize=TITLE_FONTSIZE,
                 pad=1.5, color="#cccccc")


def drawSample(
    grid_size: int = 4,
    seq_length: int = 6,
    num_seq: int = 100,
    strategy_name: str = "random",
    save_path: Optional[str] = None,
    seed: Optional[int] = None,
) -> None:
    """
    Generate *num_seq* sequences using the named strategy, then plot each
    one as a grid in a tiled figure.

    Parameters
    ----------
    grid_size     : side length of the square grid (e.g. 4 for a 4×4 grid).
    seq_length    : number of boxes selected per sequence.
    num_seq       : total number of sequences (= number of subplot panels).
    strategy_name : name of the strategy module to use (see module docstring).
    save_path     : if given, save the figure to this path instead of
                    calling plt.show().  E.g. "output/spiral_100.png".
    seed          : optional integer seed for the Python RNG (reproducibility).
    """
    if seed is not None:
        random.seed(seed)

    # ── Load strategy and generate sequences ─────────────────────────────
    mod = _load_strategy(strategy_name)
    sequences = mod.generate_sequences(
        grid_size=grid_size,
        seq_length=seq_length,
        n_seq=num_seq,
    )

    # ── Colour map and normalisation (rank 1 … seq_length) ───────────────
    cmap = plt.get_cmap(CMAP_NAME)
    norm = mcolors.Normalize(vmin=1, vmax=seq_length)

    # ── Figure layout ─────────────────────────────────────────────────────
    n_rows, n_cols = _subplot_grid(num_seq)

    # Each subplot cell is slightly larger for larger grids
    cell_size = 0.55 + 0.12 * grid_size
    fig_w = n_cols * cell_size + 1.2   # +room for colorbar
    fig_h = n_rows * cell_size + 0.7   # +room for suptitle

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(fig_w, fig_h),
        facecolor="#0d0d1a",
    )

    # Flatten and pad with None if last row is incomplete
    axes_flat = np.array(axes).flatten()

    for i, ax in enumerate(axes_flat):
        if i < num_seq:
            _draw_sequence(ax, sequences[i], grid_size, i, cmap, norm)
            ax.set_facecolor("#0d0d1a")
        else:
            ax.axis("off")          # blank panel for incomplete last row

    # ── Colour bar (visit order) ──────────────────────────────────────────
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(
        sm, ax=axes_flat, shrink=0.6, aspect=30,
        pad=0.01, fraction=0.015,
    )
    cbar.set_label("Visit order", color="white", fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="white", labelsize=7)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
    cbar.outline.set_edgecolor("#555577")

    # ── Title ─────────────────────────────────────────────────────────────
    display_name = strategy_name.replace("strategy_", "").replace("_", " ").title()
    fig.suptitle(
        f"Strategy: {display_name}   "
        f"[{grid_size}×{grid_size} grid · seq length {seq_length} · {num_seq} sequences]",
        fontsize=SUPTITLE_SIZE, color="white", y=0.995,
    )

    plt.tight_layout(pad=SUBPLOT_PAD)

    # ── Output ────────────────────────────────────────────────────────────
    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"Saved → {out.resolve()}")
    else:
        plt.show()

    plt.close(fig)


# ── CLI convenience ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualise selection strategies.")
    parser.add_argument("strategy",          type=str,           help="Strategy name")
    parser.add_argument("--grid_size",  "-g", type=int, default=4)
    parser.add_argument("--seq_length", "-l", type=int, default=6)
    parser.add_argument("--num_seq",    "-n", type=int, default=100)
    parser.add_argument("--save",       "-s", type=str, default=None,
                        help="Output file path (e.g. out.png)")
    parser.add_argument("--seed",             type=int, default=None)
    args = parser.parse_args()

    drawSample(
        grid_size=args.grid_size,
        seq_length=args.seq_length,
        num_seq=args.num_seq,
        strategy_name=args.strategy,
        save_path=args.save,
        seed=args.seed,
    )
