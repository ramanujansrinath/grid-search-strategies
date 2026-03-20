"""
draw_sample.py
--------------
Visualisation utility for all selection strategies.

Usage
-----
    from draw_sample import drawSample

    # Non-image strategy:
    drawSample(grid_size=4, seq_length=6, num_seq=100,
               strategy_name="neighbor_first",
               save_path="output.png", seed=42)

    # Image-based strategy (img_index required):
    drawSample(grid_size=4, seq_length=6, num_seq=100,
               strategy_name="image_salience",
               img_index=1,
               save_path="output.png", seed=42)

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

# ── Make sure the strategies directory is importable ─────────────────────────
_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

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
) -> list:
    """
    Call mod.generate_sequences, forwarding img_index only when the strategy
    signature requires it.  Raises ValueError if img_index is needed but None.
    """
    if _requires_image(mod):
        if img_index is None:
            raise ValueError(
                f"Strategy '{strategy_name}' is image-based and requires "
                f"img_index to be provided.  Pass img_index=<n> where n is "
                f"the 1-based index of the image in the images/ folder "
                f"(e.g. img_index=1 loads images/img_1.png)."
            )
        return mod.generate_sequences(
            grid_size=grid_size, seq_length=seq_length,
            n_seq=n_seq, img_index=img_index,
        )
    return mod.generate_sequences(
        grid_size=grid_size, seq_length=seq_length, n_seq=n_seq,
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


def drawSample(
    grid_size: int = 4,
    seq_length: int = 6,
    num_seq: int = 100,
    strategy_name: str = "random",
    img_index: Optional[int] = None,
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
    strategy_name : name of the strategy module to use.
    img_index     : 1-based image index for image-based strategies
                    (e.g. 1 → images/img_1.png).  Required for strategies
                    whose generate_sequences has an img_index parameter;
                    raises ValueError if omitted for those strategies.
    save_path     : if given, save the figure to this path; else plt.show().
    seed          : optional RNG seed for reproducibility.
    """
    if seed is not None:
        random.seed(seed)

    mod       = _load_strategy(strategy_name)
    sequences = _call_generate_sequences(
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

    display_name = strategy_name.replace("strategy_", "").replace("_", " ").title()
    img_label    = f" · img {img_index}" if img_index is not None else ""
    fig.suptitle(
        f"Strategy: {display_name}{img_label}   "
        f"[{grid_size}×{grid_size} grid · seq length {seq_length} · {num_seq} sequences]",
        fontsize=SUPTITLE_SIZE, color="white", y=0.995,
    )

    plt.tight_layout(pad=SUBPLOT_PAD)

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
    parser.add_argument("strategy",              type=str, help="Strategy name")
    parser.add_argument("--grid_size",  "-g",    type=int, default=4)
    parser.add_argument("--seq_length", "-l",    type=int, default=6)
    parser.add_argument("--num_seq",    "-n",    type=int, default=100)
    parser.add_argument("--img_index",  "-i",    type=int, default=None,
                        help="Image index for image-based strategies "
                             "(e.g. 1 → images/img_1.png)")
    parser.add_argument("--save",       "-s",    type=str, default=None,
                        help="Output file path (e.g. out.png)")
    parser.add_argument("--seed",                type=int, default=None)
    args = parser.parse_args()

    drawSample(
        grid_size=args.grid_size,
        seq_length=args.seq_length,
        num_seq=args.num_seq,
        strategy_name=args.strategy,
        img_index=args.img_index,
        save_path=args.save,
        seed=args.seed,
    )
