"""
calculate_entropy.py
--------------------
Computes the global transition entropy of a selection strategy.

Method
------
1. Generate *num_seq* sequences using the named strategy.
2. For each sequence, build an N²×N² transition count matrix T where
   T[i, j] += 1 each time box j is selected *immediately after* box i.
   (Boxes are 0-indexed internally: box k → index k-1.)
3. Sum transition matrices across all sequences to get T_total.
4. Normalise T_total into a joint probability distribution:
       P(i → j) = T_total[i, j] / Σ_{i,j} T_total[i, j]
5. Compute Shannon entropy (nats, then converted to bits):
       H = -Σ_{i,j} P(i→j) · log2( P(i→j) )   [zero terms skipped]

The maximum possible entropy is log2(N² · (N²-1)) bits — the entropy of a
uniform distribution over all ordered pairs of distinct boxes.  The function
reports both the raw entropy and the normalised entropy (H / H_max).

Returns
-------
A dict with keys:
    "entropy_bits"   : float  — Shannon entropy in bits
    "entropy_nats"   : float  — Shannon entropy in nats
    "h_max_bits"     : float  — theoretical maximum entropy (bits)
    "h_normalized"   : float  — H / H_max  (0 = fully deterministic, 1 = uniform)
    "n_transitions"  : int    — total number of transitions observed
    "transition_matrix" : np.ndarray  shape (N², N²), averaged & normalised
    "strategy_name"  : str
    "grid_size"      : int
    "seq_length"     : int
    "num_seq"        : int
"""

from __future__ import annotations

import math
import sys
import importlib
import random
from pathlib import Path
from typing import Optional

import numpy as np

# ── Make sure the strategies directory is importable ─────────────────────────
_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


def _load_strategy(strategy_name: str):
    name = strategy_name.lower().strip()
    if not name.startswith("strategy_"):
        name = "strategy_" + name
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        raise ValueError(f"Cannot find strategy module '{name}'.")


def calculate_entropy(
    grid_size: int = 4,
    seq_length: int = 6,
    num_seq: int = 100,
    strategy_name: str = "random",
    seed: Optional[int] = None,
    verbose: bool = True,
) -> dict:
    """
    Parameters
    ----------
    grid_size     : side length of the square grid.
    seq_length    : number of boxes selected per sequence.
    num_seq       : number of sequences to generate.
    strategy_name : name of the strategy (same names as drawSample).
    seed          : optional RNG seed for reproducibility.
    verbose       : if True, print a formatted summary to stdout.

    Returns
    -------
    Dictionary with entropy metrics and the joint transition matrix.
    """
    if seed is not None:
        random.seed(seed)

    N  = grid_size
    nb = N * N                        # total number of boxes

    # ── Generate sequences ────────────────────────────────────────────────
    mod = _load_strategy(strategy_name)
    sequences = mod.generate_sequences(
        grid_size=grid_size,
        seq_length=seq_length,
        n_seq=num_seq,
    )

    # ── Build cumulative transition count matrix ──────────────────────────
    # Shape: (nb, nb)  — rows = "from" box, cols = "to" box (0-indexed)
    T = np.zeros((nb, nb), dtype=np.float64)

    for seq in sequences:
        for a, b in zip(seq[:-1], seq[1:]):
            T[a - 1, b - 1] += 1      # convert 1-indexed boxes to 0-indexed

    n_transitions = int(T.sum())

    # ── Normalise to joint probability distribution ───────────────────────
    if n_transitions == 0:
        raise ValueError("No transitions observed (seq_length must be ≥ 2).")

    P = T / n_transitions              # joint P(i → j)

    # ── Shannon entropy (bits) ────────────────────────────────────────────
    # Only sum over non-zero cells (0 * log(0) = 0 by convention)
    nonzero = P[P > 0]
    entropy_bits = float(-np.sum(nonzero * np.log2(nonzero)))
    entropy_nats = float(-np.sum(nonzero * np.log(nonzero)))

    # ── Theoretical maximum ───────────────────────────────────────────────
    # Uniform over all nb*(nb-1) ordered pairs of distinct boxes
    h_max_bits = math.log2(nb * (nb - 1))

    h_normalized = entropy_bits / h_max_bits

    # ── Normalised joint matrix (for return) ─────────────────────────────
    result = {
        "entropy_bits"      : entropy_bits,
        "entropy_nats"      : entropy_nats,
        "h_max_bits"        : h_max_bits,
        "h_normalized"      : h_normalized,
        "n_transitions"     : n_transitions,
        "transition_matrix" : P,          # shape (N², N²), joint P(i→j)
        "strategy_name"     : strategy_name,
        "grid_size"         : grid_size,
        "seq_length"        : seq_length,
        "num_seq"           : num_seq,
    }

    # ── Verbose output ────────────────────────────────────────────────────
    if verbose:
        display = strategy_name.replace("strategy_", "").replace("_", " ").title()
        bar_len  = 30
        bar_fill = round(h_normalized * bar_len)
        bar      = "█" * bar_fill + "░" * (bar_len - bar_fill)

        print(f"\n{'─'*54}")
        print(f"  Strategy      : {display}")
        print(f"  Grid          : {grid_size}×{grid_size}   "
              f"seq_length={seq_length}   num_seq={num_seq}")
        print(f"{'─'*54}")
        print(f"  Transitions   : {n_transitions:,}")
        print(f"  H (bits)      : {entropy_bits:.4f}")
        print(f"  H (nats)      : {entropy_nats:.4f}")
        print(f"  H_max (bits)  : {h_max_bits:.4f}  "
              f"[log2({nb}×{nb-1}) = log2({nb*(nb-1)})]")
        print(f"  H / H_max     : {h_normalized:.4f}   [{bar}]")
        print(f"{'─'*54}\n")

    return result


# ── CLI convenience ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Calculate global transition entropy for a selection strategy."
    )
    parser.add_argument("strategy",               type=str)
    parser.add_argument("--grid_size",  "-g",     type=int, default=4)
    parser.add_argument("--seq_length", "-l",     type=int, default=6)
    parser.add_argument("--num_seq",    "-n",     type=int, default=10000)
    parser.add_argument("--seed",                 type=int, default=None)
    parser.add_argument("--quiet",      "-q",     action="store_true",
                        help="Suppress printed summary")
    args = parser.parse_args()

    calculate_entropy(
        grid_size=args.grid_size,
        seq_length=args.seq_length,
        num_seq=args.num_seq,
        strategy_name=args.strategy,
        seed=args.seed,
        verbose=not args.quiet,
    )
