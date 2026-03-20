"""
calculate_entropy.py
--------------------
Computes two entropy measures for a selection strategy.

H  — Global transition entropy
     Shannon entropy of the joint distribution P(i→j) over all observed
     consecutive (from, to) box transitions, pooled across all sequences.
     Transitions that cross sequence boundaries are excluded by construction.

z_entropy — Z-scored transition entropy  (ported from compute_zscored_global_entropy.m)
     The same H, z-scored against a shuffle null distribution built by
     permuting the *from* labels while holding *to* fixed.  This breaks real
     sequential structure while preserving each label's marginal frequency.
     z_entropy = (H − mean(H_shuffle)) / std(H_shuffle).

     Positive z  → transitions more spread out than chance.
     Negative z  → transitions more concentrated / predictable than chance.

Image-based strategies
----------------------
Strategies whose generate_sequences signature includes an img_index parameter
are detected automatically via inspect.signature.  For those strategies,
img_index must be supplied; omitting it raises a ValueError.

Tunable constant
----------------
N_SHUFFLES : int  (default 200) — shuffle iterations for the z_entropy null.

Returns
-------
A dict with keys:
    "entropy_bits"      : float       — H in bits
    "entropy_nats"      : float       — H in nats
    "h_max_bits"        : float       — log2(N² × (N²−1)), theoretical max
    "h_normalized"      : float       — H / H_max ∈ [0, 1]
    "z_entropy"         : float       — z-scored H against shuffle null
    "h_shuffle_mean"    : float       — mean H across shuffles (bits)
    "h_shuffle_std"     : float       — std  H across shuffles (bits)
    "h_shuffle_all"     : np.ndarray  — all N_SHUFFLES shuffle entropies
    "n_transitions"     : int         — total transitions observed
    "transition_matrix" : np.ndarray  — shape (N², N²), joint P(i→j)
    "strategy_name"     : str
    "grid_size"         : int
    "seq_length"        : int
    "num_seq"           : int
"""

from __future__ import annotations

import importlib
import inspect
import math
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# ── Make sure the strategies directory is importable ─────────────────────────
_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# ── Exposed tuning constant ───────────────────────────────────────────────────
N_SHUFFLES: int = 200   # shuffle iterations for z_entropy null distribution
# ─────────────────────────────────────────────────────────────────────────────


def _load_strategy(strategy_name: str):
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


def _transition_entropy(from_ids: np.ndarray, to_ids: np.ndarray) -> float:
    """
    Shannon entropy (bits) of the empirical transition distribution.
    Each (from, to) pair is treated as a single symbol.
    """
    offset = int(from_ids.max()) + int(to_ids.max()) + 2
    keys   = from_ids.astype(np.int64) * offset + to_ids.astype(np.int64)
    _, counts = np.unique(keys, return_counts=True)
    probs  = counts / counts.sum()
    return float(-np.sum(probs * np.log2(probs)))


def calculate_entropy(
    grid_size: int = 4,
    seq_length: int = 6,
    num_seq: int = 100,
    strategy_name: str = "random",
    img_index: Optional[int] = None,
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
    img_index     : 1-based image index for image-based strategies
                    (e.g. 1 → images/img_1.png).  Required for strategies
                    whose generate_sequences has an img_index parameter;
                    raises ValueError if omitted for those strategies.
    seed          : optional RNG seed for reproducibility.
    verbose       : if True, print a formatted summary to stdout.

    Returns
    -------
    Dictionary — see module docstring for full key listing.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    N  = grid_size
    nb = N * N

    # ── Generate sequences ────────────────────────────────────────────────
    mod       = _load_strategy(strategy_name)
    sequences = _call_generate_sequences(
        mod, strategy_name, grid_size, seq_length, num_seq, img_index
    )

    # ── Build flat (from, to) arrays and N²×N² count matrix ──────────────
    from_ids_list, to_ids_list = [], []
    T = np.zeros((nb, nb), dtype=np.float64)

    for seq in sequences:
        for a, b in zip(seq[:-1], seq[1:]):
            from_ids_list.append(a)
            to_ids_list.append(b)
            T[a - 1, b - 1] += 1

    from_ids = np.array(from_ids_list, dtype=np.int64)
    to_ids   = np.array(to_ids_list,   dtype=np.int64)
    n_transitions = len(from_ids)

    if n_transitions == 0:
        raise ValueError("No transitions observed (seq_length must be ≥ 2).")

    # ── H: observed transition entropy ───────────────────────────────────
    entropy_bits = _transition_entropy(from_ids, to_ids)
    P            = T / n_transitions

    nonzero      = P[P > 0]
    entropy_nats = float(-np.sum(nonzero * np.log(nonzero)))
    h_max_bits   = math.log2(nb * (nb - 1))
    h_normalized = entropy_bits / h_max_bits

    # ── z_entropy: shuffle null (permute from, keep to fixed) ────────────
    h_shuffle = np.empty(N_SHUFFLES, dtype=np.float64)
    for s in range(N_SHUFFLES):
        shuff_from   = from_ids[np.random.permutation(n_transitions)]
        h_shuffle[s] = _transition_entropy(shuff_from, to_ids)

    shuffle_mean = float(h_shuffle.mean())
    shuffle_std  = float(h_shuffle.std(ddof=1))
    z_entropy    = ((entropy_bits - shuffle_mean) / shuffle_std
                    if shuffle_std > 0 else 0.0)

    # ── Assemble result ───────────────────────────────────────────────────
    result = {
        "entropy_bits"      : entropy_bits,
        "entropy_nats"      : entropy_nats,
        "h_max_bits"        : h_max_bits,
        "h_normalized"      : h_normalized,
        "z_entropy"         : z_entropy,
        "h_shuffle_mean"    : shuffle_mean,
        "h_shuffle_std"     : shuffle_std,
        "h_shuffle_all"     : h_shuffle,
        "n_transitions"     : n_transitions,
        "transition_matrix" : P,
        "strategy_name"     : strategy_name,
        "grid_size"         : grid_size,
        "seq_length"        : seq_length,
        "num_seq"           : num_seq,
    }

    # ── Verbose output ────────────────────────────────────────────────────
    if verbose:
        display   = strategy_name.replace("strategy_", "").replace("_", " ").title()
        img_label = f"  img={img_index}" if img_index is not None else ""
        bar_len   = 30

        bar_h     = round(h_normalized * bar_len)
        bar       = "█" * bar_h + "░" * (bar_len - bar_h)

        z_clamped = max(-4.0, min(4.0, z_entropy))
        bar_z     = "█" * round((z_clamped + 4.0) / 8.0 * bar_len) + \
                    "░" * (bar_len - round((z_clamped + 4.0) / 8.0 * bar_len))
        z_sign    = "+" if z_entropy >= 0 else ""

        print(f"\n{'─'*58}")
        print(f"  Strategy      : {display}{img_label}")
        print(f"  Grid          : {grid_size}×{grid_size}   "
              f"seq_length={seq_length}   num_seq={num_seq}")
        print(f"{'─'*58}")
        print(f"  Transitions   : {n_transitions:,}   (shuffles={N_SHUFFLES})")
        print(f"  H (bits)      : {entropy_bits:.4f}")
        print(f"  H (nats)      : {entropy_nats:.4f}")
        print(f"  H_max (bits)  : {h_max_bits:.4f}  [log2({nb}×{nb-1})]")
        print(f"  H / H_max     : {h_normalized:.4f}   [{bar}]")
        print(f"  H_shuff mean  : {shuffle_mean:.4f}  std={shuffle_std:.4f}")
        print(f"  z_entropy     : {z_sign}{z_entropy:.4f}   [{bar_z}]  "
              f"(-4σ ←——→ +4σ)")
        print(f"{'─'*58}\n")

    return result


# ── CLI convenience ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Calculate global transition entropy for a selection strategy."
    )
    parser.add_argument("strategy",              type=str)
    parser.add_argument("--grid_size",  "-g",    type=int, default=4)
    parser.add_argument("--seq_length", "-l",    type=int, default=6)
    parser.add_argument("--num_seq",    "-n",    type=int, default=100)
    parser.add_argument("--img_index",  "-i",    type=int, default=None,
                        help="Image index for image-based strategies "
                             "(e.g. 1 → images/img_1.png)")
    parser.add_argument("--seed",                type=int, default=None)
    parser.add_argument("--quiet",      "-q",    action="store_true",
                        help="Suppress printed summary")
    args = parser.parse_args()

    calculate_entropy(
        grid_size=args.grid_size,
        seq_length=args.seq_length,
        num_seq=args.num_seq,
        strategy_name=args.strategy,
        img_index=args.img_index,
        seed=args.seed,
        verbose=not args.quiet,
    )
