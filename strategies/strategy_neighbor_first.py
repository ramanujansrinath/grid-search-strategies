"""
strategy_neighbor_first.py
---------------------------
Strategy: NEIGHBOR FIRST (strict adjacency)

Rule
----
1. First pick: uniformly random from all boxes.
2. At each step, look at the unvisited grid neighbours (up/down/left/right)
   of the *current* box.  Pick one uniformly at random.
3. If the current box has *no* unvisited grid neighbours, fall back to a
   uniformly random pick from all remaining unvisited boxes, then reapply
   the neighbour rule from that new box.

This is a strict structural rule (not probability-weighted): the next pick
*must* be an adjacent box when one is available.
"""

from __future__ import annotations
from typing import List

from grid_utils import all_boxes, get_neighbors, random_choice, pick_and_remove


def generate_sequences(
    grid_size: int = 4,
    seq_length: int = 6,
    n_seq: int = 100,
) -> List[List[int]]:
    """
    Parameters
    ----------
    grid_size  : side length of the square grid.
    seq_length : number of boxes per sequence.
    n_seq      : number of sequences to generate.

    Returns
    -------
    List of *n_seq* sequences, each of length *seq_length*.
    """
    if seq_length > grid_size ** 2:
        raise ValueError("seq_length cannot exceed grid_size².")

    N = grid_size
    sequences: List[List[int]] = []

    for _ in range(n_seq):
        available = set(all_boxes(N))
        seq: List[int] = []

        # ── First pick: uniformly random ──────────────────────────────────
        current = pick_and_remove(available)
        seq.append(current)

        while len(seq) < seq_length:
            unvisited_nbrs = [b for b in get_neighbors(current, N) if b in available]

            if unvisited_nbrs:
                current = random_choice(unvisited_nbrs)
            else:
                # Fallback: random pick anywhere, then rule restarts
                current = random_choice(available)

            available.remove(current)
            seq.append(current)

        sequences.append(seq)

    return sequences
