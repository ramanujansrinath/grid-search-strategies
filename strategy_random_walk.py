"""
strategy_random_walk.py
------------------------
Strategy: RANDOM WALK (self-avoiding)

Rule
----
1. First pick: uniformly random from all boxes.
2. At each step, move to a uniformly random *unvisited* grid neighbour of
   the current box (up/down/left/right).  This is a self-avoiding random walk.
3. If the walk is "trapped" (no unvisited neighbours remain adjacent), fall
   back to a uniformly random pick from *all* remaining unvisited boxes, then
   resume the walk from that new position.

This differs from neighbor_first in that the movement is genuinely a "walk"
metaphor (each step is a single grid step from the previous position), making
the spatial path more locally coherent.
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
                # Continue the walk
                current = random_choice(unvisited_nbrs)
            else:
                # Walk is trapped — teleport to a random unvisited box
                current = random_choice(available)

            available.remove(current)
            seq.append(current)

        sequences.append(seq)

    return sequences
