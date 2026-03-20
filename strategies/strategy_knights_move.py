"""
strategy_knights_move.py
------------------------
Strategy: KNIGHT'S MOVE ONLY

Rule
----
1. First pick: uniformly random from all boxes.
2. At each step, collect all unvisited boxes reachable from the *current* box
   by a single chess knight move (L-shaped: 2 squares in one axis, 1 in the
   other).  Pick one uniformly at random.
3. If no valid knight destination is available (all reachable squares already
   visited), fall back to a uniformly random pick from all remaining unvisited
   boxes, then reapply the knight rule from that new box.

Note: on a 4×4 board every square has at least 2 valid knight destinations on
an empty board.  Dead ends arise only after several boxes have been visited.
"""

from __future__ import annotations
from typing import List

from grid_utils import all_boxes, get_knight_moves, random_choice, pick_and_remove


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
            knight_dests = [b for b in get_knight_moves(current, N) if b in available]

            if knight_dests:
                current = random_choice(knight_dests)
            else:
                # Fallback: random pick from all remaining, rule restarts
                current = random_choice(available)

            available.remove(current)
            seq.append(current)

        sequences.append(seq)

    return sequences
