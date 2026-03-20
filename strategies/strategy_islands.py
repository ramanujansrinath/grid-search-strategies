"""
strategy_islands.py
--------------------
Strategy: ISLANDS (two clusters of three)

Structure of a sequence
-----------------------
  Island 1: seed_1, neighbor_1a, neighbor_1b
  Island 2: seed_2, neighbor_2a, neighbor_2b

Each island is built as:
  1. Pick a seed uniformly at random from the remaining unvisited boxes.
  2. Collect the seed's unvisited grid neighbours.
  3. If ≥ 2 neighbours are available: pick 2 uniformly at random (without
     replacement within this step).
  4. If only 1 neighbour is available: take it; pick 1 random box from the
     remaining unvisited pool to complete the island to size 3.
  5. If 0 neighbours are available: pick 2 random boxes from the remaining
     unvisited pool.

The two islands need not be spatially separated.

Examples (grid size 4, matches user's illustration):
  seed=1  → could give [1, 2, 5]
  seed=7  → could give [7, 8, 11]
  full sequence: [1, 2, 5, 7, 8, 11]
"""

from __future__ import annotations
import random
from typing import List

from grid_utils import all_boxes, get_neighbors, random_choice, pick_and_remove


def _build_island(available: set, N: int, island_size: int = 3) -> List[int]:
    """
    Pick a seed and fill an island of *island_size* boxes from *available*.
    Mutates *available* in place.
    """
    island: List[int] = []

    # Seed
    seed = pick_and_remove(available)
    island.append(seed)

    # Gather unvisited neighbours of the seed
    nbrs = [b for b in get_neighbors(seed, N) if b in available]
    random.shuffle(nbrs)

    # Fill the rest of the island from neighbours first, then random fallback
    slots_remaining = island_size - 1
    for nbr in nbrs:
        if slots_remaining == 0:
            break
        if nbr in available:
            available.remove(nbr)
            island.append(nbr)
            slots_remaining -= 1

    # Fallback: if not enough neighbours, pick randomly
    while slots_remaining > 0 and available:
        island.append(pick_and_remove(available))
        slots_remaining -= 1

    return island


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

    Note
    ----
    *seq_length* is assumed to be divisible by the island size (default 3).
    If not, the last island may be smaller than the others.
    """
    if seq_length > grid_size ** 2:
        raise ValueError("seq_length cannot exceed grid_size².")

    N = grid_size
    ISLAND_SIZE = 3
    n_islands = seq_length // ISLAND_SIZE
    remainder = seq_length % ISLAND_SIZE

    sequences: List[List[int]] = []

    for _ in range(n_seq):
        available = set(all_boxes(N))
        seq: List[int] = []

        for _ in range(n_islands):
            if not available:
                break
            island = _build_island(available, N, ISLAND_SIZE)
            seq.extend(island)

        # Handle remainder (if seq_length not divisible by ISLAND_SIZE)
        if remainder > 0 and available:
            partial = _build_island(available, N, remainder)
            seq.extend(partial)

        sequences.append(seq[:seq_length])

    return sequences
