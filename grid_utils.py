"""
grid_utils.py
-------------
Shared utilities for all selection strategies.

Grid conventions
----------------
- Boxes are 1-indexed, labelled row-wise: 1…N² for an N×N grid.
- Box i has 0-indexed row  = (i-1) // N,  col = (i-1) % N.
- Neighbours are the four cardinal grid cells (up, down, left, right).
- Geometric centre of the grid is at ((N-1)/2, (N-1)/2) in 0-indexed (row, col).
"""

from __future__ import annotations
import math
import random
from typing import Iterable, List, Set, Tuple


# ──────────────────────────────────────────────
# Basic position helpers
# ──────────────────────────────────────────────

def get_row_col(box: int, N: int) -> Tuple[int, int]:
    """Return (row, col) for a 1-indexed box in an N×N grid."""
    return (box - 1) // N, (box - 1) % N


def get_box(row: int, col: int, N: int) -> int:
    """Return 1-indexed box number for 0-indexed (row, col)."""
    return row * N + col + 1


def in_bounds(row: int, col: int, N: int) -> bool:
    return 0 <= row < N and 0 <= col < N


# ──────────────────────────────────────────────
# Neighbourhood
# ──────────────────────────────────────────────

_CARDINAL = ((-1, 0), (1, 0), (0, -1), (0, 1))   # up, down, left, right


def get_neighbors(box: int, N: int) -> List[int]:
    """Return the list of cardinal-direction grid neighbours of *box*."""
    r, c = get_row_col(box, N)
    return [
        get_box(r + dr, c + dc, N)
        for dr, dc in _CARDINAL
        if in_bounds(r + dr, c + dc, N)
    ]


def get_knight_moves(box: int, N: int) -> List[int]:
    """Return all valid chess knight destinations from *box*."""
    r, c = get_row_col(box, N)
    jumps = [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]
    return [
        get_box(r + dr, c + dc, N)
        for dr, dc in jumps
        if in_bounds(r + dr, c + dc, N)
    ]


# ──────────────────────────────────────────────
# Distance helpers
# ──────────────────────────────────────────────

def euclidean(box1: int, box2: int, N: int) -> float:
    """Euclidean distance between the centres of two boxes."""
    r1, c1 = get_row_col(box1, N)
    r2, c2 = get_row_col(box2, N)
    return math.sqrt((r1 - r2) ** 2 + (c1 - c2) ** 2)


def dist_from_center(box: int, N: int) -> float:
    """Euclidean distance from box centre to geometric centre of the grid."""
    r, c = get_row_col(box, N)
    cr, cc = (N - 1) / 2.0, (N - 1) / 2.0
    return math.sqrt((r - cr) ** 2 + (c - cc) ** 2)


# ──────────────────────────────────────────────
# Misc helpers
# ──────────────────────────────────────────────

def random_choice(items: Iterable) -> object:
    """Uniformly random choice from an iterable (sorts for determinism)."""
    lst = sorted(items)
    return random.choice(lst)


def all_boxes(N: int) -> List[int]:
    """Return all box numbers for an N×N grid."""
    return list(range(1, N * N + 1))


def pick_and_remove(available: set) -> int:
    """Pick a random element from *available*, remove it and return it."""
    choice = random_choice(available)
    available.remove(choice)
    return choice


def best_from(candidates: Iterable[int],
              key,
              available: Set[int]) -> int:
    """
    Among *candidates* that are in *available*, pick the one(s) with the
    maximum value of *key*, then break ties uniformly at random.
    """
    pool = [b for b in candidates if b in available]
    if not pool:
        return pick_and_remove(available)
    best_val = max(key(b) for b in pool)
    best = [b for b in pool if key(b) == best_val]
    choice = random_choice(best)
    available.remove(choice)
    return choice


def worst_from(candidates: Iterable[int],
               key,
               available: Set[int]) -> int:
    """Like best_from but picks the minimum."""
    pool = [b for b in candidates if b in available]
    if not pool:
        return pick_and_remove(available)
    worst_val = min(key(b) for b in pool)
    worst = [b for b in pool if key(b) == worst_val]
    choice = random_choice(worst)
    available.remove(choice)
    return choice
