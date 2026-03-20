"""
strategy_random.py
------------------
Strategy: RANDOM
Select *seq_length* boxes uniformly at random without replacement.

No rules, no constraints — serves as a baseline.
"""

from __future__ import annotations
import random
from typing import List

from grid_utils import all_boxes


def generate_sequences(
    grid_size: int = 4,
    seq_length: int = 6,
    n_seq: int = 100,
) -> List[List[int]]:
    """
    Parameters
    ----------
    grid_size  : side length of the square grid (grid_size × grid_size boxes).
    seq_length : number of boxes selected per sequence.
    n_seq      : number of independent sequences to generate.

    Returns
    -------
    List of *n_seq* sequences, each a list of *seq_length* distinct box numbers.
    """
    if seq_length > grid_size ** 2:
        raise ValueError("seq_length cannot exceed grid_size².")

    boxes = all_boxes(grid_size)
    return [random.sample(boxes, seq_length) for _ in range(n_seq)]
