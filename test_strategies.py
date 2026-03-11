"""
test_strategies.py
------------------
Smoke-tests and basic correctness checks for all selection strategies.

Checks per strategy:
  1. Returns exactly n_seq sequences.
  2. Each sequence has exactly seq_length boxes.
  3. All boxes in a sequence are valid (1 … grid_size²).
  4. No duplicates within a sequence (selections are without replacement).
"""

import sys
import traceback
import importlib
from pathlib import Path

# Make sure the strategies directory is on the path
sys.path.insert(0, str(Path(__file__).parent))

STRATEGIES = [
    "strategy_random",
    "strategy_rowwise",
    "strategy_rowwise_sequential",
    "strategy_center_out_radial",
    "strategy_center_out_spiral",
    "strategy_neighbor_first",
    "strategy_nearest_first",
    "strategy_farthest_first",
    "strategy_checkerboard",
    "strategy_knights_move",
    "strategy_snake",
    "strategy_perimeter_crawl",
    "strategy_islands",
    "strategy_random_walk",
    "strategy_diagonal_sweep",
    "strategy_weighted_center_bias",
    "strategy_hilbert_curve",
]

GRID_SIZE  = 4
SEQ_LENGTH = 6
N_SEQ      = 200

def check(name, seqs, grid_size, seq_length, n_seq):
    errors = []
    if len(seqs) != n_seq:
        errors.append(f"Expected {n_seq} sequences, got {len(seqs)}")
    for i, seq in enumerate(seqs):
        if len(seq) != seq_length:
            errors.append(f"Seq {i}: length {len(seq)} ≠ {seq_length}")
        if len(set(seq)) != len(seq):
            errors.append(f"Seq {i}: duplicate boxes {seq}")
        for b in seq:
            if not (1 <= b <= grid_size**2):
                errors.append(f"Seq {i}: invalid box {b}")
    return errors


def run_extra_checks(name, seqs, grid_size):
    """Strategy-specific sanity checks."""
    N = grid_size
    issues = []

    if name == "strategy_checkerboard":
        from grid_utils import get_row_col
        for i, seq in enumerate(seqs):
            colors = {((b-1)//N + (b-1)%N) % 2 for b in seq}
            if len(colors) > 1:
                issues.append(f"Seq {i}: mixed checkerboard colors {seq}")

    if name == "strategy_rowwise":
        from grid_utils import get_row_col
        for i, seq in enumerate(seqs):
            # Each run of boxes until a "break" should all share a row
            # Just check that the first two boxes share a row OR the second
            # was a fallback (hard to check deterministically).
            pass  # Light check only

    if name == "strategy_hilbert_curve":
        from strategy_hilbert_curve import _build_hilbert_order
        order = _build_hilbert_order(N)
        assert len(order) == N*N, f"Hilbert order length {len(order)} ≠ {N*N}"
        assert set(order) == set(range(1, N*N+1)), "Hilbert order missing boxes"

    return issues


print(f"Testing {len(STRATEGIES)} strategies  "
      f"[grid={GRID_SIZE}×{GRID_SIZE}, seq_length={SEQ_LENGTH}, n_seq={N_SEQ}]\n")

all_passed = True
for strat in STRATEGIES:
    try:
        mod = importlib.import_module(strat)
        seqs = mod.generate_sequences(
            grid_size=GRID_SIZE, seq_length=SEQ_LENGTH, n_seq=N_SEQ
        )
        errs = check(strat, seqs, GRID_SIZE, SEQ_LENGTH, N_SEQ)
        extra = run_extra_checks(strat, seqs, GRID_SIZE)
        errs += extra

        if errs:
            all_passed = False
            print(f"  FAIL  {strat}")
            for e in errs[:5]:
                print(f"        {e}")
        else:
            # Print a sample sequence for visual inspection
            sample = seqs[0]
            print(f"  PASS  {strat:<38}  sample: {sample}")

    except Exception:
        all_passed = False
        print(f"  ERROR {strat}")
        traceback.print_exc()

print()

# Additional: test with a different grid/seq_length
print("Running with grid_size=5, seq_length=8, n_seq=50 …")
for strat in STRATEGIES:
    try:
        mod = importlib.import_module(strat)
        seqs = mod.generate_sequences(grid_size=5, seq_length=8, n_seq=50)
        errs = check(strat, seqs, 5, 8, 50)
        if errs:
            all_passed = False
            print(f"  FAIL  {strat}  {errs[:2]}")
        else:
            print(f"  PASS  {strat}")
    except Exception:
        all_passed = False
        print(f"  ERROR {strat}")
        traceback.print_exc()

print()
print("=" * 60)
print("ALL TESTS PASSED" if all_passed else "SOME TESTS FAILED")
