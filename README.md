# Grid Selection Strategies

A Python library for generating and analysing structured spatial selection sequences on an N×N grid. Each strategy defines a different rule for choosing which boxes to visit, without replacement, producing sequences that vary from fully random to highly deterministic paths. The library includes visualisation and entropy analysis utilities.

---

## Motivation

When studying spatial sampling, attention patterns, or search algorithms on grids, the *order* in which cells are visited carries as much information as *which* cells are visited. This library provides a controlled, extensible set of strategies — from random baselines to structured walks like spirals, knight's moves, and Hilbert curves — along with tools to visualise and quantitatively compare them via transition entropy.

---

## Installation

No package installation required. Clone the repository and ensure dependencies are available:

```bash
git clone https://github.com/your-username/grid-selection-strategies.git
cd grid-selection-strategies
pip install matplotlib numpy
```

Python 3.8+ is supported.

---

## Quick Start

```python
from strategy_random import generate_sequences
from draw_sample import drawSample
from calculate_entropy import calculate_entropy

# Generate 100 sequences of length 6 on a 4×4 grid
seqs = generate_sequences(grid_size=4, seq_length=6, n_seq=100)

# Visualise all sequences in a single tiled figure
drawSample(grid_size=4, seq_length=6, num_seq=100,
           strategy_name="neighbor_first",
           save_path="neighbor_first.png", seed=42)

# Compute global transition entropy
result = calculate_entropy(grid_size=4, seq_length=6, num_seq=500,
                           strategy_name="hilbert_curve", seed=42)
print(f"H = {result['entropy_bits']:.4f} bits  "
      f"(H/H_max = {result['h_normalized']:.4f})")
```

---

## Strategies

All strategies share an identical function signature:

```python
generate_sequences(grid_size=4, seq_length=6, n_seq=100) -> List[List[int]]
```

Boxes are labelled 1…N² row-wise. Neighbours are the four cardinal directions (up, down, left, right). Every strategy starts with a **uniformly random first pick**. If a rule reaches a dead end, a random fallback is used and the rule restarts from there. Ties are always broken uniformly at random.

| # | Module | Strategy | Description |
|---|--------|----------|-------------|
| 1 | `strategy_random` | **Random** | Uniform sampling without replacement. Baseline. |
| 2 | `strategy_rowwise` | **Row-wise** | Stays within the current row; falls back to random when the row is exhausted. |
| 3 | `strategy_rowwise_sequential` | **Row-wise Sequential** | Reads left-to-right along the canonical row order from a random start, wrapping modularly. |
| 4 | `strategy_center_out_radial` | **Center-Out Radial** | Each step moves to the unvisited neighbour that is farthest from the geometric grid centre. |
| 5 | `strategy_center_out_spiral` | **Center-Out Spiral** | Clockwise outward spiral from a random start, using fixed-leg expansion (L1, U1, R2, D2, …). |
| 6 | `strategy_neighbor_first` | **Neighbor First** | Each pick must be a grid neighbour of the previous pick; random fallback when trapped. |
| 7 | `strategy_nearest_first` | **Nearest First** | Always picks the globally nearest unvisited box (Euclidean distance from current). |
| 8 | `strategy_farthest_first` | **Farthest First** | Always picks the globally farthest unvisited box. Maximises spatial spread. |
| 9 | `strategy_checkerboard` | **Checkerboard** | All picks share the colour (parity) of the first box; random within that colour set. |
| 10 | `strategy_knights_move` | **Knight's Move** | Each step is a chess knight move (L-shaped) from the previous box. |
| 11 | `strategy_snake` | **Snake** | Boustrophedon sweep (alternating row/column direction) from a random start with random orientation. |
| 12 | `strategy_perimeter_crawl` | **Perimeter Crawl** | Crawls the outer ring clockwise or counter-clockwise; interior first picks route to the nearest perimeter box first. |
| 13 | `strategy_islands` | **Islands** | Builds two spatially clustered islands of three: each island is a seed plus two of its neighbours. |
| 14 | `strategy_random_walk` | **Random Walk** | Self-avoiding random walk along grid edges; teleports when trapped. |
| 15 | `strategy_diagonal_sweep` | **Diagonal Sweep** | Sweeps along anti-diagonals (constant row+col), picking randomly within each diagonal. |
| 16 | `strategy_weighted_center_bias` | **Weighted Center Bias** | Weighted random sampling with weights ∝ 1/distance from grid centre, computed once and held fixed. |
| 17 | `strategy_hilbert_curve` | **Hilbert Curve** | Follows the space-filling Hilbert curve traversal order from a random offset, wrapping modularly. |

---

## Utilities

### `drawSample`

Generates sequences and renders them as a tiled grid figure. Each subplot shows one sequence: unvisited cells in dark navy, visited cells coloured from deep purple (first pick) to bright yellow (last pick) via the `plasma` colormap.

```python
drawSample(
    grid_size     = 4,
    seq_length    = 6,
    num_seq       = 100,
    strategy_name = "center_out_spiral",
    save_path     = "spiral.png",   # omit or None to call plt.show()
    seed          = 42,
)
```

Also available as a CLI:
```bash
python draw_sample.py center_out_spiral -g 4 -l 6 -n 100 -s spiral.png --seed 42
```

### `calculate_entropy`

Computes the **global transition entropy** of a strategy: builds the N²×N² transition count matrix across all sequences, normalises it into a joint probability distribution P(i→j), and calculates Shannon entropy in bits.

```python
result = calculate_entropy(
    grid_size     = 4,
    seq_length    = 6,
    num_seq       = 500,
    strategy_name = "knights_move",
    seed          = 42,
    verbose       = True,   # prints a formatted summary
)
# result keys: entropy_bits, entropy_nats, h_max_bits,
#              h_normalized, n_transitions, transition_matrix
```

H_max = log₂(N² × (N²−1)) bits — entropy of a uniform distribution over all ordered pairs of distinct boxes. `h_normalized` in [0, 1] allows direct comparison across strategies.

Also available as a CLI:
```bash
python calculate_entropy.py knights_move -g 4 -l 6 -n 500 --seed 42
```

---

## Entropy Benchmarks

Ranked by normalised entropy (4×4 grid, seq_length=6, 500 sequences, seed=42):

| Strategy | H (bits) | H / H_max |
|----------|----------|-----------|
| Random | 7.84 | 0.99 |
| Weighted Center Bias | 7.65 | 0.97 |
| Islands | 7.24 | 0.92 |
| Checkerboard | 6.77 | 0.86 |
| Row-wise | 6.63 | 0.84 |
| Center-Out Radial | 6.19 | 0.78 |
| Diagonal Sweep | 5.71 | 0.72 |
| Knight's Move | 5.69 | 0.72 |
| Neighbor First | 5.65 | 0.72 |
| Random Walk | 5.65 | 0.72 |
| Nearest First | 5.63 | 0.71 |
| Snake | 5.63 | 0.71 |
| Center-Out Spiral | 5.50 | 0.70 |
| Farthest First | 4.89 | 0.62 |
| Perimeter Crawl | 4.76 | 0.60 |
| Row-wise Sequential | 3.998 | 0.506 |
| Hilbert Curve | 3.997 | 0.506 |

H_max = 7.907 bits (log₂(240) for a 4×4 grid).

---

## File Structure

```
grid-selection-strategies/
├── grid_utils.py                     # shared grid primitives
├── strategy_random.py
├── strategy_rowwise.py
├── strategy_rowwise_sequential.py
├── strategy_center_out_radial.py
├── strategy_center_out_spiral.py
├── strategy_neighbor_first.py
├── strategy_nearest_first.py
├── strategy_farthest_first.py
├── strategy_checkerboard.py
├── strategy_knights_move.py
├── strategy_snake.py
├── strategy_perimeter_crawl.py
├── strategy_islands.py
├── strategy_random_walk.py
├── strategy_diagonal_sweep.py
├── strategy_weighted_center_bias.py
├── strategy_hilbert_curve.py
├── draw_sample.py                    # visualisation utility
├── calculate_entropy.py              # entropy analysis utility
├── test_strategies.py                # smoke tests
├── DOCUMENTATION.md
└── README.md
```

---

## Running the Tests

```bash
python test_strategies.py
```

Tests verify that every strategy returns the correct number of sequences, each of the correct length, containing only valid and non-duplicate box numbers. Both 4×4 and 5×5 grids are tested.

---

## Adding a New Strategy

1. Create `strategy_myname.py` in the project directory.
2. Implement `generate_sequences(grid_size, seq_length, n_seq) -> List[List[int]]`.
3. Import helpers from `grid_utils` as needed.
4. Pass `strategy_name="myname"` to `drawSample` and `calculate_entropy`.

The strategy will be automatically discovered by both utilities.

---

## License

MIT
