# Grid Selection Strategies — Detailed Documentation

## Table of Contents

1. [Concepts and Conventions](#1-concepts-and-conventions)
2. [Shared Infrastructure: `grid_utils.py`](#2-shared-infrastructure-grid_utilspy)
3. [Universal Rules](#3-universal-rules)
4. [Strategy Reference](#4-strategy-reference)
   - 4.1 [Random](#41-random)
   - 4.2 [Row-wise](#42-row-wise)
   - 4.3 [Row-wise Sequential](#43-row-wise-sequential)
   - 4.4 [Center-Out Radial](#44-center-out-radial)
   - 4.5 [Center-Out Spiral](#45-center-out-spiral)
   - 4.6 [Neighbor First](#46-neighbor-first)
   - 4.7 [Nearest First](#47-nearest-first)
   - 4.8 [Farthest First](#48-farthest-first)
   - 4.9 [Checkerboard](#49-checkerboard)
   - 4.10 [Knight's Move](#410-knights-move)
   - 4.11 [Snake](#411-snake)
   - 4.12 [Perimeter Crawl](#412-perimeter-crawl)
   - 4.13 [Islands](#413-islands)
   - 4.14 [Random Walk](#414-random-walk)
   - 4.15 [Diagonal Sweep](#415-diagonal-sweep)
   - 4.16 [Weighted Center Bias](#416-weighted-center-bias)
   - 4.17 [Hilbert Curve](#417-hilbert-curve)
5. [Utility: `drawSample`](#5-utility-drawsample)
6. [Utility: `calculate_entropy`](#6-utility-calculate_entropy)
7. [Design Decisions and Edge Cases](#7-design-decisions-and-edge-cases)
8. [Extending the Library](#8-extending-the-library)

---

## 1. Concepts and Conventions

### The Grid

An N×N grid of boxes, labelled 1…N² in row-major (left-to-right, top-to-bottom) order:

```
 1  2  3  4
 5  6  7  8
 9 10 11 12
13 14 15 16
```

Each box has a **1-indexed label** used throughout the public API. Internally, boxes are converted to 0-indexed `(row, col)` pairs for arithmetic.

```
box  →  row = (box - 1) // N
        col = (box - 1) %  N

(row, col)  →  box = row * N + col + 1
```

### Neighbours

Neighbours are the four **cardinal** grid cells — up, down, left, right — that exist within bounds. Corner and edge boxes have 2 or 3 neighbours respectively; interior boxes have 4.

```
Neighbours of box 6 in a 4×4 grid:  2 (up), 10 (down), 5 (left), 7 (right)
```

### Geometric Centre

The **geometric centre** of an N×N grid is the point `((N-1)/2, (N-1)/2)` in 0-indexed `(row, col)` space. For a 4×4 grid this is `(1.5, 1.5)` — equidistant between boxes 6, 7, 10, 11. Distances are Euclidean between box centres and this point.

### Sequences

A **sequence** is an ordered list of `seq_length` boxes, selected **without replacement** from the N²-box pool. Boxes within a single sequence are always distinct. Across different sequences in the same batch, repetition is entirely normal.

---

## 2. Shared Infrastructure: `grid_utils.py`

All strategy modules import from `grid_utils`. The module exposes:

| Function | Signature | Description |
|----------|-----------|-------------|
| `get_row_col` | `(box, N) → (row, col)` | Convert 1-indexed box to 0-indexed position |
| `get_box` | `(row, col, N) → box` | Convert 0-indexed position to 1-indexed box |
| `in_bounds` | `(row, col, N) → bool` | Check whether a position exists in the grid |
| `get_neighbors` | `(box, N) → List[int]` | Cardinal neighbours of a box |
| `get_knight_moves` | `(box, N) → List[int]` | All valid chess knight destinations |
| `euclidean` | `(box1, box2, N) → float` | Distance between two box centres |
| `dist_from_center` | `(box, N) → float` | Distance from box centre to grid centre |
| `random_choice` | `(items) → item` | Uniform random choice (sorts input for stability) |
| `all_boxes` | `(N) → List[int]` | All boxes `[1, …, N²]` |
| `pick_and_remove` | `(available: set) → int` | Random pick that mutates the available set |
| `best_from` | `(candidates, key, available) → int` | Pick the candidate with maximum `key`, ties broken randomly |
| `worst_from` | `(candidates, key, available) → int` | Pick the candidate with minimum `key`, ties broken randomly |

---

## 3. Universal Rules

These rules apply **across all strategies**:

**First pick** — Every sequence, regardless of strategy, begins with a uniformly random pick from all N² boxes. No strategy pre-constrains the starting position.

**Without replacement** — Once a box is selected it is removed from the available pool and cannot be re-selected within the same sequence.

**Dead-end fallback** — If the active strategy rule cannot produce a valid next pick (e.g. all neighbours are visited, no knight moves remain, the colour pool is empty), a uniformly random pick is made from the remaining available boxes, and the strategy rule is reapplied from that new position.

**Tie-breaking** — Whenever multiple candidates are equally valid under a rule (e.g. multiple neighbours at equal distance, multiple boxes in the same diagonal), one is chosen uniformly at random.

---

## 4. Strategy Reference

### 4.1 Random

**File:** `strategy_random.py`

The baseline. Each sequence is simply a uniform random sample of `seq_length` boxes without replacement, drawn independently using `random.sample`. There is no structural rule beyond the first pick — the first pick and all subsequent picks are all drawn simultaneously from a shuffle.

**Entropy:** Near-maximum (~0.99 H/H_max). All transitions roughly equally probable.

---

### 4.2 Row-wise

**File:** `strategy_rowwise.py`

**Rule:** After the first random pick, each subsequent pick is chosen uniformly at random from the **unvisited boxes in the same row** as the current box. When the current row is exhausted, a random fallback picks any remaining box, and the rule restarts from that box's row.

**Key behaviour:** Within a row segment, order is random (not sequential). The sequence is partitioned into row-runs of varying length, separated by random jumps.

**Example (4×4, seq_length=6):**
```
Pick 1 (random): 3   → row 0
Pick 2 (row 0):  1   → row 0
Pick 3 (row 0):  4   → row 0
Pick 4 (row 0):  2   → row 0  [row 0 now exhausted]
Pick 5 (random): 7   → row 1
Pick 6 (row 1):  6
```

---

### 4.3 Row-wise Sequential

**File:** `strategy_rowwise_sequential.py`

**Rule:** The canonical order is `1, 2, 3, …, N²` (left-to-right, top-to-bottom). A random starting index is chosen uniformly in `[0, N²)`. The sequence takes `seq_length` consecutive entries from this canonical order, wrapping modularly past the last box back to box 1.

This strategy is fully deterministic given a starting index — no further randomness is applied after the first pick.

**Example (4×4, seq_length=6, start=13):**
```
13 → 14 → 15 → 16 → 1 → 2
```

**Entropy:** ~0.51 H/H_max. Only N² distinct transitions are possible (each box has exactly one successor in the canonical order), severely limiting entropy.

---

### 4.4 Center-Out Radial

**File:** `strategy_center_out_radial.py`

**Rule:** After the first random pick, at each step inspect the **grid neighbours** (not all boxes) of the current box. Among those neighbours that are (a) unvisited and (b) strictly farther from the geometric grid centre than the current box, pick the one with maximum distance. If no such outward neighbour exists, fall back to a random pick and restart.

**Distance metric:** Euclidean from box centre to grid geometric centre `((N-1)/2, (N-1)/2)`.

**Distances precomputed** once per `generate_sequences` call and reused across all sequences.

**Key behaviour:** The sequence moves outward from wherever it starts. Boxes at the perimeter are the last to be reached; interior boxes early in the sequence trigger frequent fallbacks because their outward neighbours may already be visited.

---

### 4.5 Center-Out Spiral

**File:** `strategy_center_out_spiral.py`

**Rule:** A clockwise outward spiral is precomputed from each starting box. The spiral uses the **fixed-leg expansion scheme**: legs cycle through directions L→U→R→D with lengths 1, 1, 2, 2, 3, 3, 4, 4, … The starting direction is chosen randomly from {L, U} if both are in bounds, otherwise the available one, otherwise R or D.

Position always advances within bounds; out-of-bounds steps freeze the position for that step. The spiral visits every box exactly once. The sequence follows this precomputed order, collecting only unvisited boxes.

**Verified examples (4×4):**

| Start | Spiral path (first 6) |
|-------|----------------------|
| 6 | 6 → 5 → 1 → 2 → 3 → 7 |
| 10 | 10 → 9 → 5 → 6 → 7 → 11 |
| 13 | 13 → 9 → 10 → 14 → [fallback] |

**Entropy:** ~0.70 H/H_max. The fixed spiral structure concentrates transitions along specific paths, but the random start creates diversity.

---

### 4.6 Neighbor First

**File:** `strategy_neighbor_first.py`

**Rule:** After the first random pick, each pick must be an **unvisited grid neighbour** of the current box. If no unvisited neighbours remain, fall back to a random pick from all remaining boxes and restart.

This is a **strict structural** rule, not probabilistic — if a valid neighbour exists, the next pick *must* be one of them (chosen uniformly at random from all valid neighbours).

**Key behaviour:** Produces spatially contiguous runs until the current position is surrounded, then jumps. Similar to Random Walk but the fallback restarts cleanly rather than continuing the walk metaphor.

---

### 4.7 Nearest First

**File:** `strategy_nearest_first.py`

**Rule:** After the first random pick, each pick is the **globally nearest unvisited box** by Euclidean distance between box centres, across the entire remaining pool (not just neighbours). Ties broken randomly.

**Key behaviour:** Produces locally tight clusters that grow outward from the starting position. Tends to sweep nearby regions completely before jumping across the grid.

---

### 4.8 Farthest First

**File:** `strategy_farthest_first.py`

**Rule:** After the first random pick, each pick is the **globally farthest unvisited box** by Euclidean distance from the current box. Ties broken randomly.

**Key behaviour:** Produces sequences that jump maximally across the grid at every step, maximising spatial spread. Tends to bounce between opposite corners/edges.

**Entropy:** ~0.62 H/H_max. Fewer transitions are possible from a given box (only the farthest boxes qualify), creating a sparser transition matrix than random.

---

### 4.9 Checkerboard

**File:** `strategy_checkerboard.py`

**Colour rule:** Box `b` is **white** if `(row + col) % 2 == 0`, **black** otherwise (0-indexed coordinates). For N=4 this gives exactly 8 boxes per colour.

**Rule:** The first pick's colour determines the sequence colour. All subsequent picks are drawn uniformly at random from unvisited boxes of the **same colour**. If the colour pool is exhausted before `seq_length` boxes are collected, the fallback draws from the opposite colour.

**Key behaviour:** The spatial structure of a checkerboard means no two consecutive picks in the sequence are grid neighbours (since same-colour boxes are never adjacent), producing a distinctive non-local transition pattern.

---

### 4.10 Knight's Move

**File:** `strategy_knights_move.py`

**Rule:** After the first random pick, each pick must be reachable from the current box by a **single chess knight move** (L-shaped: 2 squares in one axis, 1 in the other), and must be unvisited. If no valid knight destinations remain, fall back to a random pick and restart.

**Knight destinations from box b:** up to 8 cells at offsets `(±1,±2)` and `(±2,±1)`, filtered to those in bounds.

**Note on 4×4 grids:** Every box has at least 2 valid knight destinations on an empty board (corners reach 2, most others reach 3–4). Dead ends occur only after several boxes have been visited.

**Entropy:** ~0.72 H/H_max. The knight's non-local, symmetric movement creates a rich but constrained transition structure.

---

### 4.11 Snake

**File:** `strategy_snake.py`

**Rule:** A snake (boustrophedon) traversal order is built for one of four orientations — **R** (rows, first row left→right), **L** (rows, first row right→left), **D** (columns, first column top→bottom), **U** (columns, first column bottom→top) — chosen uniformly at random per sequence.

Within each row/column, direction alternates. A random starting index is chosen within the full N²-length snake order, and the sequence takes `seq_length` consecutive entries with modular wrap.

**Example R-orientation (4×4):**
```
 1  2  3  4
 8  7  6  5
 9 10 11 12
16 15 14 13
```

**Entropy:** ~0.71 H/H_max. The four orientations and random offsets create variation, but within any sequence transitions are strictly sequential.

---

### 4.12 Perimeter Crawl

**File:** `strategy_perimeter_crawl.py`

**Perimeter definition:** The `4*(N-1)` outermost boxes in clockwise order: top row left→right, right column top→bottom (skipping top-right corner already counted), bottom row right→left (skipping bottom-right), left column bottom→top (skipping both bottom-left and top-left corners).

For 4×4: `1, 2, 3, 4, 8, 12, 16, 15, 14, 13, 9, 5`

**Rule:**
- If the first pick **is on the perimeter**: choose CW or CCW randomly, crawl from there.
- If the first pick **is interior**: find the nearest perimeter box (Euclidean, ties broken randomly), pick it, then begin the crawl CW or CCW.

The crawl wraps around the ring as needed. Already-visited perimeter boxes are skipped. If the entire perimeter is exhausted, remaining picks are drawn randomly.

**Entropy:** ~0.60 H/H_max. The crawl concentrates transitions along the 12-box perimeter ring, producing a highly constrained transition matrix.

---

### 4.13 Islands

**File:** `strategy_islands.py`

**Structure:** A sequence of `seq_length` boxes is assembled as `seq_length // 3` islands of exactly 3 boxes each (remainder handled if `seq_length` is not divisible by 3).

**Island construction:**
1. Pick a **seed** uniformly at random from all remaining unvisited boxes.
2. Collect the seed's **unvisited grid neighbours**.
3. Pick **2 neighbours** uniformly at random (without replacement within this step) to complete the island.
4. If fewer than 2 neighbours are available, fill the island with random picks from the remaining pool.

**Key behaviour:** Each island is a spatially compact cluster of 3. The seed-then-neighbours construction ensures islands are connected (the seed is adjacent to both selected neighbours), but islands are not required to be spatially separated from each other.

**Examples (4×4):**
```
Island 1: 1 → 2 → 5    (seed=1, neighbours 2 and 5)
Island 2: 7 → 8 → 11   (seed=7, neighbours 8 and 11)
Full sequence: [1, 2, 5, 7, 8, 11]
```

**Entropy:** ~0.92 H/H_max. High entropy because seeds re-randomize globally, but within each island transitions are locally constrained to neighbours.

---

### 4.14 Random Walk

**File:** `strategy_random_walk.py`

**Rule:** After the first random pick, each step moves to a uniformly random **unvisited grid neighbour** of the current box — a self-avoiding random walk on the grid graph. If the walk is trapped (no unvisited neighbours), teleport to a uniformly random unvisited box and resume.

**Difference from Neighbor First:** Both strategies are structurally nearly identical. The conceptual distinction is that Random Walk emphasises the *walk* metaphor (continuous movement along edges), while Neighbor First emphasises the *constraint* metaphor (next pick must be adjacent). In practice their implementations and entropies are identical (~0.72 H/H_max).

---

### 4.15 Diagonal Sweep

**File:** `strategy_diagonal_sweep.py`

**Diagonal convention:** Anti-diagonals defined by constant `k = row + col` (0-indexed). For a 4×4 grid, `k` runs from 0 to 6:

```
k=0: [1]
k=1: [2, 5]
k=2: [3, 6, 9]
k=3: [4, 7, 10, 13]
k=4: [8, 11, 14]
k=5: [12, 15]
k=6: [16]
```

**Rule:** The first pick determines its diagonal `k₀`. Diagonals are then swept in order `k₀, k₀+1, …` (modularly wrapping around after `k=2*(N-1)`). Within each diagonal, the available unvisited boxes are shuffled and added to the sequence until `seq_length` is reached.

**Entropy:** ~0.72 H/H_max. The diagonal structure constrains which boxes transition to which, but the random diagonal start and intra-diagonal shuffle create variety.

---

### 4.16 Weighted Center Bias

**File:** `strategy_weighted_center_bias.py`

**Rule:** All N² boxes are assigned fixed weights before any pick:

```
w(b) = 1 / dist(b, centre) ^ WEIGHT_POWER
```

where `dist` is Euclidean distance to the geometric centre and `WEIGHT_POWER = 1.0` (module-level constant, easily tuned). Boxes exactly at the centre (possible for odd N) receive a large finite weight (10× the maximum finite weight).

Picks are drawn via weighted random sampling (`random.choices`) over the remaining available pool, using the **pre-computed fixed weights**. Weights are **not renormalised** as boxes are removed — the probability of selecting box `b` from pool `A` is `w(b) / Σ_{j∈A} w(j)`.

**Entropy:** ~0.97 H/H_max. Despite the centre bias, the soft probabilistic weighting leaves most boxes accessible, producing near-random entropy.

**Tuning:** Increasing `WEIGHT_POWER` sharpens the centre bias; at very high values the strategy approaches a deterministic center-first ordering.

---

### 4.17 Hilbert Curve

**File:** `strategy_hilbert_curve.py`

**Rule:** The Hilbert curve traversal order for the N×N grid is precomputed using the standard `d2xy` algorithm (converts Hilbert distance `d` to `(x, y)` coordinates). For non-power-of-2 `N`, the next larger power-of-2 grid is used and out-of-bounds cells are filtered.

The 4×4 Hilbert order is:
```
1, 2, 6, 5, 9, 13, 14, 10, 11, 15, 16, 12, 8, 7, 3, 4
```

A random starting index is chosen within this order. The sequence takes `seq_length` consecutive entries with modular wrap.

**Key property:** The Hilbert curve is space-filling and locality-preserving — boxes that are close in the 1D Hilbert order tend to be close in 2D space.

**Entropy:** ~0.51 H/H_max — tied with Row-wise Sequential and for the same reason: the strategy is a fixed-offset read of a deterministic path, so only N² transitions are possible.

---

## 5. Utility: `drawSample`

**File:** `draw_sample.py`

### Signature

```python
drawSample(
    grid_size     : int  = 4,
    seq_length    : int  = 6,
    num_seq       : int  = 100,
    strategy_name : str  = "random",
    save_path     : str  = None,      # None → plt.show()
    seed          : int  = None,      # None → non-reproducible
) -> None
```

### Layout

Subplots are arranged in a near-square grid: `n_cols = ceil(sqrt(num_seq))`, `n_rows = ceil(num_seq / n_cols)`. Incomplete last rows are filled with blank (hidden) axes.

### Visual Encoding

| Element | Encoding |
|---------|----------|
| Unvisited cell | Dark navy (`#1a1a2e`) |
| Visited cell | `plasma` colormap: deep purple (pick 1) → bright yellow (pick `seq_length`) |
| Number inside cell | Visit order (1-indexed) |
| Number colour | White for dark cells, near-black for bright cells |
| Subplot title | Sequence index (`#1`, `#2`, …) |
| Colorbar | Maps colour to visit order |

### Tunable Constants

All visual parameters are module-level constants at the top of `draw_sample.py`:

```python
CMAP_NAME       = "plasma"     # any matplotlib colormap
UNVISITED_COLOR = "#1a1a2e"
VISITED_ALPHA   = 0.92
LABEL_FONTSIZE  = 5.5
TITLE_FONTSIZE  = 5.0
SUPTITLE_SIZE   = 11
CELL_LINEWIDTH  = 0.4
SUBPLOT_PAD     = 0.35
FIG_DPI         = 150
```

### CLI

```bash
python draw_sample.py <strategy> [options]

Options:
  -g, --grid_size   Grid side length (default: 4)
  -l, --seq_length  Sequence length (default: 6)
  -n, --num_seq     Number of sequences (default: 100)
  -s, --save        Output file path (default: plt.show())
      --seed        RNG seed (default: None)
```

---

## 6. Utility: `calculate_entropy`

**File:** `calculate_entropy.py`

### Method

1. Generate `num_seq` sequences via the named strategy.
2. Build an N²×N² transition count matrix **T** where `T[i, j]` is incremented each time box `j` is selected immediately after box `i` (0-indexed internally). Each sequence of length `seq_length` contributes `seq_length - 1` transitions.
3. Normalise: `P = T / T.sum()` — this is the **joint distribution** P(i→j).
4. Compute Shannon entropy: `H = -Σ P(i,j) * log₂(P(i,j))` over all non-zero cells.
5. Report `H / H_max` where `H_max = log₂(N² × (N²-1))` is the entropy of a uniform distribution over all ordered pairs of distinct boxes.

### Signature

```python
calculate_entropy(
    grid_size     : int  = 4,
    seq_length    : int  = 6,
    num_seq       : int  = 100,
    strategy_name : str  = "random",
    seed          : int  = None,
    verbose       : bool = True,
) -> dict
```

### Return Value

| Key | Type | Description |
|-----|------|-------------|
| `entropy_bits` | float | Shannon entropy in bits |
| `entropy_nats` | float | Shannon entropy in nats |
| `h_max_bits` | float | log₂(N²×(N²−1)) — theoretical maximum |
| `h_normalized` | float | H / H_max ∈ [0, 1] |
| `n_transitions` | int | Total transitions observed |
| `transition_matrix` | ndarray (N²×N²) | Joint P(i→j) |
| `strategy_name` | str | Strategy name as passed |
| `grid_size` | int | Grid side length |
| `seq_length` | int | Sequence length |
| `num_seq` | int | Number of sequences generated |

### Interpreting H_normalized

| H/H_max | Interpretation |
|---------|---------------|
| ≈ 1.0 | Transitions nearly uniformly distributed — strategy is close to random in sequential behaviour |
| 0.7–0.9 | Moderate structure — transitions are constrained but diverse |
| 0.5–0.7 | Strong structure — many transitions concentrated on a small set of pairs |
| < 0.5 | Highly deterministic — very few distinct transitions occur |

### CLI

```bash
python calculate_entropy.py <strategy> [options]

Options:
  -g, --grid_size   Grid side length (default: 4)
  -l, --seq_length  Sequence length (default: 6)
  -n, --num_seq     Number of sequences (default: 100)
      --seed        RNG seed (default: None)
  -q, --quiet       Suppress printed summary
```

---

## 7. Design Decisions and Edge Cases

**Why 1-indexed boxes?** The 1-indexed labelling matches natural human notation (boxes 1–16 rather than 0–15) and the user's original specification. All internal arithmetic converts to 0-indexed `(row, col)` pairs only where needed.

**Why precompute distances?** For strategies like `center_out_radial` and `weighted_center_bias`, distances from the geometric centre are identical across all sequences. Computing them once and reusing avoids redundant work across potentially thousands of sequences.

**Fallback design:** The dead-end fallback (random pick, rule restarts) is uniform across all strategies. This ensures `seq_length` is always exactly satisfied regardless of grid size or rule stringency. It also means even highly constrained strategies (e.g. knight's move on a small grid) never fail.

**Tie-breaking:** All tie-breaking sorts candidates before sampling to ensure that given the same RNG state, the same choice is always made. This makes behaviour reproducible with a fixed seed even when the internal candidate ordering might otherwise be non-deterministic (e.g. set iteration order in Python).

**`seq_length > N²/2` with checkerboard:** If you request more picks than there are boxes of one colour, the strategy silently falls back to the opposite colour. For `seq_length ≤ N²/2` this never happens on a standard even-N grid.

**Spiral for non-interior starts:** When the first pick is at a corner, the `_choose_start_dir` function falls through L/U checks (both may be out of bounds) to R or D. This is the only case where a spiral starts in a non-standard direction.

**Hilbert curve for non-power-of-2 grids:** The standard `d2xy` algorithm requires N to be a power of 2. For arbitrary N (e.g. N=5), the next power of 2 (8) is used, and any `(x, y)` outside the N×N region is simply skipped. The resulting path still visits all N² boxes exactly once and preserves approximate spatial locality.

**Perimeter crawl skip logic:** When the crawler encounters an already-visited perimeter box (only possible after a fallback places a box on the perimeter), it advances the index without adding a pick. A safety check detects if the entire remaining perimeter is visited and switches to a pure random fallback to avoid an infinite loop.

---

## 8. Extending the Library

### Adding a Strategy

```python
# strategy_myname.py

from __future__ import annotations
from typing import List
from grid_utils import all_boxes, get_neighbors, pick_and_remove, random_choice

def generate_sequences(
    grid_size: int = 4,
    seq_length: int = 6,
    n_seq: int = 100,
) -> List[List[int]]:
    N = grid_size
    sequences = []

    for _ in range(n_seq):
        available = set(all_boxes(N))
        seq = []

        # First pick: always random
        current = pick_and_remove(available)
        seq.append(current)

        while len(seq) < seq_length:
            # Apply your rule here
            # ...
            # Fallback: if no rule-valid candidates, pick randomly
            current = pick_and_remove(available)
            seq.append(current)

        sequences.append(seq)

    return sequences
```

The strategy is automatically discovered by `drawSample` and `calculate_entropy` — pass `strategy_name="myname"` and the utilities will import `strategy_myname`.

### Adding a Visualisation Mode to `drawSample`

The `_draw_sequence` function in `draw_sample.py` receives the full sequence as a list and the axes object. To add arrows showing transition order, for example:

```python
# Inside _draw_sequence, after drawing cells:
for (a, b) in zip(seq[:-1], seq[1:]):
    r1, c1 = get_row_col(a, N);  x1, y1 = c1 + 0.5, (N-1-r1) + 0.5
    r2, c2 = get_row_col(b, N);  x2, y2 = c2 + 0.5, (N-1-r2) + 0.5
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="white", lw=0.5))
```

### Computing Additional Entropy Metrics

The `transition_matrix` in the return dict of `calculate_entropy` is the full joint P(i→j). Row-normalising gives conditional distributions P(j|i) from which per-row (conditional) entropies can be computed:

```python
result = calculate_entropy(...)
P = result["transition_matrix"]    # shape (N², N²)

# Conditional entropy H(next | current)
row_sums = P.sum(axis=1, keepdims=True)
with np.errstate(divide='ignore', invalid='ignore'):
    P_cond = np.where(row_sums > 0, P / row_sums, 0)
nonzero = P_cond > 0
H_cond = -np.sum(P_cond[nonzero] * np.log2(P_cond[nonzero]))

# Marginal entropy H(box selected)
P_marginal = P.sum(axis=0)         # sum over "from" axis
H_marginal = -np.sum(P_marginal[P_marginal > 0] * np.log2(P_marginal[P_marginal > 0]))
```
