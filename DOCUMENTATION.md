# Grid Selection Strategies — Detailed Documentation

## Table of Contents

1. [Concepts and Conventions](#1-concepts-and-conventions)
2. [Shared Infrastructure](#2-shared-infrastructure)
   - 2.1 [grid_utils.py](#21-grid_utilspy)
   - 2.2 [image_utils.py](#22-image_utilspy)
3. [Universal Rules](#3-universal-rules)
4. [Geometric Strategy Reference](#4-geometric-strategy-reference)
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
5. [Image-Guided Strategy Reference](#5-image-guided-strategy-reference)
   - 5.1 [Image pipeline](#51-image-pipeline)
   - 5.2 [Image Salience](#52-image-salience)
   - 5.3 [Image Contrast](#53-image-contrast)
   - 5.4 [Image Colour Concentration](#54-image-colour-concentration)
   - 5.5 [Image Texture](#55-image-texture)
6. [Utility: drawSample](#6-utility-drawsample)
7. [Utility: calculate_entropy](#7-utility-calculate_entropy)
8. [Utility: compare_strategies_entropy](#8-utility-compare_strategies_entropy)
9. [Utility: compare_image_strategies_entropy](#9-utility-compare_image_strategies_entropy)
10. [Design Decisions and Edge Cases](#10-design-decisions-and-edge-cases)
11. [Extending the Library](#11-extending-the-library)

---

## 1. Concepts and Conventions

### The Grid

An N×N grid of boxes labelled 1…N² in row-major (left-to-right, top-to-bottom) order:

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

Neighbours are the four **cardinal** grid cells — up, down, left, right — that exist within bounds. Corner boxes have 2 neighbours; edge boxes have 3; interior boxes have 4.

```
Neighbours of box 6 in a 4×4 grid:  2 (up), 10 (down), 5 (left), 7 (right)
```

### Geometric Centre

The **geometric centre** of an N×N grid is the point `((N-1)/2, (N-1)/2)` in 0-indexed `(row, col)` space. For a 4×4 grid this is `(1.5, 1.5)` — equidistant between boxes 6, 7, 10, 11. Distances are Euclidean between box centres and this point.

### Sequences

A **sequence** is an ordered list of `seq_length` boxes, selected **without replacement** from the N²-box pool. Boxes within a single sequence are always distinct. Repetition across sequences in the same batch is normal.

---

## 2. Shared Infrastructure

### 2.1 `utils/grid_utils.py`

All geometric strategy modules import from `grid_utils`. The module exposes:

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
| `best_from` | `(candidates, key, available) → int` | Candidate with maximum `key`; ties broken randomly |
| `worst_from` | `(candidates, key, available) → int` | Candidate with minimum `key`; ties broken randomly |

### 2.2 `utils/image_utils.py`

All image-guided strategy modules import from `image_utils`. The module exposes:

| Function | Signature | Description |
|----------|-----------|-------------|
| `load_image` | `(img_index) → ndarray (H,W,3) uint8` | Load `images/img_<n>.png` as RGB |
| `resize_to_multiple` | `(img_rgb, N) → ndarray` | Resize to nearest multiple of N (bilinear) |
| `to_grayscale` | `(img_rgb) → ndarray (P,P) float64` | BT.601 luminance in [0,1] |
| `to_float` | `(img_rgb) → ndarray (P,P,3) float64` | RGB in [0,1] |
| `add_noise` | `(metric, metric_sd, A) → ndarray` | Add symmetric uniform noise ∈ [−SD·A, +SD·A] |
| `grid_weights` | `(noisy_metric, N) → ndarray (N²,)` | Average into N×N cells, normalise to [0.1, 0.9], flatten |

**Image naming convention:** `images/img_1.png`, `images/img_2.png`, …  Images are assumed square; non-square images will be treated as if square using the height dimension.

---

## 3. Universal Rules

These rules apply **across all strategies**, geometric and image-guided:

**First pick** — Every sequence begins with a uniformly random pick from all N² boxes. No strategy pre-constrains the starting position.

**Without replacement** — Once selected, a box is removed from the pool and cannot be re-selected within the same sequence.

**Dead-end fallback** — If the strategy rule cannot produce a valid next pick (all neighbours visited, no knight moves remaining, colour pool empty, no outward neighbour exists), a uniformly random pick is made from the remaining available boxes, and the rule is reapplied from that new position.

**Tie-breaking** — Whenever multiple candidates are equally valid, one is chosen uniformly at random. Candidate lists are sorted before sampling for reproducibility with a fixed seed.

---

## 4. Geometric Strategy Reference

### 4.1 Random

**File:** `strategies/strategy_random.py`

The unconditional baseline. Each sequence is a uniform random sample of `seq_length` boxes drawn via `random.sample`. No structural rule is applied — all picks are drawn simultaneously from a shuffle, so the "first pick is random" rule is trivially satisfied.

**Entropy:** Near-maximum (~0.99 H/H_max). All transitions roughly equally probable. z_entropy is close to zero (transitions are no more structured than a random pairing of the same labels).

---

### 4.2 Row-wise

**File:** `strategies/strategy_rowwise.py`

**Rule:** After the first random pick, each subsequent pick is chosen uniformly at random from the **unvisited boxes in the same row** as the current box. When that row is exhausted, a random fallback picks any remaining box, and the rule restarts from that new box's row.

**Key behaviour:** Within a row segment, order is random (not sequential). The sequence partitions into row-runs of varying length, separated by random jumps across rows.

**Example (4×4, seq_length=6):**
```
Pick 1 (random): 3   → row 0
Pick 2 (row 0):  1   → row 0
Pick 3 (row 0):  4   → row 0
Pick 4 (row 0):  2   → row 0  [row exhausted]
Pick 5 (random): 7   → row 1
Pick 6 (row 1):  6
```

---

### 4.3 Row-wise Sequential

**File:** `strategies/strategy_rowwise_sequential.py`

**Rule:** The canonical order `1, 2, 3, …, N²` (left-to-right, top-to-bottom) is established once. A random starting index is chosen uniformly in `[0, N²)`. The sequence takes `seq_length` consecutive entries from this order, wrapping modularly.

**Example (4×4, seq_length=6, start=13):**
```
13 → 14 → 15 → 16 → 1 → 2
```

**Entropy:** ~0.51 H/H_max. Only N² distinct transitions are possible (each box has exactly one successor), severely limiting entropy.

---

### 4.4 Center-Out Radial

**File:** `strategies/strategy_center_out_radial.py`

**Rule:** After the first random pick, at each step inspect the **grid neighbours** of the current box. Among those that are unvisited and strictly farther from the geometric centre than the current box, pick the one with maximum distance. If none exists, fall back to a random pick from all remaining boxes and restart.

**Distance:** Euclidean from box centre to `((N-1)/2, (N-1)/2)`, precomputed once per `generate_sequences` call.

**Key behaviour:** The sequence moves outward from wherever it starts. Interior starts trigger more frequent fallbacks because outward neighbours are quickly exhausted.

---

### 4.5 Center-Out Spiral

**File:** `strategies/strategy_center_out_spiral.py`

**Rule:** A clockwise outward spiral is precomputed from the first (random) pick using the **fixed-leg expansion scheme**: directions cycle L→U→R→D with leg lengths 1, 1, 2, 2, 3, 3, 4, 4, … The starting direction is chosen randomly from {L, U} if both are in bounds; otherwise whichever is available; otherwise R or D for corners. Position always advances in bounds; out-of-bounds steps freeze position for that step. The spiral covers all N² boxes exactly once.

**Verified examples (4×4):**

| Start | First 6 boxes |
|-------|--------------|
| 6 | 6 → 5 → 1 → 2 → 3 → 7 |
| 10 | 10 → 9 → 5 → 6 → 7 → 11 |
| 13 | 13 → 9 → 10 → 14 → [fallback] → … |

---

### 4.6 Neighbor First

**File:** `strategies/strategy_neighbor_first.py`

**Rule:** Each pick must be an unvisited cardinal neighbour of the current box. If no unvisited neighbours remain, fall back to a random pick from all remaining boxes and restart. This is a strict structural rule — not probabilistic — so whenever a valid neighbour exists the next pick *must* be one.

---

### 4.7 Nearest First

**File:** `strategies/strategy_nearest_first.py`

**Rule:** After the first random pick, each pick is the globally nearest unvisited box by Euclidean distance from the current box's centre. Ties broken randomly. No fallback is needed: there is always an unvisited box while the sequence is incomplete.

**Key behaviour:** Produces spatially compact clusters radiating outward from the start.

---

### 4.8 Farthest First

**File:** `strategies/strategy_farthest_first.py`

**Rule:** After the first random pick, each pick is the globally farthest unvisited box by Euclidean distance from the current box. Ties broken randomly.

**Key behaviour:** Sequence jumps maximally across the grid at every step. Tends to alternate between opposite corners and edges.

---

### 4.9 Checkerboard

**File:** `strategies/strategy_checkerboard.py`

**Colour convention:** Box b is **white** if `(row + col) % 2 == 0`, **black** otherwise (0-indexed). For even N, exactly N²/2 boxes of each colour exist.

**Rule:** The first pick's colour determines the sequence colour. All subsequent picks are drawn uniformly at random from remaining unvisited boxes of the **same colour**. If that colour pool is exhausted before `seq_length` boxes are collected, the fallback draws from the opposite colour.

**Key behaviour:** No two consecutive picks can be cardinal neighbours (same-colour boxes are never adjacent), creating a distinctive non-local transition pattern.

---

### 4.10 Knight's Move

**File:** `strategies/strategy_knights_move.py`

**Rule:** Each pick must be reachable from the current box by a single chess knight move (2+1 squares in any axis orientation). Pick uniformly at random from all valid unvisited knight destinations. If none remain, fall back to a random pick and restart.

**Note:** On a 4×4 board every box has at least 2 valid knight destinations on an empty board. Dead ends arise only after several boxes have been visited.

---

### 4.11 Snake

**File:** `strategies/strategy_snake.py`

**Rule:** A sweep orientation — R (rows, first row left→right), L (rows, first row right→left), D (columns, first column top→bottom), or U (columns, first column bottom→top) — is chosen uniformly at random per sequence. Within each band, direction alternates (boustrophedon). A random starting index within the full N²-length snake order is chosen, and the sequence takes `seq_length` consecutive entries with modular wrap.

**Example (R-orientation, 4×4):**
```
 1  2  3  4
 8  7  6  5
 9 10 11 12
16 15 14 13
```

---

### 4.12 Perimeter Crawl

**File:** `strategies/strategy_perimeter_crawl.py`

**Perimeter:** The 4(N−1) outermost boxes in clockwise order. For a 4×4 grid: `1, 2, 3, 4, 8, 12, 16, 15, 14, 13, 9, 5`.

**Rule:** If the first pick is on the perimeter, crawl CW or CCW (chosen randomly). If the first pick is interior, first pick the nearest perimeter box (Euclidean, ties broken randomly), then begin the crawl. Already-visited perimeter boxes are skipped. If the entire perimeter is exhausted, remaining picks are drawn randomly.

---

### 4.13 Islands

**File:** `strategies/strategy_islands.py`

**Structure:** A sequence of `seq_length` boxes is built as `seq_length // 3` islands of exactly 3 boxes each. Any remainder is filled as a partial island.

**Island construction:**
1. Pick a seed uniformly at random from all remaining unvisited boxes.
2. Collect the seed's unvisited cardinal neighbours.
3. Pick 2 neighbours uniformly at random (without replacement). If fewer than 2 are available, fill with random picks from the remaining pool.

**Examples:**
```
Island 1: seed=1, neighbours 2 and 5  → [1, 2, 5]
Island 2: seed=7, neighbours 8 and 11 → [7, 8, 11]
Full sequence: [1, 2, 5, 7, 8, 11]
```

---

### 4.14 Random Walk

**File:** `strategies/strategy_random_walk.py`

**Rule:** After the first random pick, each step moves to a uniformly random **unvisited cardinal neighbour** of the current box — a self-avoiding random walk on the grid graph. If the walk is trapped, teleport to a uniformly random unvisited box and resume.

Structurally equivalent to Neighbor First; the conceptual framing emphasises continuous movement rather than constraint.

---

### 4.15 Diagonal Sweep

**File:** `strategies/strategy_diagonal_sweep.py`

**Diagonal convention:** Anti-diagonals defined by constant `k = row + col` (0-indexed). For a 4×4 grid, k ranges from 0 (box 1 only) to 6 (box 16 only). Within each diagonal, boxes are ordered top-to-bottom.

**Rule:** The first pick's diagonal k₀ is the starting diagonal. Diagonals are swept in order k₀, k₀+1, … wrapping modularly. Within each diagonal, available unvisited boxes are shuffled and added to the sequence until `seq_length` is reached. Exhausted diagonals are skipped.

---

### 4.16 Weighted Center Bias

**File:** `strategies/strategy_weighted_center_bias.py`

**Rule:** All N² boxes are assigned fixed weights before any pick:

```
w(b) = 1 / dist(b, centre) ^ WEIGHT_POWER
```

where `dist` is Euclidean distance to the geometric centre and `WEIGHT_POWER = 1.0` (exposed module-level constant). Boxes at the exact centre receive a large finite weight (10× the maximum finite weight). Picks are drawn via weighted sampling (`random.choices`) over the remaining available pool using the **pre-computed fixed weights** — weights are not renormalised after each pick.

**Tuning:** Increasing `WEIGHT_POWER` sharpens the centre bias. At very high values the strategy approaches deterministic centre-first ordering.

---

### 4.17 Hilbert Curve

**File:** `strategies/strategy_hilbert_curve.py`

**Rule:** The Hilbert curve traversal order for the N×N grid is precomputed using the standard `d2xy` algorithm (converts Hilbert distance d to (x,y) coordinates). For non-power-of-2 N, the next-larger power-of-2 grid is used and out-of-bounds cells are filtered. The 4×4 Hilbert order is:

```
1, 2, 6, 5, 9, 13, 14, 10, 11, 15, 16, 12, 8, 7, 3, 4
```

A random starting index within this order is chosen, and the sequence takes `seq_length` consecutive entries with modular wrap.

**Key property:** Spatial locality is preserved — boxes close together in the 1D Hilbert order tend to be close in 2D space.

**Entropy:** ~0.51 H/H_max — tied with Row-wise Sequential and for the same reason: both are fixed-offset reads of a deterministic path, so only N² distinct transitions are possible.

---

## 5. Image-Guided Strategy Reference

### 5.1 Image Pipeline

All four image strategies follow an identical flow, split between a one-time setup phase and a per-sequence phase.

**Before the sequence loop (once per `generate_sequences` call):**

1. `load_image(img_index)` — load `images/img_<n>.png` as (H,W,3) uint8 RGB.
2. `resize_to_multiple(img_rgb, N)` — resize to P×P where P = round(H/N)·N, using bilinear interpolation.
3. `_compute_<metric>(img_rgb)` — compute the metric for every pixel → (P,P) float64 array.
4. Compute `metric_sd = metric.std()` (used to scale noise).

**Inside the loop (once per sequence):**

5. `add_noise(metric, metric_sd, A)` — draw symmetric uniform noise ∈ [−SD·A, +SD·A] fresh for this sequence and add to the metric map.
6. `grid_weights(noisy_metric, N)` — divide into N×N cells of (P/N × P/N) pixels, average each cell, normalise the N×N result to [0.1, 0.9] via `b = 0.1 + 0.8·(a−min)/(max−min)`, flatten row-wise to a length-N² weight vector.
7. Draw `seq_length` boxes via `random.choices(available, weights=pool_weights)` without replacement, using the weight vector. Weights are **not renormalised** after each pick.

**Normalisation formula:** `b = 0.1 + 0.8 · (a − min(a)) / (max(a) − min(a))` maps any cell value to [0.1, 0.9]. The floor of 0.1 ensures no cell is ever unselectable. If all cells are identical (flat metric), all weights are set to 0.5.

**Noise constant `A`:** Exposed as a module-level constant (`NOISE_A = 0.3` by default) at the top of each strategy file. Setting A=0 disables noise entirely, making weights identical across all sequences for a given image.

---

### 5.2 Image Salience

**File:** `strategies/strategy_image_salience.py`

**Metric:** Difference-of-Gaussians (DoG) applied to the luminance channel:

```
saliency(x,y) = G(x,y; σ_centre) − G(x,y; σ_surround)
```

Centre and surround standard deviations are `SIGMA_CENTER = 2.0` and `SIGMA_CENTER × SURROUND_RATIO = 8.0` (both exposed constants). Positive values indicate bright-on-dark regions; negative values indicate dark-on-bright. Both are preserved through normalisation, so both types of centre-surround contrast are treated as salient.

**TODO:** Replace with a full Itti-Koch saliency implementation including orientation-selective Gabor channels, colour opponency (R-G, B-Y), and the iterative normalisation operator N(·) from Itti, Koch & Niebur (1998).

**Exposed constants:**
```python
SIGMA_CENTER   = 2.0    # centre Gaussian σ in pixels
SURROUND_RATIO = 4.0    # σ_surround = SURROUND_RATIO × SIGMA_CENTER
NOISE_A        = 0.3
```

---

### 5.3 Image Contrast

**File:** `strategies/strategy_image_contrast.py`

**Metric:** Local RMS contrast (local standard deviation) on the luminance channel, computed via box-filter variance:

```
var(x,y) = E[I²](x,y) − E[I]²(x,y)
rms(x,y) = sqrt(max(var, 0))
```

where E[·] denotes a uniform (box) filter with window size K = `max(3, floor(P/10))` forced odd. This scales with image resolution: a 400-pixel image uses a 40-pixel window; a 200-pixel image uses a 21-pixel window.

**Exposed constant:**
```python
NOISE_A = 0.3
```

---

### 5.4 Image Colour Concentration

**File:** `strategies/strategy_image_color_concentration.py`

**Metric:** Per-pixel HSV saturation derived from RGB:

```
S(x,y) = (max(R,G,B) − min(R,G,B)) / max(R,G,B)
S = 0  when max(R,G,B) = 0  (pure black)
```

Computed directly from the float RGB array; no HSV conversion library is used. Achromatic pixels (R=G=B) have S=0 and map to the floor weight of 0.1 after normalisation — they remain selectable but are least likely.

**Exposed constant:**
```python
NOISE_A = 0.3
```

---

### 5.5 Image Texture

**File:** `strategies/strategy_image_texture.py`

**Metric:** Local variance on the luminance channel, using the same box-filter formulation as the contrast strategy:

```
texture(x,y) = max(E[I²] − E[I]², 0)
```

The raw variance (not its square root) is used, so high-frequency texture regions are emphasised more strongly than in the contrast strategy. Window size K = `max(3, floor(P/10))` forced odd.

**Exposed constant:**
```python
NOISE_A = 0.3
```

---

## 6. Utility: `drawSample`

**File:** `draw_sample.py`

### Signature

```python
drawSample(
    grid_size     : int           = 4,
    seq_length    : int           = 6,
    num_seq       : int           = 100,
    strategy_name : str           = "random",
    img_index     : Optional[int] = None,
    save_path     : Optional[str] = None,
    seed          : Optional[int] = None,
) -> None
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `grid_size` | Side length N of the square grid |
| `seq_length` | Number of boxes per sequence |
| `num_seq` | Number of sequences = number of subplot panels |
| `strategy_name` | Strategy name with or without `strategy_` prefix |
| `img_index` | 1-based image index (required for image strategies; raises `ValueError` if omitted) |
| `save_path` | Output file path; `None` calls `plt.show()` |
| `seed` | Optional RNG seed for reproducibility |

### Image Strategy Detection

`drawSample` calls `inspect.signature(mod.generate_sequences)` and checks for `img_index` in the parameter list. No manual registry is maintained — any strategy module that includes `img_index` in its `generate_sequences` signature is treated as image-based.

### Layout

Subplots are arranged in a near-square grid: `n_cols = ceil(sqrt(num_seq))`, `n_rows = ceil(num_seq / n_cols)`. Incomplete rows are padded with hidden axes.

### Visual Encoding

| Element | Encoding |
|---------|----------|
| Unvisited cell | Dark navy `#1a1a2e` |
| Visited cell | `plasma` colormap: deep purple (pick 1) → bright yellow (pick `seq_length`) |
| Number inside cell | Visit order (1-indexed) |
| Colorbar | Right side; maps colour to visit order |
| Subplot title | Sequence index (#1, #2, …) |
| Figure title | Strategy name, img index if applicable, grid parameters |

### Tunable Constants

```python
CMAP_NAME       = "plasma"    # any matplotlib colormap name
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

  -g, --grid_size    Grid side length (default: 4)
  -l, --seq_length   Sequence length (default: 6)
  -n, --num_seq      Number of sequences (default: 100)
  -i, --img_index    Image index for image strategies (e.g. 1 → images/img_1.png)
  -s, --save         Output file path (default: plt.show())
      --seed         RNG seed (default: None)
```

---

## 7. Utility: `calculate_entropy`

**File:** `calculate_entropy.py`

### Method

**H (global transition entropy):**
1. Generate `num_seq` sequences via the named strategy.
2. Collect all consecutive (from, to) box pairs across all sequences. Cross-sequence transitions are excluded by construction (pairs are taken only within each sequence).
3. Encode each pair as a single integer key; count unique keys.
4. Compute `P(i→j) = count(i→j) / total_transitions`.
5. Compute `H = −Σ P log₂(P)` over all non-zero entries.
6. Report `H / H_max` where `H_max = log₂(N² × (N²−1))`.

**z_entropy (shuffle null):**
1. For each of `N_SHUFFLES = 200` iterations: randomly permute the `from` array while holding `to` fixed (mirrors the MATLAB `compute_zscored_global_entropy.m` implementation). This breaks real sequential structure while preserving each label's marginal frequency.
2. Compute H on each shuffled transition set → distribution `H_shuffle`.
3. `z_entropy = (H − mean(H_shuffle)) / std(H_shuffle, ddof=1)`.

Positive z: transitions more spread out than a random pairing of the same labels. Negative z: transitions more concentrated/predictable than chance.

### Signature

```python
calculate_entropy(
    grid_size     : int           = 4,
    seq_length    : int           = 6,
    num_seq       : int           = 100,
    strategy_name : str           = "random",
    img_index     : Optional[int] = None,
    seed          : Optional[int] = None,
    verbose       : bool          = True,
) -> dict
```

### Return Value

| Key | Type | Description |
|-----|------|-------------|
| `entropy_bits` | float | H in bits |
| `entropy_nats` | float | H in nats |
| `h_max_bits` | float | log₂(N²×(N²−1)) — theoretical maximum |
| `h_normalized` | float | H / H_max ∈ [0, 1] |
| `z_entropy` | float | Z-scored H against shuffle null |
| `h_shuffle_mean` | float | Mean H across shuffles (bits) |
| `h_shuffle_std` | float | Std H across shuffles (bits) |
| `h_shuffle_all` | ndarray (N_SHUFFLES,) | All individual shuffle entropies |
| `n_transitions` | int | Total transitions observed |
| `transition_matrix` | ndarray (N²,N²) | Joint P(i→j) |
| `strategy_name` | str | Strategy name as passed |
| `grid_size` | int | Grid side length |
| `seq_length` | int | Sequence length |
| `num_seq` | int | Number of sequences generated |

### Interpreting the Metrics

| H/H_max | Interpretation |
|---------|---------------|
| ≈ 1.0 | Transitions nearly uniformly distributed; strategy behaves close to random |
| 0.7–0.9 | Moderate structure; transitions constrained but diverse |
| 0.5–0.7 | Strong structure; many transitions concentrated on a small set of pairs |
| < 0.5 | Highly deterministic; very few distinct transitions occur |

| z_entropy | Interpretation |
|-----------|---------------|
| Near 0 | Transition structure is no greater than expected by random pairing |
| Strongly negative | Transitions far more concentrated than a random pairing of the same labels |
| Positive | Transitions more spread out than random pairing (unusual; suggests anti-structure) |

### Tunable Constant

```python
N_SHUFFLES : int = 200    # shuffle iterations for z_entropy null distribution
```

### CLI

```bash
python calculate_entropy.py <strategy> [options]

  -g, --grid_size    Grid side length (default: 4)
  -l, --seq_length   Sequence length (default: 6)
  -n, --num_seq      Number of sequences (default: 100)
  -i, --img_index    Image index for image strategies
      --seed         RNG seed (default: None)
  -q, --quiet        Suppress printed summary
```

---

## 8. Utility: `compare_strategies_entropy`

**File:** `compare_strategies_entropy.py`

Runs all 17 geometric strategies and produces a scatter plot of H/H_max (x-axis) vs z_entropy (y-axis). Points are colour-coded by strategy category. Quadrant annotations describe the interpretation of each region.

```bash
python compare_strategies_entropy.py -g 4 -l 6 -n 500 -s entropy_scatter.png
```

---

## 9. Utility: `compare_image_strategies_entropy`

**File:** `compare_image_strategies_entropy.py`

Runs all 4 image strategies across every image found in `images/img_*.png`, plus Random as a plotted baseline and Hilbert Curve as a title-only reference. Produces a scatter plot of H/H_max vs z_entropy.

**Visual encoding:**
- **Colour** → image index (up to 12 distinct colours; cycles for larger collections)
- **Marker shape** → strategy (○ Salience, □ Contrast, △ Colour Concentration, ◇ Texture)
- **Random baseline** → large star (★), labelled directly on the plot
- **Hilbert Curve** → not plotted; H/H_max and z_entropy shown in the figure title for reference

Image discovery is automatic — any `img_*.png` file in the `images/` folder is included without code changes.

```bash
python compare_image_strategies_entropy.py -g 4 -l 6 -n 500 -s entropy_scatter_image.png
```

---

## 10. Design Decisions and Edge Cases

**Why 1-indexed boxes?** Matches natural human notation and the original specification. Internal arithmetic converts to 0-indexed `(row, col)` only where needed.

**Why precompute distances?** For `center_out_radial` and `weighted_center_bias`, distances from the geometric centre are identical across all sequences. Computing once avoids redundant work over thousands of sequences.

**Fallback design:** The uniform dead-end fallback guarantees `seq_length` is always exactly satisfied, regardless of how stringent the strategy rule is or how small the grid. Strategies never raise exceptions due to exhausted move sets.

**Tie-breaking reproducibility:** All tie-breaking sorts candidates before sampling so that given the same RNG state, the same choice is always made, even when internal data structures (e.g. sets) have non-deterministic iteration order in Python.

**Checkerboard and `seq_length > N²/2`:** If more picks are requested than there are boxes of one colour, the strategy silently falls back to the opposite colour for the remainder. For `seq_length ≤ N²/2` on even-N grids this never occurs.

**Spiral for corner starts:** `_choose_start_dir` checks L then U; for corners where both are out of bounds it falls through to R then D.

**Hilbert curve for non-power-of-2 grids:** `d2xy` requires power-of-2 N. For arbitrary N, the next-larger power is used and any cell outside the N×N region is filtered. The resulting path visits all N² boxes exactly once.

**Perimeter crawl skip logic:** When the crawler encounters an already-visited perimeter box (possible only after a fallback lands on the perimeter), the index advances without adding a pick. A safety check detects full perimeter exhaustion and switches to random fallback to avoid an infinite loop.

**Image resize rounding:** `P = round(H/N) * N`. For H=201, N=4, this gives P=200 rather than P=204, choosing the nearest multiple rather than always rounding up. This minimises information loss from resizing.

**Noise timing:** Noise is added to the full P×P pixel-level metric map (before cell averaging), not to the N×N cell-average map. This means pixel-level spatial variation within each cell contributes to the noisy average in a realistic way, rather than just shifting the already-averaged cell values.

**Noise does not overflow normalisation bounds:** The noise is added before normalisation. The normalisation formula always maps the noisy map's actual min and max to 0.1 and 0.9, so noise amplitude does not cause out-of-range weights regardless of A.

**z_entropy asymmetry:** The shuffle permutes `from` while holding `to` fixed, following the original MATLAB implementation. This asymmetric shuffle tests whether the *source* of each transition is more structured than chance, not the destination.

---

## 11. Extending the Library

### Adding a Geometric Strategy

```python
# strategies/strategy_myname.py

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
            # Apply your rule here ...
            # Fallback: if no rule-valid candidates, pick randomly
            current = pick_and_remove(available)
            seq.append(current)

        sequences.append(seq)

    return sequences
```

Pass `strategy_name="myname"` to `drawSample` and `calculate_entropy` — both utilities import `strategy_myname` automatically.

### Adding an Image-Guided Strategy

```python
# strategies/strategy_image_mymetric.py

from __future__ import annotations
import random
from typing import List, Optional
import numpy as np
from image_utils import load_image, resize_to_multiple, add_noise, grid_weights
from grid_utils import all_boxes

NOISE_A: float = 0.3    # ← expose prominently

def _compute_mymetric(img_rgb: np.ndarray) -> np.ndarray:
    """Return a (P, P) float64 metric map."""
    ...

def generate_sequences(
    grid_size: int = 4,
    seq_length: int = 6,
    n_seq: int = 100,
    img_index: int = 1,          # ← presence of this triggers image detection
    seed: Optional[int] = None,
) -> List[List[int]]:
    N = grid_size
    img_rgb   = load_image(img_index)
    img_rgb   = resize_to_multiple(img_rgb, N)
    metric    = _compute_mymetric(img_rgb)
    metric_sd = float(metric.std())

    boxes = all_boxes(N)
    sequences = []

    for _ in range(n_seq):
        noisy   = add_noise(metric, metric_sd, NOISE_A)
        weights = grid_weights(noisy, N)

        available = list(boxes)
        seq = []
        while len(seq) < seq_length:
            pool_w = [weights[b - 1] for b in available]
            chosen = random.choices(available, weights=pool_w, k=1)[0]
            available.remove(chosen)
            seq.append(chosen)

        sequences.append(seq)

    return sequences
```

Both `drawSample` and `calculate_entropy` automatically detect `img_index` in the signature and will require it to be supplied at call time.

### Computing Additional Entropy Metrics from the Transition Matrix

```python
result = calculate_entropy(strategy_name="neighbor_first",
                           grid_size=4, seq_length=6, num_seq=500)
P = result["transition_matrix"]    # shape (N², N²), joint P(i→j)

# Conditional entropy H(next | current)
row_sums = P.sum(axis=1, keepdims=True)
with np.errstate(divide='ignore', invalid='ignore'):
    P_cond = np.where(row_sums > 0, P / row_sums, 0.0)
nz = P_cond > 0
H_conditional = float(-np.sum(P_cond[nz] * np.log2(P_cond[nz])))

# Marginal entropy H(box selected)
P_marginal = P.sum(axis=0)
nz = P_marginal > 0
H_marginal = float(-np.sum(P_marginal[nz] * np.log2(P_marginal[nz])))

# Full null distribution for custom significance testing
h_null = result["h_shuffle_all"]   # shape (N_SHUFFLES,)
p_value = (h_null <= result["entropy_bits"]).mean()
```
