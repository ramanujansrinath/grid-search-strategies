# Grid Selection Strategies

A Python library for generating and analysing structured spatial selection sequences on an N×N grid. Each strategy defines a different rule for choosing which cells to visit, without replacement, producing sequences that range from fully random to highly deterministic paths. The library includes image-guided strategies driven by visual saliency, contrast, colour, and texture maps, along with visualisation and entropy analysis tools.

---

## Motivation

When studying spatial sampling, visual attention, or search behaviour on grids, the *order* in which cells are visited carries as much information as *which* cells are visited. This library provides a controlled, extensible set of strategies — from random baselines to structured walks like spirals, knight's moves, and Hilbert curves, to image-guided sampling weighted by pixel-level metrics — along with tools to visualise and quantitatively compare them via transition entropy and z-scored entropy.

---

## Installation

No package installation required. Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/grid-selection-strategies.git
cd grid-selection-strategies
pip install matplotlib numpy scipy Pillow
```

Python 3.8+ is supported.

---

## Quick Start

```python
from draw_sample import drawSample
from calculate_entropy import calculate_entropy

# Visualise 100 sequences from a geometric strategy
drawSample(grid_size=4, seq_length=6, num_seq=100,
           strategy_name="neighbor_first",
           save_path="neighbor_first.png", seed=42)

# Visualise 100 sequences from an image-guided strategy
drawSample(grid_size=4, seq_length=6, num_seq=100,
           strategy_name="image_salience",
           img_index=1,                      # loads images/img_1.png
           save_path="salience.png", seed=42)

# Compute global transition entropy + z-scored entropy
result = calculate_entropy(grid_size=4, seq_length=6, num_seq=500,
                           strategy_name="knights_move", seed=42)
print(f"H/H_max = {result['h_normalized']:.4f}   "
      f"z = {result['z_entropy']:+.2f}")

# Same for an image-based strategy
result = calculate_entropy(grid_size=4, seq_length=6, num_seq=500,
                           strategy_name="image_texture", img_index=1)
print(f"H/H_max = {result['h_normalized']:.4f}   "
      f"z = {result['z_entropy']:+.2f}")
```

---

## Strategies

### Geometric Strategies

All geometric strategies share this signature:

```python
generate_sequences(grid_size=4, seq_length=6, n_seq=100) -> List[List[int]]
```

Boxes are labelled 1…N² row-wise. Neighbours are the four cardinal directions. Every strategy starts with a **uniformly random first pick**. Dead ends trigger a random fallback and the rule restarts. Ties are broken uniformly at random.

| # | Module | Strategy | Description |
|---|--------|----------|-------------|
| 1 | `strategy_random` | **Random** | Uniform sampling without replacement. Baseline. |
| 2 | `strategy_rowwise` | **Row-wise** | Stays within the current row; random fallback when the row is exhausted. |
| 3 | `strategy_rowwise_sequential` | **Row-wise Sequential** | Left-to-right canonical sweep from a random start, wrapping modularly. |
| 4 | `strategy_center_out_radial` | **Center-Out Radial** | Each step moves to the neighbour farthest from the geometric grid centre. |
| 5 | `strategy_center_out_spiral` | **Center-Out Spiral** | Clockwise outward spiral from a random start using fixed-leg expansion. |
| 6 | `strategy_neighbor_first` | **Neighbor First** | Each pick must be a cardinal neighbour of the previous pick. |
| 7 | `strategy_nearest_first` | **Nearest First** | Always picks the globally nearest unvisited box by Euclidean distance. |
| 8 | `strategy_farthest_first` | **Farthest First** | Always picks the globally farthest unvisited box. Maximises spatial spread. |
| 9 | `strategy_checkerboard` | **Checkerboard** | All picks share the colour parity of the first box; random within that colour. |
| 10 | `strategy_knights_move` | **Knight's Move** | Each step is a chess knight move (L-shaped) from the previous box. |
| 11 | `strategy_snake` | **Snake** | Boustrophedon sweep from a random start with a randomly chosen orientation. |
| 12 | `strategy_perimeter_crawl` | **Perimeter Crawl** | Crawls the outer ring CW or CCW; interior starts route to the nearest perimeter box first. |
| 13 | `strategy_islands` | **Islands** | Two clusters of three: each is a random seed plus two of its neighbours. |
| 14 | `strategy_random_walk` | **Random Walk** | Self-avoiding walk along grid edges; teleports when trapped. |
| 15 | `strategy_diagonal_sweep` | **Diagonal Sweep** | Sweeps anti-diagonals (constant row+col), picking randomly within each. |
| 16 | `strategy_weighted_center_bias` | **Weighted Center Bias** | Weighted sampling with w ∝ 1/distance from grid centre, fixed before the loop. |
| 17 | `strategy_hilbert_curve` | **Hilbert Curve** | Follows the space-filling Hilbert curve traversal order from a random offset. |

### Image-Guided Strategies

Image strategies extend the signature with an `img_index` parameter:

```python
generate_sequences(grid_size=4, seq_length=6, n_seq=100, img_index=1) -> List[List[int]]
```

Images are loaded from the `images/` subfolder, named `img_1.png`, `img_2.png`, etc. Each image must be square; it is resized to the nearest multiple of `grid_size` using bilinear interpolation.

For each sequence, a pixel-level metric map is computed from the image, divided into an N×N grid, and the per-cell average is normalised to [0.1, 0.9] to form selection weights. A small amount of symmetric uniform noise (scaled to `NOISE_A × SD` of the metric) is added fresh for each sequence, introducing controlled variability while preserving the overall spatial bias.

| # | Module | Strategy | Metric |
|---|--------|----------|--------|
| 18 | `strategy_image_salience` | **Image Salience** | Difference-of-Gaussians (DoG) centre-surround saliency on the luminance channel. |
| 19 | `strategy_image_contrast` | **Image Contrast** | Local RMS contrast (standard deviation within a ⌊W/10⌋-pixel window) on luminance. |
| 20 | `strategy_image_color_concentration` | **Image Colour Concentration** | Per-pixel HSV saturation from the RGB image. |
| 21 | `strategy_image_texture` | **Image Texture** | Local variance within a ⌊W/10⌋-pixel window on the luminance channel. |

All four strategies share utility functions from `image_utils.py`.

---

## Utilities

### `drawSample`

Generates sequences and renders them as a tiled grid figure. Unvisited cells appear in dark navy; visited cells are coloured from deep purple (first pick) to bright yellow (last pick) via the `plasma` colormap, with the visit-order number printed inside each cell.

```python
drawSample(
    grid_size     = 4,
    seq_length    = 6,
    num_seq       = 100,
    strategy_name = "center_out_spiral",
    img_index     = None,          # required for image-based strategies
    save_path     = "spiral.png",  # None → plt.show()
    seed          = 42,
)
```

Image-based strategies are detected automatically via signature inspection. Passing an image-based strategy name without `img_index` raises a `ValueError` with a descriptive message.

CLI:
```bash
python draw_sample.py center_out_spiral -g 4 -l 6 -n 100 -s spiral.png --seed 42
python draw_sample.py image_salience -i 1 -g 4 -l 6 -n 100 -s salience.png
```

### `calculate_entropy`

Computes two entropy measures for any strategy.

**H / H_max** — Shannon entropy of the joint transition distribution P(i→j), normalised by the theoretical maximum log₂(N²×(N²−1)).

**z_entropy** — The same H z-scored against a shuffle null: `from` labels are permuted while `to` labels are held fixed (N_SHUFFLES=200 iterations). Positive z means transitions are more spread out than chance; negative z means more concentrated/predictable.

```python
result = calculate_entropy(
    grid_size     = 4,
    seq_length    = 6,
    num_seq       = 500,
    strategy_name = "image_contrast",
    img_index     = 1,             # required for image-based strategies
    seed          = 42,
    verbose       = True,
)
# Keys: entropy_bits, entropy_nats, h_max_bits, h_normalized,
#       z_entropy, h_shuffle_mean, h_shuffle_std, h_shuffle_all,
#       n_transitions, transition_matrix
```

CLI:
```bash
python calculate_entropy.py image_contrast -i 1 -g 4 -l 6 -n 500 --seed 42
python calculate_entropy.py knights_move -g 4 -l 6 -n 500 -q
```

### `compare_strategies_entropy`

Runs all 17 geometric strategies and scatter-plots H/H_max (x) vs z_entropy (y), colour-coded by strategy category.

```bash
python compare_strategies_entropy.py -g 4 -l 6 -n 500 -s entropy_scatter.png
```

### `compare_image_strategies_entropy`

Runs all 4 image strategies across every image found in `images/`, plus Random as a plotted baseline and Hilbert Curve as a title reference. Points are colour-coded by image; marker shapes distinguish strategies.

```bash
python compare_image_strategies_entropy.py -g 4 -l 6 -n 500 -s entropy_scatter_image.png
```

---

## Entropy Benchmarks

Geometric strategies, ranked by z_entropy (4×4 grid, seq_length=6, 500 sequences):

| Strategy | H (bits) | H / H_max | z_entropy |
|----------|----------|-----------|-----------|
| Random | 7.84 | 0.99 | −16.3 |
| Weighted Center Bias | 7.65 | 0.97 | −15.5 |
| Islands | 7.24 | 0.92 | −102.4 |
| Checkerboard | 6.77 | 0.86 | −180.8 |
| Row-wise | 6.63 | 0.84 | −208.5 |
| Center-Out Radial | 6.19 | 0.78 | −223.3 |
| Diagonal Sweep | 5.71 | 0.72 | −343.9 |
| Knight's Move | 5.69 | 0.72 | −343.2 |
| Neighbor First | 5.65 | 0.72 | −373.5 |
| Random Walk | 5.65 | 0.72 | −373.5 |
| Nearest First | 5.63 | 0.71 | −375.8 |
| Snake | 5.63 | 0.71 | −377.5 |
| Center-Out Spiral | 5.50 | 0.70 | −340.4 |
| Farthest First | 4.89 | 0.62 | −322.8 |
| Perimeter Crawl | 4.76 | 0.60 | −494.3 |
| Row-wise Sequential | 4.00 | 0.51 | −629.0 |
| Hilbert Curve | 4.00 | 0.51 | −636.4 |

H_max = 7.907 bits (log₂(240) for a 4×4 grid).

---

## File Structure

```
grid-selection-strategies/
│
├── images/                               # input images for image strategies
│   ├── img_1.png
│   ├── img_2.png
│   └── …
│
├── grid_utils.py                         # shared grid primitives (all strategies)
├── image_utils.py                        # shared image loading, metric, noise, normalisation
│
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
│
├── strategy_image_salience.py
├── strategy_image_contrast.py
├── strategy_image_color_concentration.py
├── strategy_image_texture.py
│
├── draw_sample.py                        # visualisation utility
├── calculate_entropy.py                  # entropy analysis (H/H_max + z_entropy)
├── compare_strategies_entropy.py         # scatter plot: all geometric strategies
├── compare_image_strategies_entropy.py   # scatter plot: image strategies × images
│
├── test_strategies.py                    # smoke tests for all geometric strategies
├── DOCUMENTATION.md
└── README.md
```

---

## Running the Tests

```bash
python test_strategies.py
```

Verifies that every geometric strategy returns the correct number of sequences, each of the correct length, with valid and non-duplicate box numbers, on both 4×4 and 5×5 grids.

---

## Adding a New Strategy

**Geometric strategy:**

1. Create `strategy_myname.py`.
2. Implement `generate_sequences(grid_size, seq_length, n_seq) -> List[List[int]]`.
3. Import helpers from `grid_utils` as needed.
4. Pass `strategy_name="myname"` to `drawSample` and `calculate_entropy`.

**Image-guided strategy:**

1. Create `strategy_image_mymetric.py`.
2. Implement `generate_sequences(grid_size, seq_length, n_seq, img_index) -> List[List[int]]`.
3. Use `image_utils.load_image`, `resize_to_multiple`, `add_noise`, and `grid_weights`.
4. Implement your `_compute_mymetric(img_rgb)` function returning a (P×P) float64 array.
5. Both `drawSample` and `calculate_entropy` will automatically detect `img_index` in the signature and require it to be supplied.

---

## License

MIT License

Copyright (c) 2025 [NAME]

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

## Contact

**[NAME]**
[AFFILIATION]
[EMAIL]
