# penile-cancer

Three-stage WSI pipeline for the MDACC penile-cancer cohort (29 `.svs`, 40×):

1. **Trident** — tissue QC, 20× / 512 px patch coords, CONCH patch features, TITAN slide features.
2. **HistoPLUS** — CellViT cell segmentation on the tissue Trident kept.
3. **`pathomics/`** — 145 handcrafted, named per-slide features from the cell masks: nuclear morphology, immune spatial organisation (spaTIL v2), composition.

Only code and the 145-feature table are tracked; WSIs and intermediate outputs are git-ignored and live on gu.

## Layout

```
scripts/                       gu-side shell drivers for stages 1–2
histoplus_penile/              python package for stages 1–2
pathomics/                     python package for stage 3
cell_masks_to_geojson.py       histoplus json → QuPath GeoJSON
features/penile_features.*     stage-3 output (26 slides × 145 features)
```

## Requirements

Stages 1–2 (on gu): NVIDIA GPU, conda. A Hugging Face token with access to
`MahmoodLab/conchv1_5`, `MahmoodLab/TITAN`, GrandQC weights, and `owkin/histoplus`
exported in `~/.bashrc` as `HF_TOKEN`. The Trident repo cloned at `~/trident`.

Stage 3 (runs anywhere, no GPU):

```bash
pip install numpy pandas scipy shapely pyarrow matplotlib scikit-learn
```

## Stage 1 — Trident  (env `trident`, gu)

```bash
bash scripts/run_trident_penile.sh all 2>&1 | tee logs/trident_penile.log
```

`all` = stage WSIs → A (GrandQC seg + 20× / 512 px patch coords + CONCH v1.5 features) → B (TITAN slide features). Run a single stage with `stage | A | B`. Resumable.

Outputs under `trident_output/`:

| path | contents |
|---|---|
| `contours/<slide>.jpg`, `contours_geojson/<slide>.geojson` | tissue QC overlays |
| `20x_512px_0px_overlap/patches/<slide>_patches.h5` | patch coords (N × 2 level-0 px, top-left) |
| `…/features_conch_v15/<slide>.h5` | CONCH v1.5 patch features |
| `…/slide_features_titan/<slide>.h5` | TITAN slide features |

## Stage 2 — HistoPLUS  (env `histoplus`, gu)

```bash
bash scripts/run_histoplus_penile.sh all 2>&1 | tee logs/histoplus_penile.log
```

`all` = `cells` (CellViT 40× @ MPP 0.25 over Trident's tiles) → `diversity` (per-patch counts). Resumable.

Outputs under `histoplus_output/`:

| path | contents |
|---|---|
| `<slide>/cell_masks.json` | HistoPLUS cell masks (~100–300 MB / slide) |
| `patch_diversity/<slide>.csv` | one row per Trident patch, same order as `<slide>_patches.h5` (drop-in alongside CONCH) |
| `slide_diversity.csv` | one row per slide (richness, Shannon, Simpson, cells / mm²) |

To re-pull and re-run only slides that failed: `bash scripts/reprocess_failed.sh`.

Edge note: HistoPLUS internally re-grids any tiling to 656 px (`784 − 2·64`) by `floor(x / 656)·656` and dedups — it never *expands* a tile. So `trident_io.to_histoplus_coords` pre-expands each Trident patch to the set of 656 px tiles it overlaps and dedups, with `tile_size = 656`. `deepzoom_level = ceil(log2(max(W, H)))`; the right/bottom edge of the grid is clamped one cell short to avoid OpenSlide's `Invalid address`.

## Stage 3 — `pathomics/` handcrafted features

Bring the histoplus outputs over:

```bash
rsync -avh gu:/home/smedin7/Documents/penile/histoplus_output/ histoplus_output/
```

Extract:

```bash
python -m pathomics histoplus_output -o features/penile_features.parquet --min-confidence 0.5
```

First call on a slide streams its `cell_masks.json` (multi-GB possible), derives morphology, and writes a compact `cells.parquet` cache next to it. Subsequent loads: ~0.1 s. Feature extraction off the cache: 2 s @ 135 k cells → 75 s @ 1.3 M cells.

Or from Python:

```python
from pathomics import load_cells, extract_features, extract_features_roi, viz
import shapely

cells = load_cells("histoplus_output/MPe08B/cell_masks.json", min_confidence=0.5)
feats = extract_features(cells)                                # dict[str, float], fixed schema
roi   = extract_features_roi(cells, shapely.box(1000, 1000, 5000, 5000))

viz.plot_clusters(cells, "immune")
viz.plot_heterogeneity(cells, "area")
viz.plot_graph(cells)
```

Also: `python cell_masks_to_geojson.py` → one QuPath-importable GeoJSON per slide (cells classified by type, stable colours).

### Design

- Units: micrometres throughout (`inference_mpp` read per slide).
- Cancer-agnostic: the only domain knowledge is `pathomics.io.DEFAULT_ROLES` (cell-type → `immune | tumor | stroma | epithelial | endothelium | other`).
- Base graph: Delaunay triangulation, edges > `max_edge_um` (50 µm) pruned. Density-adaptive, parameter-light, scales to millions of cells.
- No learned GNN in the feature layer: a label-trained graph constructor is a model, not a reusable feature extractor. Neighbourhood aggregation is the unlearned GraphSAGE-mean equivalent. Learning belongs in the downstream survival model.

Parameters: `min_confidence=0.0`, `max_edge_um=50`, `min_cluster=5`, `near_um=20`.

### Feature catalogue

Names follow `<family>__<quantity>[__<stat>]`. Stats (when present): `mean / std / p10 / p50 / p90 / skew`. Morphology channels: `area, perimeter, major_axis, minor_axis, equiv_diameter` (µm / µm²) and `eccentricity, circularity, solidity, extent` (unitless). Roles: `immune, tumor, stroma, epithelial, endothelium, other`.

| family | n | what's in it |
|---|---:|---|
| `slide__` | 3 | `n_cells`, `tissue_area_mm2`, `mean_neighbors_per_cell` |
| `comp__` | 9 | `density_per_mm2`, `<role>_frac` (×6), `tumor_immune_ratio`, `role_entropy` |
| `morph__` | 25 | bulk nuclear morphometry — 5 channels (`area`, `eccentricity`, `circularity`, `solidity`, `major_axis`) × {mean, std, p10, p90, skew} |
| `morph_tumor__` | 12 | same restricted to the tumour compartment — 3 channels (`area`, `eccentricity`, `circularity`) × {mean, std, p10, p90} |
| `diversity__` | 46 | nuclear morphological heterogeneity (next table) |
| `spatil__` | 49 | immune spatial organisation (next table) |

**`diversity__` (46).** How variable nuclear shape is locally and across cell-type boundaries.

| pattern | meaning |
|---|---|
| `<channel>_local_sd__{mean,std,p10,p90}` | within-neighbourhood SD of `channel`, aggregated across cells |
| `<channel>_at_type_boundary__{mean,std,p10,p90}` | `\|Δchannel\|` across Delaunay edges that cross a cell-type boundary |
| `cell_type_diversity__{mean,std,p50,p90}` | Shannon entropy of cell-type composition in each cell's 1-hop neighbourhood |
| `neighborhood_mixedness__mean` | `1 − max_role_fraction` in the neighbourhood, averaged across cells |
| `cross_type_edge_frac` | fraction of Delaunay edges crossing a cell-type boundary |

**`spatil__` (49).** Spatial organisation of immune relative to tumour and stroma.

*Per-population clusters* — Delaunay-connected components ≥ 5 cells, 6 features × {immune, tumor, stroma} = 18:

| feature | meaning |
|---|---|
| `<role>__n_clusters` | how many clusters |
| `<role>__clustered_frac` | fraction of role cells in a cluster |
| `<role>__mean_cluster_size`, `<role>__largest_cluster_size` | cells per cluster |
| `<role>__mean_cluster_area_mm2` | convex-hull area |
| `<role>__mean_cluster_density` | cells per µm² inside the hull |

*Pairwise nearest-neighbour*, 4 features × 3 ordered pairs ({immune→tumor, immune→stroma, tumor→stroma}) = 12: `<A>_to_<B>__nndist_um__{median, p10, p90}` and `<A>_near_<B>__frac` (within `near_um = 20`).

*Cluster-cluster hull overlap*, 2 features × 3 pairs = 6: `<A>_cluster_meets_<B>_cluster__frac` (do the hulls touch at all) and `<A>_cluster_overlap_<B>_cluster__mean_ratio` (mean intersection area / A-hull area).

*Infiltration and spatial preference* (13):

| feature | meaning |
|---|---|
| `tumor_local_immune_frac__{mean,std,p50,p90}` | immune fraction in each tumour cell's neighbourhood. `mean` = overall infiltration; `std`/`p90` = its spatial heterogeneity |
| `tumor_with_immune_neighbor_frac` | fraction of tumour cells with ≥ 1 immune Delaunay neighbour |
| `intratumoral_vs_stromal_immune` | ratio of immune-near-tumor frac to immune-near-stroma frac |
| `<role>_same_role_frac` | mean fraction of same-role neighbours, per role |
| `immune_tumor_edge_enrichment` | `log(observed / expected)` of immune–tumour edges vs the chance level given the marginals. Negative ⇒ immune-excluded |
| `immune_spatial_dispersion` | observed mean immune–immune NN distance / random expectation. > 1 dispersed, < 1 clumped |
| `immune_cluster_roundness__mean` | `4πA/P²` of immune-cluster hulls (0 = elongated, 1 = perfect circle) — proxy for TLS-like aggregates |
| `largest_immune_cluster_area_mm2` | area of the biggest immune aggregate |

## Status

**26 / 29 slides** fully through stages 1–3. `MPe13LN.svs` and `MPe20P.svs` are truncated on the share (OpenSlide cannot open them); `MPe3P-A9.svs` opens but has a corrupt JPEG tile in the pyramid — re-pulling from the share returns byte-identical files, so these need a fresh export from MDACC. `MPe3P-A9` still has Trident output but no HistoPLUS output and so no stage-3 features yet.
