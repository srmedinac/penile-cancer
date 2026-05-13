# Penile cancer — MDACC cohort WSI processing

Three-stage histopathology pipeline for the MDACC penile-cancer cohort (29 `.svs` WSIs, scanned at 40×):

1. **Trident** — GrandQC tissue QC → 20× / 512 px patch coordinates → CONCH v1.5 patch features → TITAN slide features.
2. **HistoPLUS** — CellViT cell segmentation + classification, run on *exactly* the tissue tiles Trident kept (no second tissue detection).
3. **Handcrafted WSI features** — the `pathomics/` package turns HistoPLUS cell masks into a tidy, interpretable per-slide feature table: nuclear morphological heterogeneity, immune spatial organisation (spaTIL v2), composition, and bulk morphometry. Plus a baseline cell-type richness/Shannon/Simpson summary aligned to Trident's patch grid.

All outputs land under this folder. WSIs, `*.h5`, result `*.json`/`*.csv`/`*.parquet`, and `logs/` are git-ignored — only code, the small feature table (`features/penile_features.parquet` + `.csv`), and these docs are tracked.

---

## Repository layout

```
scripts/
  run_trident_penile.sh        stage WSIs locally → seg+coords+CONCH (A) → TITAN (B)
  run_histoplus_penile.sh      HistoPLUS cell masks (cells) → diversity tables (diversity)
  reprocess_failed.sh          re-pull + re-run only the slides that failed a previous run
histoplus_penile/              python package used by run_histoplus_penile.sh
  trident_io.py                read Trident patch h5  →  HistoPLUS DeepZoom inference tiles
  extract_cells.py             run the CellViT segmentor per slide  →  cell_masks.json
  nuclear_diversity.py         cell_masks.json  →  per-Trident-patch / per-slide diversity CSVs
pathomics/                     python package: cell_masks.json → 145 handcrafted WSI features
  io.py, morphology.py, graph.py, features.py, viz.py, __main__.py
cell_masks_to_geojson.py       histoplus json → QuPath-compatible GeoJSON, one per slide
features/penile_features.{parquet,csv}   the analysis output (one row per slide × 145 columns)
README.md
trident_output/                (git-ignored) Trident results
histoplus_output/              (git-ignored) HistoPLUS results + diversity tables + cells.parquet cache
logs/                          (git-ignored) run logs
```

---

## Prerequisites

- **Conda** (miniconda) and an NVIDIA GPU (developed on a single 24 GB card).
- The **Trident** repo cloned locally — the scripts call `python run_batch_of_slides.py` from it:
  ```bash
  git clone https://github.com/mahmoodlab/trident.git ~/trident
  ```
- A **Hugging Face account with access granted** to the gated model weights the pipeline pulls:
  `MahmoodLab/conchv1_5`, `MahmoodLab/TITAN`, the GrandQC segmentation weights, and `owkin/histoplus`.
  Put a token in `~/.bashrc` (both `run_*` scripts read it from there, because a non-interactive
  shell skips most of `.bashrc`):
  ```bash
  echo 'export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx' >> ~/.bashrc
  ```
- **Source WSIs.** The cohort lives on the Madabhushi-Lab CIFS share at
  `…/curtis-pettaway/40x_Niki/*.svs` (see `SRC_WSI` in `scripts/run_trident_penile.sh`). The Trident
  script copies them once to a local SSD cache `/tmp/penile_wsi_local/` — the share is slow/contended,
  and Trident's own `--wsi_cache` stager is much slower than a plain `cp`.

### Conda environments

Two separate envs (Trident and HistoPLUS pin conflicting torch/pandas versions):

```bash
# Trident
conda create -y -n trident python=3.10
conda activate trident
pip install -e ~/trident          # installs torch, openslide-python, timm, … per Trident's pyproject

# HistoPLUS
conda create -y -n histoplus python=3.11
conda activate histoplus
pip install histoplus              # → histoplus 1.0.0
pip install h5py "pandas<3"        # pandas 3.0 breaks histoplus post-processing
```

The `run_*` scripts `conda activate` the right env themselves — you only need the envs to exist.

For **stage 3** (`pathomics/`, runs anywhere, no GPU):

```bash
pip install numpy pandas scipy shapely pyarrow matplotlib scikit-learn
```

---

## Running the pipeline

> Both gu-side runs are **resumable**: every stage skips slides whose output already exists, so you can
> re-launch after an interruption. Launch them under `tmux` — the Trident run in particular is long.

### 1 — Trident  (env `trident`, on gu)

```bash
cd ~/Documents/penile
tmux new -s trident_penile
bash scripts/run_trident_penile.sh all 2>&1 | tee logs/trident_penile.log
```

`all` = **stage** (`cp` the `.svs` from the CIFS share → `/tmp/penile_wsi_local`) → **A** (GrandQC tissue
seg + QC contours + 20×/512 px/0 px-overlap patch coords + CONCH v1.5 patch features; `--seg_batch_size 16
--feat_batch_size 64` — in a fused `--task all` run the seg model and CONCH share the GPU, and a larger
seg batch OOMs a 24 GB card) → **B** (TITAN slide features; reads the CONCH `.h5`s, no WSI re-read).

Run a single stage with `bash scripts/run_trident_penile.sh stage|A|B`.

Outputs under `trident_output/`:

| path | contents |
|---|---|
| `contours/<slide>.jpg`, `contours_geojson/<slide>.geojson` | tissue QC overlays (QuPath-importable) |
| `20x_512px_0px_overlap/patches/<slide>_patches.h5` | patch coords — `coords` (N×2 level-0 px TL) + attrs (`patch_size_level0`, `level0_width/height`, …) |
| `20x_512px_0px_overlap/features_conch_v15/<slide>.h5` | CONCH v1.5 patch features |
| `20x_512px_0px_overlap/slide_features_titan/<slide>.h5` | TITAN slide-level features |

### 2 — HistoPLUS + per-patch diversity  (env `histoplus`, on gu)

Needs Trident's `patches/` from step 1.

```bash
cd ~/Documents/penile
tmux new -s histoplus_penile
bash scripts/run_histoplus_penile.sh all 2>&1 | tee logs/histoplus_penile.log
```

`all` = **cells** (CellViT on each slide — 40× model @ MPP 0.25, batch 8 — tiled over Trident's kept
tissue; skips slides that already have `cell_masks.json`) → **diversity**. Run one stage with
`bash scripts/run_histoplus_penile.sh cells|diversity`.

Outputs under `histoplus_output/`:

| path | contents |
|---|---|
| `<slide>/cell_masks.json` | HistoPLUS cell masks in their native format (QuPath-importable — see owkin/histoplus#23). Big: ~100–300 MB / ~100–250 k cells per slide |
| `patch_diversity/<slide>.csv` | one row per Trident patch, **same order as `<slide>_patches.h5`** — `patch_x_level0, patch_y_level0, n_cells, richness, shannon, shannon_norm, simpson, inv_simpson, n_<celltype>…` — drop-in alongside the CONCH features |
| `slide_diversity.csv` | one row per slide (+ tissue area, cells/mm²) |

### 3 — WSI handcrafted features (`pathomics/`, runs anywhere)

Bring the HistoPLUS outputs over from gu:

```bash
rsync -avh gu:/home/smedin7/Documents/penile/histoplus_output/ histoplus_output/
```

Optionally convert each `cell_masks.json` to a QuPath-importable GeoJSON (detection objects
classified by cell type, stable colours):

```bash
python cell_masks_to_geojson.py
```

Extract the per-slide feature table — one row per slide, 145 columns, fixed schema:

```bash
python -m pathomics histoplus_output -o features/penile_features.parquet --min-confidence 0.5
```

Or from Python:

```python
from pathomics import load_cells, extract_features, viz, extract_features_roi
import shapely

cells = load_cells("histoplus_output/MPe08B/cell_masks.json", min_confidence=0.5)
feats = extract_features(cells)                            # dict[str, float], fixed schema

# ROI-level: pass a shapely polygon in micrometres (matching cells.cx / cells.cy)
roi_feats = extract_features_roi(cells, shapely.box(1000, 1000, 5000, 5000))

viz.plot_clusters(cells, "immune")                         # immune-aggregate hulls
viz.plot_heterogeneity(cells, "area")                      # local nuclear-size dispersion
viz.plot_graph(cells)                                      # pruned-Delaunay cell graph
```

The first call on a slide streams the (possibly multi-GB) cell file, derives morphology, and writes
a compact `cells.parquet` next to it; subsequent loads are ~0.1 s. Per-slide feature extraction off
the cache: ~2 s @135 k cells → ~75 s @1.3 M cells.

#### What `pathomics/` extracts

145 named features per slide, organised into six families. **Naming convention:**
`<family>__<quantity>[__<stat>]` — the family says *what kind of measurement*, the quantity
says *what is measured*, the stat says *how it was aggregated across cells / edges*. Stat
suffixes (when present): `__mean`, `__std`, `__p10`, `__p50`, `__p90`, `__skew`. *Morphology
channels* used throughout: `area` (µm²), `perimeter` / `major_axis` / `minor_axis` /
`equiv_diameter` (µm), `eccentricity` / `circularity` / `solidity` / `extent` (unitless).
*Roles* used in spatial features: `immune`, `tumor`, `stroma`, `epithelial`, `endothelium`,
`other` (configurable via `pathomics.io.DEFAULT_ROLES`).

| family | n | what's in it |
|---|---:|---|
| `meta__` | 3 | `n_cells`, `tissue_area_mm2` (occupied-grid area), `mean_degree` (Delaunay) |
| `comp__` | 10 | `n_cells`, `density_per_mm2`, `role_entropy`, `tumor_immune_ratio`, plus one `<role>_frac` per role |
| `morph_global__` | 25 | bulk nuclear morphometry over **all** cells: 5 channels (`area`, `eccentricity`, `circularity`, `solidity`, `major_axis`) × {`mean`, `std`, `p10`, `p90`, `skew`} |
| `morph_tumor__` | 12 | same, restricted to tumour: 3 channels (`area`, `eccentricity`, `circularity`) × {`mean`, `std`, `p10`, `p90`} |
| `het__` | 46 | **nuclear morphological heterogeneity** — see below |
| `spatil__` | 49 | **immune spatial organisation (spaTIL v2)** — see below |

**`het__` — nuclear morphological heterogeneity (46).** Two views per morphology channel
(5 channels × 2 views × 4 stats = 40), plus a few global scalars:

| pattern | meaning |
|---|---|
| `<channel>_nbhd_disp__{mean,std,p10,p90}` | within-neighbourhood standard deviation of the channel for each cell, aggregated across cells — *how morphologically variable each cell's local neighbourhood is* |
| `<channel>_heterotypic_contrast__{mean,std,p10,p90}` | `|Δchannel|` over Delaunay edges connecting two **different** cell types, aggregated — *morphological distinctness of neighbouring populations* |
| `local_type_entropy__{mean,std,p50,p90}` | Shannon entropy of cell-type composition in each cell's 1-hop neighbourhood, aggregated |
| `local_role_mixing__mean` | `1 − max_role_fraction` in each neighbourhood, averaged — global tissue "mixedness" |
| `heterotypic_edge_frac` | fraction of all Delaunay edges that cross a cell-type boundary |

**`spatil__` — immune spatial organisation (49).** Four blocks:

*Per-population clusters* (Delaunay-connected components ≥5 cells), 6 features × 3 roles = 18:

| feature | meaning |
|---|---|
| `<role>__n_clusters` | how many spatial clusters of that role |
| `<role>__clustered_frac` | fraction of role cells that sit in a cluster (vs isolated) |
| `<role>__mean_cluster_size` | average cells per cluster |
| `<role>__largest_cluster_size` | size of the biggest cluster |
| `<role>__mean_cluster_area_mm2` | average convex-hull area |
| `<role>__mean_cluster_density` | average cells per µm² inside the hulls |

*Pairwise nearest-neighbour co-localisation*, 4 features × 3 ordered pairs (immune→tumor,
immune→stroma, tumor→stroma) = 12: `<A>_to_<B>__nndist_um__{median,p10,p90}` (distance from
each A cell to the nearest B cell) and `<A>_near_<B>__frac` (fraction of A cells within
`near_um=20` µm of a B cell).

*Cluster-cluster hull overlap*, 2 features × 3 pairs = 6: `<A>cl_intersect_<B>cl__frac`
(fraction of A-clusters whose convex hull touches some B-cluster hull) and
`<A>cl_overlap_<B>cl__mean_ratio` (mean intersection area / A-hull area).

*Infiltration, homophily, and assortativity* (13):

| feature | meaning |
|---|---|
| `tumor_local_immune_frac__{mean,std,p50,p90}` | per tumour cell, the immune fraction of its 1-hop neighbourhood, aggregated — `mean` = overall infiltration; `std` / `p90` = **how heterogeneous infiltration is across the slide** |
| `tumor_adj_immune_frac` | fraction of tumour cells with ≥1 immune Delaunay neighbour |
| `intratumoral_vs_stromal_immune` | ratio of immune-near-tumor frac to immune-near-stroma frac |
| `<role>_homophily` | mean fraction of a role's 1-hop neighbours of the **same** role |
| `immune_tumor_assortativity` | `log(observed / expected)` of immune–tumor Delaunay edges vs the chance level given the marginals — negative ⇒ immune-excluded |
| `immune_dispersion_index` | observed mean immune-immune NN distance / random expectation given immune density — >1 dispersed, <1 clumped |
| `immune_cluster_compactness__mean` | `P²/(4πA)` of immune cluster hulls (1 = circle) — proxy for tertiary-lymphoid-structure-like aggregates |
| `largest_immune_cluster_area_mm2` | area of the biggest immune aggregate |

Current cohort: `features/penile_features.parquet` is **26 × 145** (every histoplus-segmented slide;
`MPe3P-A9` is still missing histoplus output). No NaN columns, no constant columns. Headline checks:
`immune_homophily` and `tumor_local_immune_frac__mean` track immune fraction (r ≈ 0.93, expected) but
`immune_tumor_assortativity` does *not* (r = 0.15) — it's strongly negative cohort-wide and flags the
most immune-excluded slides (`MPe01P`, `MPe15LN` ≈ −1.5) independent of bulk composition.
`immunecl_intersect_tumorcl__frac` spans 0.06–0.95 — not saturated.

#### Design notes

- **Units are micrometres everywhere.** `inference_mpp` is read from the histoplus header so
  features transfer across scanners and cohorts.
- **Cancer-agnostic.** The *only* domain knowledge is the cell-type → tissue-role map in
  `pathomics.io.DEFAULT_ROLES` (`immune / tumor / stroma / epithelial / endothelium / other`).
  A new project edits that dict; nothing else changes.
- **One graph for everything: pruned Delaunay** — the natural neighbour graph (no scale parameter,
  density-adaptive, planar and sparse at millions of cells); edges longer than `max_edge_um`
  (default 50 µm) are cut to sever links across tissue gaps.
- **No learned GNN in the feature layer.** A label-trained graph constructor is a model, not a
  reusable feature extractor — it breaks "cancer-agnostic" and "explainable". The message-passing
  *summary* is here, unlearned: per-cell neighbourhood-composition vectors are what a GraphSAGE
  mean-aggregator computes with identity weights. A learned GNN belongs in the downstream survival
  model, which can consume this pipeline's graph + node features as input.

Key parameters: `min_confidence=0.0`, `max_edge_um=50`, `min_cluster=5`, `near_um=20`.

`pathomics/` and `histoplus_penile/nuclear_diversity.py` are complementary, not duplicate:
`nuclear_diversity.py` emits one diversity row per **Trident patch** (drop-in alongside CONCH
features for patch-level models); `pathomics/` emits one row per **slide** with morphology +
spatial-organisation features suited to survival / slide-level analysis.

### Reprocessing only the slides that failed

After a run, to re-pull the failed slides' source `.svs` from the share, check whether the fresh copy
is usable, and re-run both pipelines on just those (everything already done is skipped):

```bash
tmux new -s trident_penile
bash scripts/reprocess_failed.sh        # logs to logs/reprocess_failed_<timestamp>.log
```

A re-pulled slide is installed only if OpenSlide can open it; ones that are still broken are moved aside
as `*.svs.broken` in `/tmp/penile_wsi_local/` so the pipeline globs skip them cleanly.

---

## How Trident's tissue grid feeds HistoPLUS

`histoplus.extract(slide, coords, deepzoom_level, segmentor, tile_size=…)` wants tiles as DeepZoom
`(col, row)` indices at a DeepZoom level. HistoPLUS internally re-grids to its own **656 px** inference
tiles (`INFERENCE_TILE_SIZE 784 − 2·64`) by `floor(x / 656)·656` and dedups — it never *expands* a tile.
So we don't hand it Trident's 1024 px-at-level-0 patches directly (each would collapse to one 656 tile
covering ~40 % of the patch). Instead `trident_io.to_histoplus_coords` **expands** every Trident patch
into the full set of 656 px tiles it overlaps and dedups, with `tile_size = 656` so HistoPLUS's re-grid
is a no-op. Result: cells are segmented over exactly the tissue Trident kept (better than HistoPLUS's
built-in Otsu), nothing is re-tiled, and `deepzoom_level` = the full-resolution level
`ceil(log2(max(W, H)))` — computable from the h5 attrs without opening the slide.

Edge note: a Trident patch flush against the right/bottom edge can imply a tile index one cell past the
DeepZoom grid (OpenSlide raises `ValueError: Invalid address`); `to_histoplus_coords` clamps the high
corner to `ceil(W/656)−1 × ceil(H/656)−1`.

---

## Known-bad source slides

| slide | problem | recoverable? |
|---|---|---|
| `MPe13LN.svs`  | truncated — TIFF first-IFD offset points past EOF; OpenSlide can't open it | only with a fresh export from MDACC |
| `MPe20P.svs`   | truncated — same | only with a fresh export from MDACC |
| `MPe3P-A9.svs` | opens, but ≥1 corrupt JPEG tile in the pyramid → `OpenSlideError` during tiling | only with a fresh export from MDACC |

Re-pulling these from the CIFS share returns byte-identical files — the damaged copies are what's on the
share. `--skip_errors` lets the rest of the cohort run; `scripts/reprocess_failed.sh` moves the two
truncated ones aside automatically. **Current state: 26 / 29 slides fully through stages 1–3.**
