# Penile cancer — MDACC cohort WSI processing

Two-stage histopathology pipeline for the MDACC penile-cancer cohort (29 `.svs` WSIs, scanned at 40×):

1. **Trident** — GrandQC tissue QC → 20× / 512 px patch coordinates → CONCH v1.5 patch features → TITAN slide features.
2. **HistoPLUS** — CellViT cell segmentation + classification, run on *exactly* the tissue tiles Trident kept (no second tissue detection).
3. **Nuclear diversity** — per-patch and per-slide cell-type diversity (richness / Shannon / Simpson) derived from the HistoPLUS masks.

All outputs land under this folder. WSIs, `*.h5`, result `*.json`/`*.csv`, and `logs/` are git-ignored — only code and these docs are tracked.

---

## Repository layout

```
scripts/
  run_trident_penile.sh      stage WSIs locally → seg+coords+CONCH (A) → TITAN (B)
  run_histoplus_penile.sh    HistoPLUS cell masks (cells) → diversity tables (diversity)
  reprocess_failed.sh        re-pull + re-run only the slides that failed a previous run
histoplus_penile/            python package used by run_histoplus_penile.sh
  trident_io.py              read Trident patch h5  →  HistoPLUS DeepZoom inference tiles
  extract_cells.py           run the CellViT segmentor per slide  →  cell_masks.json
  nuclear_diversity.py       cell_masks.json  →  per-patch / per-slide diversity CSVs
README.md
trident_output/              (git-ignored) Trident results
histoplus_output/            (git-ignored) HistoPLUS results + diversity tables
logs/                        (git-ignored) run logs
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

---

## Running the pipeline

> Both runs are **resumable**: every stage skips slides whose output already exists, so you can
> re-launch after an interruption. Launch them under `tmux` — the Trident run in particular is long.

### 1 — Trident  (env `trident`)

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

### 2 — HistoPLUS + nuclear diversity  (env `histoplus`)

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

### 3 — Reprocessing only the slides that failed

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
truncated ones aside automatically. **Current state: 26 / 29 slides fully through both pipelines.**
