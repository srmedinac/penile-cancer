# Penile cancer — MDACC cohort processing

Pipeline for the MDACC penile cancer WSIs (29 `.svs`):

1. **Trident** — GrandQC tissue QC + 20x/512px patch coords + CONCH v1.5 patch features + TITAN slide features.
2. **HistoPLUS** — CellViT cell segmentation/classification, reusing Trident's tissue grid (no re-detection).
3. **Nuclear diversity** — per-patch and per-slide cell-type diversity (Shannon / Simpson / richness) from the HistoPLUS masks.

Everything writes under this folder. Slides, `*.h5`, results CSV/JSON, and `logs/` are git-ignored.

## Layout

```
scripts/
  run_trident_penile.sh      stage WSIs locally, then seg+coords+CONCH (Stage A) and TITAN (Stage B)
  run_histoplus_penile.sh    HistoPLUS cell masks (cells) and diversity tables (diversity)
histoplus_penile/            python package used by run_histoplus_penile.sh
  trident_io.py              read Trident patch h5 -> HistoPLUS DeepZoom tiles
  extract_cells.py           run the CellViT segmentor per slide -> cell_masks.json
  nuclear_diversity.py       cell_masks.json -> per-patch / per-slide diversity CSVs
trident_output/              (git-ignored) Trident results
histoplus_output/            (git-ignored) HistoPLUS results + diversity tables
logs/                        (git-ignored) run logs
```

## Run it

Both `run_*` scripts grab `HF_TOKEN` from `~/.bashrc` and `conda activate` the right env themselves; launch them in `tmux` (the Trident run in particular is long).

### 1. Trident  (env: `trident`)

```bash
cd ~/Documents/penile
tmux new -s trident_penile
bash scripts/run_trident_penile.sh all 2>&1 | tee logs/trident_penile.log
```

`all` = `stage` (cp the 29 `.svs` from the slow Emory CIFS share to `/tmp/penile_wsi_local`) → `A` (seg + coords + CONCH, `--seg_batch_size 16 --feat_batch_size 64`) → `B` (TITAN). Stages are re-runnable individually: `bash scripts/run_trident_penile.sh stage|A|B`. Trident skips already-done slides, so the run is resumable.

Outputs in `trident_output/`:
- `contours/*.jpg`, `contours_geojson/*.geojson` — tissue QC overlays
- `20x_512px_0px_overlap/patches/<slide>_patches.h5` — patch coords
- `20x_512px_0px_overlap/features_conch_v15/<slide>.h5` — CONCH patch features
- `20x_512px_0px_overlap/slide_features_titan/<slide>.h5` — TITAN slide features

### 2. HistoPLUS + nuclear diversity  (env: `histoplus`)

Needs Trident's `patches/` from step 1.

```bash
cd ~/Documents/penile
tmux new -s histoplus_penile
bash scripts/run_histoplus_penile.sh all 2>&1 | tee logs/histoplus_penile.log
```

`all` = `cells` (CellViT on each slide, 40x model @ MPP 0.25, batch 8 — reuses Trident's kept tissue as the tile grid; skips slides whose `cell_masks.json` exists) → `diversity`. Or run a stage alone: `bash scripts/run_histoplus_penile.sh cells|diversity`.

Outputs in `histoplus_output/`:
- `<slide>/cell_masks.json` — HistoPLUS cell masks (their native format; QuPath-importable, see owkin/histoplus#23)
- `patch_diversity/<slide>.csv` — one row per Trident patch, **same order as `<slide>_patches.h5`** (`patch_x_level0,patch_y_level0,n_cells,richness,shannon,shannon_norm,simpson,inv_simpson,n_<celltype>…`) — drop-in alongside the CONCH features
- `slide_diversity.csv` — one row per slide (+ tissue area, cells/mm²)

## How Trident's coords feed HistoPLUS

Trident's `*_patches.h5` stores level-0 pixel top-left of each 512px@20x patch (= `patch_size_level0` px at level 0) plus the slide's level-0 dims. `histoplus.extract` wants tiles as DeepZoom (col,row) at a DeepZoom level. We pass the patch grid straight through: `deepzoom_level` = the full-resolution level (`ceil(log2(max(W,H)))`), `tile_size` = `patch_size_level0`, `coords` = `level0_xy // patch_size_level0`. HistoPLUS then re-grids these to its ~656px inference tiles at MPP 0.25. So cells are segmented in exactly the tissue Trident kept — better than HistoPLUS's built-in Otsu, and nothing is re-tiled. (See `histoplus_penile/trident_io.py` for the details and the edge-coverage note.)

## Known bad source slides

`MPe13LN.svs` and `MPe20P.svs` are truncated on the CIFS share (TIFF first-IFD offset past EOF — OpenSlide can't open them). `MPe3P-A9.svs` opens but has ≥1 corrupt JPEG tile that may fail feature/cell extraction. All three are skipped by `--skip_errors`; re-pull from source to recover them.
