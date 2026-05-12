"""Turn HistoPLUS cell masks into nuclear-diversity features.

From each ``<histoplus_dir>/<slide>/cell_masks.json`` we read every cell's centroid and
``cell_type`` and compute, over the 13 HistoPLUS cell types:

  - ``richness``       : number of distinct cell types present
  - ``shannon``        : Shannon entropy  -Σ p_i ln p_i        (in nats)
  - ``shannon_norm``   : shannon / ln(richness)                 (Pielou's evenness, 0..1)
  - ``simpson``        : Gini-Simpson index  1 - Σ p_i²
  - ``inv_simpson``    : inverse Simpson     1 / Σ p_i²
  - ``n_cells``        : total cells
  - one ``n_<celltype>`` column per cell type (raw counts)

…at two granularities:

  * **per Trident patch** — cells are binned to the 512px/20x patch that contains their
    centroid, so the rows line up 1-for-1 (and in the same order) with Trident's
    ``coords`` / CONCH feature h5. Written to ``<out>/patch_diversity/<slide>.csv``.
  * **per slide** — one row per slide (plus tissue area / cell density from Trident's
    patch count). Appended to ``<out>/slide_diversity.csv``.

Assumes cell centroids in the JSON are in level-0 pixels (true for the 20x/40x HistoPLUS
models on 20x/40x slides: the inference DeepZoom level is the slide's full-resolution
level). A sanity warning is printed if many centroids fall outside Trident's patches.

Usage::

    python -m histoplus_penile.nuclear_diversity \
        --histoplus_dir ~/Documents/penile/histoplus_output \
        --trident_dir   ~/Documents/penile/trident_output \
        --coords_subdir 20x_512px_0px_overlap \
        --out           ~/Documents/penile/histoplus_output \
        [--min_confidence 0.0]
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from histoplus_penile.trident_io import iter_trident_slides, load_trident_tiling

CELL_MASKS_FILENAME = "cell_masks.json"


# --------------------------------------------------------------------------- IO


def load_cells(cell_masks_json: Path, min_confidence: float = 0.0):
    """Return (centroids: (M,2) float, cell_types: list[str], inference_mpp: float)."""
    with open(cell_masks_json) as f:
        data = json.load(f)
    centroids, types = [], []
    for tile in data.get("cell_masks", []):
        for cell in tile.get("masks", []):
            if cell.get("confidence", 1.0) < min_confidence:
                continue
            cx, cy = cell["centroid"]
            centroids.append((float(cx), float(cy)))
            types.append(str(cell["cell_type"]))
    arr = np.asarray(centroids, dtype=float).reshape(-1, 2)
    return arr, types, float(data.get("inference_mpp", float("nan")))


# ------------------------------------------------------------------- diversity


def diversity_from_counts(counts: Counter, all_types: list[str]) -> dict:
    """Compute the diversity scalars + per-type counts for one Counter of cell types."""
    n = sum(counts.values())
    out = {"n_cells": int(n)}
    out.update({f"n_{t}": int(counts.get(t, 0)) for t in all_types})
    if n == 0:
        out.update(richness=0, shannon=0.0, shannon_norm=0.0, simpson=0.0, inv_simpson=0.0)
        return out
    p = np.array([counts.get(t, 0) for t in all_types], dtype=float) / n
    p_nz = p[p > 0]
    richness = int(len(p_nz))
    shannon = float(-(p_nz * np.log(p_nz)).sum())
    sum_sq = float((p ** 2).sum())
    out.update(
        richness=richness,
        shannon=shannon,
        shannon_norm=float(shannon / math.log(richness)) if richness > 1 else 0.0,
        simpson=float(1.0 - sum_sq),
        inv_simpson=float(1.0 / sum_sq) if sum_sq > 0 else 0.0,
    )
    return out


# ------------------------------------------------------------------ per slide


def process_slide(name: str, patches_h5: Path, cell_masks_json: Path,
                  all_types: list[str], min_confidence: float):
    """Return (patch_df, slide_row, n_cells_outside)."""
    tiling = load_trident_tiling(patches_h5)
    centroids, types, inference_mpp = load_cells(cell_masks_json, min_confidence)
    side = tiling.patch_size_level0

    # bin every cell to a Trident patch by centroid
    patch_index = {(int(x // side), int(y // side)): i for i, (x, y) in enumerate(tiling.coords_px)}
    per_patch_counts = [Counter() for _ in range(tiling.n_patches)]
    n_outside = 0
    for (cx, cy), t in zip(centroids, types):
        key = (int(cx // side), int(cy // side))
        idx = patch_index.get(key)
        if idx is None:
            n_outside += 1
        else:
            per_patch_counts[idx][t] += 1

    # per-patch table, rows in Trident's coords order
    rows = []
    for i, (x, y) in enumerate(tiling.coords_px):
        row = {"patch_x_level0": int(x), "patch_y_level0": int(y)}
        row.update(diversity_from_counts(per_patch_counts[i], all_types))
        rows.append(row)
    patch_df = pd.DataFrame(rows)

    # per-slide row
    total_counts = Counter(types)
    slide_row = {"slide": name,
                 "n_patches": tiling.n_patches,
                 "patch_size_level0": side,
                 "level0_magnification": tiling.level0_magnification,
                 "inference_mpp": inference_mpp,
                 "n_cells_outside_patches": int(n_outside)}
    slide_row.update(diversity_from_counts(total_counts, all_types))
    # crude tissue area / density (level-0 px and, if magnification known, mm^2)
    tissue_px2 = tiling.n_patches * side * side
    slide_row["tissue_area_level0_px2"] = int(tissue_px2)
    if not math.isnan(inference_mpp):
        # the kept patches are at target magnification; their level-0 px size 'side'
        # corresponds to side*mpp0 microns. mpp0 ~= inference_mpp for 40x, ~= inference_mpp
        # for 20x as well (both are the slide's level-0 mpp). Use inference_mpp as mpp0.
        mm2 = tissue_px2 * (inference_mpp ** 2) / 1e6
        slide_row["tissue_area_mm2"] = float(mm2)
        slide_row["cells_per_mm2"] = float(slide_row["n_cells"] / mm2) if mm2 > 0 else 0.0
    return patch_df, slide_row, n_outside


# ------------------------------------------------------------------------ main


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--histoplus_dir", required=True, type=Path)
    ap.add_argument("--trident_dir", required=True, type=Path)
    ap.add_argument("--coords_subdir", default="20x_512px_0px_overlap")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--min_confidence", type=float, default=0.0,
                    help="Drop cells below this HistoPLUS confidence (default: %(default)s).")
    args = ap.parse_args(argv)

    # pair up Trident patch h5s with HistoPLUS results
    pairs = []
    for name, patches_h5 in iter_trident_slides(args.trident_dir, args.coords_subdir):
        cm = args.histoplus_dir / name / CELL_MASKS_FILENAME
        if cm.exists():
            pairs.append((name, patches_h5, cm))
        else:
            print(f"[{name}] no {CELL_MASKS_FILENAME} yet; skipping.", flush=True)
    if not pairs:
        print("No HistoPLUS results found to summarise.", flush=True)
        return 1

    # pass 1: global cell-type vocabulary (keeps columns consistent across slides)
    all_types: set[str] = set()
    for _, _, cm in pairs:
        _, types, _ = load_cells(cm, args.min_confidence)
        all_types.update(types)
    all_types = sorted(all_types)
    print(f"{len(pairs)} slides; cell types seen: {all_types}", flush=True)

    patch_out = args.out / "patch_diversity"
    patch_out.mkdir(parents=True, exist_ok=True)
    slide_rows = []
    for name, patches_h5, cm in pairs:
        patch_df, slide_row, n_outside = process_slide(name, patches_h5, cm, all_types, args.min_confidence)
        patch_df.to_csv(patch_out / f"{name}.csv", index=False)
        slide_rows.append(slide_row)
        msg = f"[{name}] {slide_row['n_cells']} cells, richness={slide_row['richness']}, " \
              f"shannon={slide_row['shannon']:.3f}, simpson={slide_row['simpson']:.3f}"
        if n_outside:
            msg += f"  (warning: {n_outside} cells outside Trident patches)"
        print(msg, flush=True)

    slide_df = pd.DataFrame(slide_rows)
    slide_csv = args.out / "slide_diversity.csv"
    slide_df.to_csv(slide_csv, index=False)
    print(f"\nWrote per-patch tables -> {patch_out}/<slide>.csv")
    print(f"Wrote per-slide summary -> {slide_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
