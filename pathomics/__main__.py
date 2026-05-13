"""Batch feature extraction: a folder of slide subdirectories -> one feature table.

    python -m pathomics histoplus_output -o features.parquet --min-confidence 0.5

Each immediate subdirectory of ROOT that contains a matching cell file (default
``cell_masks.geojson``) becomes one row, indexed by the subdirectory name. The
per-cell ``cells.parquet`` cache is written next to each cell file on first use.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from .features import extract_features
from .io import load_cells


_FALLBACKS = ["cell_masks.json", "cell_masks.geojson"]


def _find(root: Path, pattern: str) -> list[tuple[str, Path]]:
    hits = []
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        for name in [pattern, *(f for f in _FALLBACKS if f != pattern)]:
            f = sub / name
            if f.exists():
                hits.append((sub.name, f))
                break
    return hits


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pathomics", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="folder of slide subdirectories")
    ap.add_argument("-o", "--out", type=Path, default=Path("features.parquet"))
    ap.add_argument("--pattern", default="cell_masks.json", help="cell file name inside each subdir")
    ap.add_argument("--min-confidence", type=float, default=0.0)
    ap.add_argument("--max-edge-um", type=float, default=50.0)
    ap.add_argument("--min-cluster", type=int, default=5)
    ap.add_argument("--near-um", type=float, default=20.0)
    ap.add_argument("--mpp", type=float, default=None, help="override pixel size (µm/px) if not in the file")
    ap.add_argument("--rebuild-cache", action="store_true")
    args = ap.parse_args(argv)

    slides = _find(args.root, args.pattern)
    if not slides:
        sys.exit(f"no '{args.pattern}' found under {args.root}")
    print(f"{len(slides)} slides under {args.root}")

    rows = []
    for name, path in slides:
        t0 = time.time()
        cells = load_cells(path, mpp=args.mpp, min_confidence=args.min_confidence, rebuild=args.rebuild_cache)
        feats = extract_features(cells, max_edge_um=args.max_edge_um, min_cluster=args.min_cluster, near_um=args.near_um)
        rows.append({"slide": name, **feats})
        print(f"  {name}: {len(cells):>9,} cells  {feats.get('slide__tissue_area_mm2', float('nan')):7.1f} mm²  {time.time()-t0:6.1f}s")

    df = pd.DataFrame(rows).set_index("slide").sort_index()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out)
    df.to_csv(args.out.with_suffix(".csv"))
    print(f"wrote {df.shape[0]} x {df.shape[1]} -> {args.out} (+ .csv)")


if __name__ == "__main__":
    main()
