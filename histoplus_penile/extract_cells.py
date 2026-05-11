"""Run the HistoPLUS CellViT segmentor on each slide, reusing Trident's tissue grid.

For every slide that Trident produced patch coordinates for:
  1. open the WSI with OpenSlide,
  2. turn Trident's level-0 patch grid into HistoPLUS DeepZoom tiles (see ``trident_io``),
  3. run ``histoplus.extract`` with the CellViT segmentor matching the slide's magnification,
  4. save the result as ``<out>/<slide_name>/cell_masks.json`` (HistoPLUS's own format).

The segmentor (and its HF weights) is loaded once and reused across slides. Slides whose
``cell_masks.json`` already exists are skipped, so the run is resumable.

Usage::

    python -m histoplus_penile.extract_cells \
        --wsi_dir /tmp/penile_wsi_local \
        --trident_dir ~/Documents/penile/trident_output \
        --coords_subdir 20x_512px_0px_overlap \
        --out ~/Documents/penile/histoplus_output \
        --batch_size 8 [--skip_errors] [--limit N]

Needs ``HF_TOKEN`` in the environment (the HistoPLUS weights are gated on Hugging Face).
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import openslide

from histoplus_penile.trident_io import (
    iter_trident_slides,
    load_trident_tiling,
    segmentor_mpp_for_tiling,
)

CELL_MASKS_FILENAME = "cell_masks.json"      # matches histoplus.helpers.constants.OutputFileType
INFERENCE_IMAGE_SIZE = 784                   # the only supported value (multiple of 14 & 16)


def _find_wsi(wsi_dir: Path, name: str) -> Path | None:
    for ext in (".svs", ".tif", ".tiff", ".ndpi", ".scn", ".mrxs"):
        p = wsi_dir / f"{name}{ext}"
        if p.exists():
            return p
    return None


def build_segmentor(mpp: float, mixed_precision: bool = True):
    """Load the HistoPLUS CellViT segmentor for the given model MPP (0.25 or 0.5)."""
    from histoplus.helpers.segmentor import CellViTSegmentor

    return CellViTSegmentor.from_histoplus(
        mpp=mpp,
        mixed_precision=mixed_precision,
        inference_image_size=INFERENCE_IMAGE_SIZE,
    )


def extract_one(
    wsi_path: Path,
    patches_h5: Path,
    out_dir: Path,
    segmentor_cache: dict,
    batch_size: int,
    n_workers: int,
    verbose: int,
) -> None:
    from histoplus.extract import extract

    name = patches_h5.stem.replace("_patches", "")
    slide_out = out_dir / name
    result_path = slide_out / CELL_MASKS_FILENAME
    if result_path.exists():
        print(f"[{name}] already done -> {result_path}", flush=True)
        return

    tiling = load_trident_tiling(patches_h5)
    if tiling.n_patches == 0:
        print(f"[{name}] Trident kept 0 patches; skipping.", flush=True)
        return

    mpp = segmentor_mpp_for_tiling(tiling)
    if mpp not in segmentor_cache:
        print(f"[{name}] loading CellViT segmentor (mpp={mpp}) ...", flush=True)
        segmentor_cache[mpp] = build_segmentor(mpp)
    segmentor = segmentor_cache[mpp]

    print(
        f"[{name}] {tiling.n_patches} Trident patches "
        f"(level0 {tiling.level0_width}x{tiling.level0_height}, "
        f"patch {tiling.tile_size}px, dz_level {tiling.deepzoom_level}); extracting cells...",
        flush=True,
    )
    slide = openslide.OpenSlide(str(wsi_path))
    try:
        result = extract(
            slide=slide,
            coords=tiling.coords_tiles,
            deepzoom_level=tiling.deepzoom_level,
            segmentor=segmentor,
            tile_size=tiling.tile_size,
            n_workers=n_workers,
            batch_size=batch_size,
            verbose=verbose,
        )
    finally:
        slide.close()

    slide_out.mkdir(parents=True, exist_ok=True)
    result.save(result_path)
    print(f"[{name}] saved -> {result_path}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wsi_dir", required=True, type=Path, help="Directory with the WSIs.")
    ap.add_argument("--trident_dir", required=True, type=Path, help="Trident job_dir.")
    ap.add_argument("--coords_subdir", default="20x_512px_0px_overlap",
                    help="Sub-dir under the Trident job_dir holding patches/ (default: %(default)s).")
    ap.add_argument("--out", required=True, type=Path, help="Output dir for cell_masks.json files.")
    ap.add_argument("--batch_size", type=int, default=8, help="CellViT inference batch size (default: %(default)s).")
    ap.add_argument("--n_workers", type=int, default=4, help="Dataloader workers (default: %(default)s).")
    ap.add_argument("--verbose", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="Only process the first N slides (debugging).")
    ap.add_argument("--skip_errors", action="store_true", help="Continue past slides that error out.")
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    slides = list(iter_trident_slides(args.trident_dir, args.coords_subdir))
    if args.limit:
        slides = slides[: args.limit]
    if not slides:
        print(f"No *_patches.h5 found under {args.trident_dir}/{args.coords_subdir}/patches/", file=sys.stderr)
        return 1

    print(f"Found {len(slides)} slides with Trident patch coords.", flush=True)
    segmentor_cache: dict = {}
    n_ok = n_skip = n_err = 0
    for name, patches_h5 in slides:
        wsi_path = _find_wsi(args.wsi_dir, name)
        if wsi_path is None:
            print(f"[{name}] WSI not found under {args.wsi_dir}; skipping.", flush=True)
            n_skip += 1
            continue
        t0 = time.time()
        try:
            extract_one(wsi_path, patches_h5, args.out, segmentor_cache,
                        args.batch_size, args.n_workers, args.verbose)
            n_ok += 1
            print(f"[{name}] done in {time.time() - t0:.1f}s", flush=True)
        except Exception as e:  # noqa: BLE001
            n_err += 1
            print(f"[{name}] ERROR: {e}", file=sys.stderr, flush=True)
            traceback.print_exc()
            if not args.skip_errors:
                return 2

    print(f"\nDone. ok={n_ok} skipped={n_skip} errored={n_err}", flush=True)
    return 0 if n_err == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
