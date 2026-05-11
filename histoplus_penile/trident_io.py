"""Read Trident's patch-coordinate h5 files and adapt them for HistoPLUS.

Trident writes, per slide, ``<job_dir>/<coords_subdir>/patches/<name>_patches.h5`` with:
  - dataset ``coords`` : (N, 2) int — level-0 pixel (x, y) of each patch's top-left
  - attrs on ``coords`` : ``patch_size_level0`` (patch side in level-0 px),
    ``level0_width``/``level0_height`` (slide level-0 dims), ``level0_magnification``,
    ``target_magnification``, ``patch_size``, ``overlap``, ``name``, ...

HistoPLUS's ``extract(slide, coords, deepzoom_level, segmentor, tile_size=...)`` wants
``coords`` as **DeepZoom tile (col, row) indices** at ``deepzoom_level``, where one tile
is ``tile_size`` px wide *at that DeepZoom level*. It then re-grids those tiles to the
segmentor's inference tile size at the segmentor's MPP.

So we hand HistoPLUS the Trident patch grid directly:
  - ``deepzoom_level`` = the full-resolution DeepZoom level (the level-0 pixel grid).
    For an OpenSlide DeepZoomGenerator this is ``level_count - 1`` and
    ``level_count == ceil(log2(max(W, H))) + 1`` — computable from the stored dims,
    no need to open the slide.
  - ``tile_size``    = ``patch_size_level0`` (so one DeepZoom tile == one Trident patch).
  - ``coords``       = ``trident_level0_xy // patch_size_level0``.

Coverage note: HistoPLUS coarsens these to its (~656 px) inference tiles by snapping each
patch's top-left onto that grid. Because Trident's grid is dense and contiguous over
tissue, the union of snapped inference tiles covers the kept tissue; only an isolated
patch at a tissue edge can lose ~half a patch of coverage, which the segmentor's tile
overlap largely absorbs. This mirrors HistoPLUS's own native behaviour.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


@dataclass
class TridentTiling:
    """Trident's patch grid for one slide, in the form HistoPLUS expects."""

    name: str
    coords_px: np.ndarray          # (N, 2) int — level-0 px top-left of each patch
    coords_tiles: np.ndarray       # (N, 2) int — DeepZoom (col, row) on the patch grid
    deepzoom_level: int            # full-resolution DeepZoom level
    tile_size: int                 # patch side, in level-0 px (== patch_size_level0)
    level0_width: int
    level0_height: int
    level0_magnification: int
    target_magnification: int

    @property
    def n_patches(self) -> int:
        return len(self.coords_px)


def _full_res_deepzoom_level(width: int, height: int) -> int:
    """DeepZoom level whose pixel grid equals OpenSlide level 0 (no bounds limiting)."""
    n_levels = int(math.ceil(math.log2(max(width, height)))) + 1
    return n_levels - 1


def load_trident_tiling(patches_h5: str | Path) -> TridentTiling:
    """Load one ``*_patches.h5`` and return a :class:`TridentTiling`."""
    patches_h5 = Path(patches_h5)
    with h5py.File(patches_h5, "r") as f:
        coords_px = np.asarray(f["coords"][:], dtype=np.int64)
        attrs = dict(f["coords"].attrs)

    tile_size = int(attrs["patch_size_level0"])
    width = int(attrs["level0_width"])
    height = int(attrs["level0_height"])
    name = str(attrs.get("name", patches_h5.stem.replace("_patches", "")))

    if np.any(coords_px % tile_size != 0):
        # Trident lays patches on a multiple-of-patch_size_level0 grid; warn rather than
        # silently round, since a mismatch would mean a different reader/config.
        raise ValueError(
            f"{name}: patch coords are not multiples of patch_size_level0={tile_size}; "
            "the Trident h5 may have been written with an unexpected config."
        )

    return TridentTiling(
        name=name,
        coords_px=coords_px,
        coords_tiles=coords_px // tile_size,
        deepzoom_level=_full_res_deepzoom_level(width, height),
        tile_size=tile_size,
        level0_width=width,
        level0_height=height,
        level0_magnification=int(attrs.get("level0_magnification", 0)),
        target_magnification=int(attrs.get("target_magnification", 0)),
    )


def segmentor_mpp_for_tiling(tiling: TridentTiling) -> float:
    """Pick the HistoPLUS model MPP (0.25 for 40x slides, 0.5 for 20x slides).

    HistoPLUS only ships a 20x (MPP 0.5) and a 40x (MPP 0.25) CellViT; it then snaps to
    whichever DeepZoom level is closest to that MPP (20% rel-tol), so the exact native
    MPP (e.g. 0.2513) does not need to match.
    """
    return 0.25 if tiling.level0_magnification >= 40 else 0.5


def iter_trident_slides(trident_dir: str | Path, coords_subdir: str):
    """Yield (name, patches_h5_path) for every slide Trident produced patch coords for."""
    patches_dir = Path(trident_dir) / coords_subdir / "patches"
    for h5 in sorted(patches_dir.glob("*_patches.h5")):
        yield h5.stem.replace("_patches", ""), h5
