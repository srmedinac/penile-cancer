"""Read Trident's patch-coordinate h5 files and convert them to HistoPLUS coords.

Trident writes, per slide, ``<job_dir>/<coords_subdir>/patches/<name>_patches.h5`` with:
  - dataset ``coords`` : (N, 2) int — level-0 pixel (x, y) of each patch's top-left
  - attrs on ``coords`` : ``patch_size_level0`` (patch side in level-0 px),
    ``level0_width``/``level0_height`` (slide level-0 dims), ``level0_magnification``,
    ``target_magnification``, ``patch_size``, ``overlap``, ``name``, …

HistoPLUS's ``extract(slide, coords, deepzoom_level, segmentor, tile_size=...)`` wants
``coords`` as DeepZoom tile (col, row) indices at ``deepzoom_level``, where one DeepZoom
tile is ``tile_size`` px wide. HistoPLUS then *snaps* each tile's TL to its own inference
grid of side ``INFERENCE_TILE_SIZE - 2*INFERENCE_TILE_OVERLAP`` (= 656 px) via
``floor(x0 / 656) * 656`` and *dedups* — it never expands. So if we hand it 1024-px
trident tiles, each one collapses to a single 656-tile that covers only ~41% of the
patch's area (and is offset relative to the patch), which is exactly the gap pattern
visible in QuPath.

Fix: build the HistoPLUS coords *on the 656-px grid directly*, by expanding each Trident
1024-px patch into the full set of 656-tiles it overlaps and deduping. With
``tile_size = 656`` the HistoPLUS re-tiling step is a no-op (``original_tile_size ==
target_tile_size``), so it processes exactly the tiles we hand it — fully covering
Trident's kept tissue (plus a thin fringe at patch borders where the 1024/656 grids
disagree).

``deepzoom_level`` is the **full-resolution** DeepZoom level — the level whose pixel
grid is OpenSlide level 0. Without bounds limiting that is
``ceil(log2(max(W, H)))`` (= ``DeepZoomGenerator.level_count - 1``), computable from
the dims stored in the h5 alone — no need to open the slide.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np


# HistoPLUS bakes in INFERENCE_TILE_SIZE=784 (multiple of 14 & 16, sqrt is integer) and
# INFERENCE_TILE_OVERLAP=64, giving an *inner* tile of 656 px. Import lazily so this
# module is usable without the histoplus env (e.g. when summarising results).
def _inference_inner_tile_size() -> int:
    from histoplus.helpers.constants import (
        INFERENCE_TILE_OVERLAP,
        INFERENCE_TILE_SIZE,
    )

    return INFERENCE_TILE_SIZE - 2 * INFERENCE_TILE_OVERLAP  # 784 - 128 = 656


@dataclass
class TridentTiling:
    """Trident's patch grid for one slide, plus enough metadata to map it forward."""

    name: str
    coords_px: np.ndarray            # (N, 2) — level-0 px TL of each patch
    patch_size_level0: int           # patch side, level-0 px (e.g. 1024 for 40x→20x)
    level0_width: int
    level0_height: int
    level0_magnification: int
    target_magnification: int

    @property
    def n_patches(self) -> int:
        return len(self.coords_px)


@dataclass
class HistoPlusCoords:
    """Inputs ready for ``histoplus.extract``."""

    coords_tiles: np.ndarray         # (M, 2) — DeepZoom (col, row) on the 656-px grid
    deepzoom_level: int              # full-resolution DeepZoom level of the slide
    tile_size: int                   # 656 = INFERENCE_TILE_SIZE - 2*INFERENCE_TILE_OVERLAP

    @property
    def n_inference_tiles(self) -> int:
        return len(self.coords_tiles)


def _full_res_deepzoom_level(width: int, height: int) -> int:
    """DeepZoom level whose pixel grid equals OpenSlide level 0 (no bounds limiting)."""
    return int(math.ceil(math.log2(max(width, height)))) + 1 - 1


def load_trident_tiling(patches_h5: str | Path) -> TridentTiling:
    """Load one ``*_patches.h5`` and return a :class:`TridentTiling`."""
    patches_h5 = Path(patches_h5)
    with h5py.File(patches_h5, "r") as f:
        coords_px = np.asarray(f["coords"][:], dtype=np.int64)
        attrs = dict(f["coords"].attrs)

    patch_size_l0 = int(attrs["patch_size_level0"])
    width = int(attrs["level0_width"])
    height = int(attrs["level0_height"])
    name = str(attrs.get("name", patches_h5.stem.replace("_patches", "")))

    if np.any(coords_px % patch_size_l0 != 0):
        raise ValueError(
            f"{name}: patch coords are not multiples of patch_size_level0={patch_size_l0}; "
            "the Trident h5 may have been written with an unexpected config."
        )

    return TridentTiling(
        name=name,
        coords_px=coords_px,
        patch_size_level0=patch_size_l0,
        level0_width=width,
        level0_height=height,
        level0_magnification=int(attrs.get("level0_magnification", 0)),
        target_magnification=int(attrs.get("target_magnification", 0)),
    )


def to_histoplus_coords(tiling: TridentTiling) -> HistoPlusCoords:
    """Expand Trident's patch grid into the 656-px inference-tile grid.

    For each Trident patch with level-0 TL ``(x0, y0)`` and side ``p`` we emit every
    inference tile ``(c, r)`` whose pixel range ``[c*656 : (c+1)*656]`` × ``[r*656 :
    (r+1)*656]`` intersects ``[x0 : x0+p]`` × ``[y0 : y0+p]``. Tiles shared across
    Trident patches are deduped. The result fully covers the kept tissue.
    """
    inf = _inference_inner_tile_size()
    p = tiling.patch_size_level0

    # The full-res DeepZoom level has ceil(W/inf) × ceil(H/inf) tiles; valid (col, row)
    # indices are [0, n_cols-1] × [0, n_rows-1]. A Trident patch flush against the right
    # or bottom edge (x0 a multiple of p, x0 + p > W) can expand to a tile index one past
    # that grid, and OpenSlide's DeepZoomGenerator raises "Invalid address" for it during
    # inference (this is what broke MPe18LN). Clamp the high corner to the grid bounds;
    # for any patch that doesn't straddle the slide edge this is a no-op.
    max_c = (tiling.level0_width + inf - 1) // inf - 1
    max_r = (tiling.level0_height + inf - 1) // inf - 1

    # vectorised expansion of the (c0..c1) × (r0..r1) rectangle per patch, then dedup
    xs = tiling.coords_px[:, 0]
    ys = tiling.coords_px[:, 1]
    c0 = xs // inf
    c1 = np.minimum((xs + p - 1) // inf, max_c)
    r0 = ys // inf
    r1 = np.minimum((ys + p - 1) // inf, max_r)

    tiles: set[tuple[int, int]] = set()
    for ci0, ci1, ri0, ri1 in zip(c0.tolist(), c1.tolist(), r0.tolist(), r1.tolist()):
        for c in range(ci0, ci1 + 1):
            for r in range(ri0, ri1 + 1):
                tiles.add((c, r))

    coords_tiles = np.asarray(sorted(tiles), dtype=np.int64) if tiles else np.empty((0, 2), dtype=np.int64)
    return HistoPlusCoords(
        coords_tiles=coords_tiles,
        deepzoom_level=_full_res_deepzoom_level(tiling.level0_width, tiling.level0_height),
        tile_size=inf,
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
