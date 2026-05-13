"""Load cell segmentations into a tidy per-cell table (the ``CellTable``).

Supported inputs:
  * histoplus ``cell_masks.json`` (tile grid, tile-local polygon coordinates,
    carries ``inference_mpp``) -- the canonical source.
  * QuPath GeoJSON ``FeatureCollection`` of detection polygons -- ``mpp`` must
    be supplied (or read from a sibling ``cell_masks.json``).

The first load parses the (potentially multi-GB) file, derives per-nucleus
morphology, and caches a compact ``cells.parquet`` next to it; later loads read
the parquet in seconds. Coordinates are stored in micrometres.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import morphology

# Cell type -> tissue role. Edit this (or pass your own) for a new cohort;
# nothing else in the pipeline hard-codes cell-type names.
DEFAULT_ROLES: dict[str, str] = {
    "Lymphocytes": "immune",
    "Plasmocytes": "immune",
    "Macrophages": "immune",
    "Neutrophils": "immune",
    "Eosinophils": "immune",
    "Cancer cell": "tumor",
    "Epithelial": "epithelial",
    "Fibroblasts": "stroma",
    "Minor Stromal Cell": "stroma",
    "Muscle Cell": "stroma",
    "Endothelial Cell": "endothelium",
    "Apoptotic Body": "other",
    "Mitotic Figures": "tumor",
    "Red blood cell": "other",
}

CACHE_NAME = "cells.parquet"


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #
def _read_mpp(path: Path) -> float | None:
    """Pull ``inference_mpp`` from a histoplus json header (or a sibling one)."""
    for candidate in (path, path.with_name("cell_masks.json"), path.with_name("cell_masks.geojson")):
        if not candidate.exists():
            continue
        with open(candidate, "rb") as fh:
            head = fh.read(4096).decode("utf-8", "ignore")
        i = head.find('"inference_mpp"')
        if i != -1:
            try:
                return float(head[i:].split(":", 1)[1].split(",", 1)[0])
            except (IndexError, ValueError):
                pass
    return None


def _parse(path: Path):
    """Return ``(coords, offsets, cell_types, confidences)`` for all cells.

    ``coords`` is the global-pixel vertex array ``(M, 2)`` (no repeated closing
    vertex), ``offsets`` the length-``n+1`` per-cell index array. Handles the
    histoplus ``cell_masks.json`` (tile grid + tile-local coords) and a QuPath
    GeoJSON ``FeatureCollection`` of detection polygons.
    """
    with open(path) as fh:
        doc = json.load(fh)
    flat: list = []          # vertices, [[x, y], ...]
    ring_len: list[int] = []
    tile_off: list = []      # (ox, oy) per cell, added back vectorised below
    types: list[str] = []
    confs: list[float] = []

    if isinstance(doc, dict) and "cell_masks" in doc:
        for tile in doc["cell_masks"]:
            ox = float(tile["x"]) * float(tile["width"])
            oy = float(tile["y"]) * float(tile["height"])
            for cell in tile.get("masks", ()):
                c = cell.get("coordinates")
                if not c or len(c) < 3:
                    continue
                if c[0] == c[-1]:
                    c = c[:-1]
                flat.extend(c)
                ring_len.append(len(c))
                tile_off.append((ox, oy))
                types.append(cell.get("cell_type", "Unknown"))
                confs.append(float(cell.get("confidence", 1.0)))
    elif isinstance(doc, dict) and doc.get("type") == "FeatureCollection":
        for feat in doc.get("features", ()):
            geom = feat.get("geometry") or {}
            if geom.get("type") != "Polygon":
                continue
            c = geom["coordinates"][0]
            if not c or len(c) < 3:
                continue
            if c[0] == c[-1]:
                c = c[:-1]
            if len(c) < 3:
                continue
            flat.extend(c)
            ring_len.append(len(c))
            tile_off.append((0.0, 0.0))
            props = feat.get("properties") or {}
            types.append((props.get("classification") or {}).get("name", "Unknown"))
            confs.append(float((props.get("measurements") or {}).get("confidence", 1.0)))
    else:
        raise ValueError(f"unrecognised cell file: {path}")

    if not ring_len:
        return np.zeros((0, 2)), np.array([0]), [], []
    coords = np.asarray(flat, dtype=np.float64)
    ring_len = np.asarray(ring_len, dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(ring_len)])
    coords += np.repeat(np.asarray(tile_off, dtype=np.float64), ring_len, axis=0)
    return coords, offsets, types, confs


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def load_cells(
    path: str | Path,
    *,
    mpp: float | None = None,
    roles: dict[str, str] | None = None,
    min_confidence: float = 0.0,
    cache: bool = True,
    rebuild: bool = False,
) -> pd.DataFrame:
    """Return a per-cell DataFrame with columns:

    ``cell_type, role, confidence, cx, cy`` (micrometres) and the morphology
    columns in :data:`pathomics.morphology.MORPH_COLUMNS` (um / um^2 / unitless).

    Parameters
    ----------
    mpp : microns per pixel; required for GeoJSON input that has no sibling json.
    roles : cell-type -> role mapping; defaults to :data:`DEFAULT_ROLES`.
    min_confidence : drop cells below this detector confidence.
    cache / rebuild : read / (re)write ``cells.parquet`` beside the input file.
    """
    path = Path(path)
    roles = roles or DEFAULT_ROLES
    cache_path = path.with_name(CACHE_NAME)

    if cache and not rebuild and cache_path.exists() and cache_path.stat().st_mtime >= path.stat().st_mtime:
        df = pd.read_parquet(cache_path)
    else:
        if mpp is None:
            mpp = _read_mpp(path)
        if mpp is None:
            raise ValueError("mpp not given and no inference_mpp found; pass mpp=")
        coords, offsets, types, confs = _parse(path)
        centroids, morph = morphology.describe_flat(coords, offsets, mpp)
        df = pd.DataFrame({"cell_type": pd.Categorical(types), "confidence": np.asarray(confs, np.float32),
                           "cx": centroids[:, 0], "cy": centroids[:, 1], **morph})
        df["role"] = pd.Categorical([roles.get(t, "other") for t in df["cell_type"]])
        df = df[["cell_type", "role", "confidence", "cx", "cy", *morphology.MORPH_COLUMNS]]
        if cache:
            df.to_parquet(cache_path, index=False)

    if "role" not in df or set(df["role"].cat.categories) != set(roles.values()) | {"other"}:
        df["role"] = pd.Categorical([roles.get(t, "other") for t in df["cell_type"]])
    if min_confidence > 0:
        df = df[df["confidence"] >= min_confidence].reset_index(drop=True)
    return df
