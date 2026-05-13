"""Vectorised nuclear morphology descriptors from polygon rings.

All shape descriptors come from closed-form polygon moments (no rasterisation,
no image), so a few million nuclei are processed in one pass. Lengths are
returned in micrometres and areas in micrometres-squared given a pixel size
``mpp`` (microns per pixel).
"""
from __future__ import annotations

import numpy as np

# Descriptor columns produced by :func:`describe`, in order.
MORPH_COLUMNS = [
    "area",            # um^2
    "perimeter",       # um
    "equiv_diameter",  # um, diameter of a circle of equal area
    "major_axis",      # um, ellipse-equivalent major axis length
    "minor_axis",      # um, ellipse-equivalent minor axis length
    "eccentricity",    # 0 = circle, ->1 = elongated
    "orientation",     # radians in (-pi/2, pi/2], major-axis angle
    "circularity",     # 4*pi*area / perimeter^2, 1 = perfect circle
    "extent",          # area / axis-aligned bounding-box area
    "solidity",        # area / convex-hull area, 1 = convex
]


def _flatten_rings(rings: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate polygon rings into one (M, 2) array + per-polygon offsets.

    The closing vertex (a repeat of the first) is dropped from each ring.
    Returns ``(coords, offsets)`` where ``offsets`` has length ``len(rings)+1``.
    """
    parts, lengths = [], []
    for r in rings:
        a = np.asarray(r, dtype=np.float64)
        if len(a) >= 2 and np.array_equal(a[0], a[-1]):
            a = a[:-1]
        parts.append(a)
        lengths.append(len(a))
    coords = np.concatenate(parts, axis=0) if parts else np.zeros((0, 2))
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    return coords, offsets


def describe(
    rings: list[np.ndarray], mpp: float, *, with_solidity: bool = True
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Describe a list of polygon rings (closing vertices are handled either way).

    Returns ``(centroids_um, descriptors)`` -- ``centroids_um`` is an ``(n, 2)``
    array of area-centroids in micrometres, ``descriptors`` a dict of per-polygon
    arrays keyed by :data:`MORPH_COLUMNS`.
    """
    coords, off = _flatten_rings(rings)
    return describe_flat(coords, off, mpp, with_solidity=with_solidity)


def describe_flat(
    coords: np.ndarray, off: np.ndarray, mpp: float, *, with_solidity: bool = True
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Like :func:`describe` but on pre-flattened input: ``coords`` is ``(M, 2)``
    and ``off`` is the length-``n+1`` offsets array (no repeated closing vertex).
    """
    n = len(off) - 1
    if n == 0:
        return np.zeros((0, 2), np.float32), {c: np.zeros(0, np.float32) for c in MORPH_COLUMNS}

    x, y = coords[:, 0], coords[:, 1]
    start, stop = off[:-1], off[1:]                       # per-polygon vertex range
    nxt = np.arange(len(coords)) + 1
    nxt[stop - 1] = start                                 # wrap last vertex -> first
    xn, yn = x[nxt], y[nxt]

    cross = x * yn - xn * y                                # shoelace term per edge
    seg = np.hypot(xn - x, yn - y)

    def psum(v):                                           # per-polygon sum
        return np.add.reduceat(v, start)

    sa = 0.5 * psum(cross)                                 # signed area (px^2)
    sa_safe = np.where(sa == 0.0, np.nan, sa)
    area = np.abs(sa)
    perimeter = psum(seg)

    cx = psum((x + xn) * cross) / (6.0 * sa_safe)
    cy = psum((y + yn) * cross) / (6.0 * sa_safe)

    # area moments about the origin, normalised by signed area -> E[x^2] etc.
    exx = psum((x * x + x * xn + xn * xn) * cross) / (12.0 * sa_safe)
    eyy = psum((y * y + y * yn + yn * yn) * cross) / (12.0 * sa_safe)
    exy = psum((x * yn + 2.0 * x * y + 2.0 * xn * yn + xn * y) * cross) / (24.0 * sa_safe)
    cxx = exx - cx * cx                                    # covariance of the area distribution
    cyy = eyy - cy * cy
    cxy = exy - cx * cy

    tr = cxx + cyy
    det = cxx * cyy - cxy * cxy
    disc = np.sqrt(np.maximum(0.25 * tr * tr - det, 0.0))
    l1 = np.maximum(0.5 * tr + disc, 0.0)                  # larger eigenvalue
    l2 = np.maximum(0.5 * tr - disc, 0.0)
    # uniform ellipse with semi-axes (a, b): eigenvalues are a^2/4, b^2/4
    major_axis = 4.0 * np.sqrt(l1)
    minor_axis = 4.0 * np.sqrt(l2)
    with np.errstate(invalid="ignore", divide="ignore"):
        eccentricity = np.sqrt(np.clip(1.0 - l2 / l1, 0.0, 1.0))
    orientation = 0.5 * np.arctan2(2.0 * cxy, cxx - cyy)

    xmin, xmax = np.minimum.reduceat(x, start), np.maximum.reduceat(x, start)
    ymin, ymax = np.minimum.reduceat(y, start), np.maximum.reduceat(y, start)
    bbox_area = np.maximum((xmax - xmin) * (ymax - ymin), 1e-9)
    extent = area / bbox_area
    equiv_diameter = 2.0 * np.sqrt(area / np.pi)
    with np.errstate(invalid="ignore", divide="ignore"):
        circularity = 4.0 * np.pi * area / np.maximum(perimeter * perimeter, 1e-9)

    if with_solidity:
        solidity = _solidity(coords, off, area)
    else:
        solidity = np.full(n, np.nan)

    px2um, px2um2 = float(mpp), float(mpp) ** 2
    out = {
        "area": area * px2um2,
        "perimeter": perimeter * px2um,
        "equiv_diameter": equiv_diameter * px2um,
        "major_axis": major_axis * px2um,
        "minor_axis": minor_axis * px2um,
        "eccentricity": eccentricity,
        "orientation": orientation,
        "circularity": np.clip(circularity, 0.0, 1.0),
        "extent": np.clip(extent, 0.0, 1.0),
        "solidity": np.clip(solidity, 0.0, 1.0),
    }
    # degenerate polygons (zero / near-zero area) -> NaN, dropped or imputed downstream
    bad = ~np.isfinite(sa) | (area < 1.0)
    for k, v in out.items():
        vv = v.astype(np.float32)
        vv[bad] = np.nan
        out[k] = vv
    # fall back to the vertex mean where the area centroid is undefined
    vmean = np.column_stack([np.add.reduceat(x, start), np.add.reduceat(y, start)]) / np.diff(off)[:, None]
    cxy = np.column_stack([cx, cy])
    cxy[bad] = vmean[bad]
    centroids_um = (cxy * px2um).astype(np.float32)
    return centroids_um, out


def _solidity(coords: np.ndarray, off: np.ndarray, area: np.ndarray) -> np.ndarray:
    """area / convex-hull area, computed with vectorised shapely."""
    n = len(off) - 1
    try:
        import shapely

        idx = np.repeat(np.arange(n), np.diff(off))
        rings = shapely.linearrings(coords, indices=idx)
        hull_area = shapely.area(shapely.convex_hull(shapely.polygons(rings)))
    except Exception:
        return np.full(n, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        return area / np.where(hull_area > 0, hull_area, np.nan)
