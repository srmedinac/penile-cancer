"""QuPath-compatible GeoJSON exports for `pathomics` results.

Three overlays you can drag onto a slide in QuPath (or load via *File → Import
objects*). All outputs are GeoJSON ``FeatureCollection``s in **level-0 pixel
coordinates** (QuPath's native space).

* :func:`export_cluster_hulls` -- convex hull of every spaTIL cluster per role,
  as ``Polygon`` annotations classified by role (`immune_cluster`,
  `tumor_cluster`, `stroma_cluster`). The WSI-level spaTIL view.
* :func:`export_interaction_graph` -- Delaunay edges connecting cells of two
  roles only (default immune-tumor) as ``LineString`` annotations. The spatial
  interaction graph at slide scale.
* :func:`export_cells_with_measurements` -- re-emits the cell_masks polygons
  as QuPath ``detection``\s with per-cell measurements baked in
  (`neighborhood_immune_frac`, `area_local_sd_um2`, `neighborhood_mixedness`,
  …) so QuPath's *Measurement Maps* paints heatmaps over the cells.

CLI::

    python -m pathomics.qupath histoplus_output --clusters --interactions --measurements
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import shapely

from . import graph as G
from . import io as _io
from .io import DEFAULT_ROLES, load_cells

ROLES = ["immune", "tumor", "stroma", "epithelial", "endothelium", "other"]

# RGB colours (0-255) for QuPath classifications.
CLUSTER_COLOR = {"immune": [27, 158, 138], "tumor": [224, 123, 57], "stroma": [122, 122, 122]}
EDGE_COLOR = {
    frozenset(("immune", "tumor")):  [155,  93, 229],
    frozenset(("immune", "stroma")): [  0, 187, 249],
    frozenset(("tumor",  "stroma")): [181, 201,  40],
}
CELL_COLOR = {
    "Lymphocytes":        [  0, 128, 255],
    "Plasmocytes":        [128,   0, 255],
    "Macrophages":        [255,   0, 128],
    "Neutrophils":        [255, 255,   0],
    "Eosinophils":        [255,  64,  64],
    "Cancer cell":        [220,  20,  60],
    "Epithelial":         [200, 200, 200],
    "Fibroblasts":        [255, 165,   0],
    "Minor Stromal Cell": [160, 120,  90],
    "Muscle Cell":        [128,  64,   0],
    "Endothelial Cell":   [  0, 200, 200],
    "Apoptotic Body":     [ 80,  80,  80],
    "Red blood cell":     [180,   0,   0],
    "Mitotic Figures":    [255,   0,   0],
}
_DEFAULT_COLOR = [150, 150, 150]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _resolve_mpp(mpp: float | None, src_path: str | Path | None) -> float:
    if mpp is not None:
        return float(mpp)
    if src_path is None:
        raise ValueError("pass mpp= or src_path= so pixel coords can be derived")
    m = _io._read_mpp(Path(src_path))
    if m is None:
        raise ValueError(f"no inference_mpp found near {src_path}; pass mpp=")
    return float(m)


def _write_features(features: list[dict], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write('{"type":"FeatureCollection","features":[')
        for i, feat in enumerate(features):
            if i:
                f.write(",")
            json.dump(feat, f)
        f.write("]}")
    return out_path


# --------------------------------------------------------------------------- #
# exports
# --------------------------------------------------------------------------- #
def export_cluster_hulls(
    cells,
    out_path: str | Path,
    *,
    mpp: float | None = None,
    src_path: str | Path | None = None,
    roles: tuple[str, ...] = ("immune", "tumor", "stroma"),
    max_edge_um: float = 50.0,
    min_cluster: int = 5,
) -> Path:
    """Convex hull of each spatial cluster as a QuPath annotation polygon.

    Coordinates are written in level-0 pixels (``cells.cx/cy / mpp``). Provide
    ``mpp`` or ``src_path`` (the sibling ``cell_masks.json`` header is read).
    """
    mpp = _resolve_mpp(mpp, src_path)
    features = []
    for role in roles:
        m = (cells["role"] == role).to_numpy()
        if not m.any():
            continue
        xy_um = cells.loc[m, ["cx", "cy"]].to_numpy(dtype=float)
        for cl in G.clusters(xy_um, max_edge_um, min_cluster):
            pts_um = xy_um[cl]
            hull_um = shapely.convex_hull(shapely.MultiPoint(pts_um))
            if hull_um.geom_type != "Polygon" or hull_um.area <= 0:
                continue
            hull_px = shapely.convex_hull(shapely.MultiPoint(pts_um / mpp))
            ring = [list(c) for c in hull_px.exterior.coords]
            features.append({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [ring]},
                "properties": {
                    "objectType": "annotation",
                    "classification": {
                        "name": f"{role}_cluster",
                        "color": CLUSTER_COLOR.get(role, _DEFAULT_COLOR),
                    },
                    "measurements": {
                        "n_cells": int(len(cl)),
                        "area_um2": float(hull_um.area),
                    },
                },
            })
    return _write_features(features, out_path)


def export_interaction_graph(
    cells,
    out_path: str | Path,
    *,
    role_a: str = "immune",
    role_b: str = "tumor",
    mpp: float | None = None,
    src_path: str | Path | None = None,
    max_edge_um: float = 50.0,
    max_edges: int = 100_000,
) -> Path:
    """Delaunay edges connecting cells of two roles as ``LineString`` annotations.

    Subsamples uniformly down to ``max_edges`` when more exist. Useful for
    visualising the immune-tumour spatial interaction graph at WSI scale.
    """
    mpp = _resolve_mpp(mpp, src_path)
    xy_um = cells[["cx", "cy"]].to_numpy(dtype=float)
    cg = G.build_graph(xy_um, max_edge_um=max_edge_um)
    role = cells["role"].to_numpy()
    if len(cg.edges) == 0:
        return _write_features([], out_path)
    ei, ej = cg.edges[:, 0], cg.edges[:, 1]
    mask = ((role[ei] == role_a) & (role[ej] == role_b)) | \
           ((role[ei] == role_b) & (role[ej] == role_a))
    e = cg.edges[mask]
    if len(e) > max_edges:
        rng = np.random.default_rng(0)
        e = e[rng.choice(len(e), max_edges, replace=False)]

    color = EDGE_COLOR.get(frozenset((role_a, role_b)), _DEFAULT_COLOR)
    name = f"{role_a}_{role_b}_edge"
    xy_px = xy_um / mpp
    features = [
        {
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": [xy_px[a].tolist(), xy_px[b].tolist()]},
            "properties": {
                "objectType": "annotation",
                "classification": {"name": name, "color": color},
            },
        }
        for a, b in e
    ]
    return _write_features(features, out_path)


def export_cells_with_measurements(
    src_path: str | Path,
    out_path: str | Path,
    *,
    roles: dict[str, str] | None = None,
    max_edge_um: float = 50.0,
) -> Path:
    """Re-emit the cell_masks polygons as QuPath detections with measurements.

    Each cell becomes a detection classified by cell type, carrying both its
    polygon morphology (area, eccentricity, circularity, solidity, confidence)
    and three neighbourhood-derived measurements that drive heterogeneity
    heatmaps in QuPath's *Measurement Maps*:

    * ``neighborhood_immune_frac`` / ``neighborhood_tumor_frac`` -- role
      fractions in each cell's 1-hop Delaunay neighbourhood.
    * ``neighborhood_mixedness`` -- ``1 − max_role_fraction`` in the
      neighbourhood (0 = pure, 1 = maximally mixed).
    * ``<channel>_local_sd_um2`` -- within-neighbourhood SD of `area`,
      `eccentricity`, `circularity`.

    Loads with ``min_confidence=0`` to keep the polygon-to-cell ordering
    aligned with the source file.
    """
    src_path = Path(src_path)
    cells = load_cells(src_path, min_confidence=0.0, roles=roles or DEFAULT_ROLES)
    coords, offsets, types, confs = _io._parse(src_path)
    n = len(offsets) - 1
    if n != len(cells):
        raise RuntimeError(f"polygon/cell count mismatch: {n} vs {len(cells)}")

    cg = G.build_graph(cells[["cx", "cy"]].to_numpy(dtype=float), max_edge_um=max_edge_um)
    oh_role = G.onehot(cells["role"].to_numpy(), ROLES)
    nbhd_role = G.neighbor_mean(cg, oh_role, include_self=False)
    immune_frac = nbhd_role[:, ROLES.index("immune")]
    tumor_frac = nbhd_role[:, ROLES.index("tumor")]
    with warnings.catch_warnings():  # nanmax warns on all-NaN rows (isolated cells)
        warnings.simplefilter("ignore", RuntimeWarning)
        mixedness = 1.0 - np.nanmax(nbhd_role, axis=1)

    local_sds: dict[str, np.ndarray] = {}
    for ch in ("area", "eccentricity", "circularity"):
        v = cells[ch].to_numpy(dtype=float)
        med = np.nanmedian(v)
        v_filled = np.where(np.isfinite(v), v, med if np.isfinite(med) else 0.0)
        local_sds[ch] = G.neighbor_dispersion(cg, v_filled)

    cols = {ch: cells[ch].to_numpy() for ch in ("area", "eccentricity", "circularity", "solidity")}

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write('{"type":"FeatureCollection","features":[')
        first = True
        for i in range(n):
            ring = [list(c) for c in coords[offsets[i]:offsets[i + 1]].tolist()]
            if ring[0] != ring[-1]:
                ring.append(ring[0])
            t = types[i]
            feat = {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [ring]},
                "properties": {
                    "objectType": "detection",
                    "classification": {"name": t, "color": CELL_COLOR.get(t, _DEFAULT_COLOR)},
                    "measurements": {
                        "confidence":               float(confs[i]),
                        "area_um2":                 _safe(cols["area"][i]),
                        "eccentricity":             _safe(cols["eccentricity"][i]),
                        "circularity":              _safe(cols["circularity"][i]),
                        "solidity":                 _safe(cols["solidity"][i]),
                        "neighborhood_immune_frac": _safe(immune_frac[i]),
                        "neighborhood_tumor_frac":  _safe(tumor_frac[i]),
                        "neighborhood_mixedness":   _safe(mixedness[i]),
                        "area_local_sd_um2":        _safe(local_sds["area"][i]),
                        "eccentricity_local_sd":    _safe(local_sds["eccentricity"][i]),
                        "circularity_local_sd":     _safe(local_sds["circularity"][i]),
                    },
                },
            }
            if not first:
                f.write(",")
            json.dump(feat, f)
            first = False
        f.write("]}")
    return out_path


def _safe(x) -> float:
    f = float(x)
    return f if np.isfinite(f) else 0.0


# --------------------------------------------------------------------------- #
# batch CLI
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pathomics.qupath", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="folder of slide subdirectories (each with cell_masks.json)")
    ap.add_argument("--pattern", default="cell_masks.json", help="cell file name inside each subdir")
    ap.add_argument("--clusters", action="store_true", help="write qupath_clusters.geojson per slide")
    ap.add_argument("--interactions", action="store_true",
                    help="write qupath_edges_immune_tumor.geojson per slide")
    ap.add_argument("--measurements", action="store_true",
                    help="write qupath_cells.geojson with per-cell measurements per slide")
    ap.add_argument("--min-confidence", type=float, default=0.0)
    ap.add_argument("--max-edge-um", type=float, default=50.0)
    ap.add_argument("--min-cluster", type=int, default=5)
    ap.add_argument("--max-edges", type=int, default=100_000)
    ap.add_argument("--edge-roles", nargs=2, default=("immune", "tumor"),
                    metavar=("ROLE_A", "ROLE_B"))
    args = ap.parse_args(argv)

    if not (args.clusters or args.interactions or args.measurements):
        ap.error("pick at least one of --clusters / --interactions / --measurements")

    slides = sorted(p for p in args.root.iterdir() if p.is_dir() and (p / args.pattern).exists())
    if not slides:
        print(f"no '{args.pattern}' found under {args.root}", file=sys.stderr)
        return 1
    print(f"{len(slides)} slides")

    for slide in slides:
        t0 = time.time()
        src = slide / args.pattern
        out_clusters = slide / "qupath_clusters.geojson"
        out_edges = slide / f"qupath_edges_{args.edge_roles[0]}_{args.edge_roles[1]}.geojson"
        out_cells = slide / "qupath_cells.geojson"

        cells = None
        if args.clusters or args.interactions:
            cells = load_cells(src, min_confidence=args.min_confidence)
        if args.clusters:
            export_cluster_hulls(cells, out_clusters, src_path=src,
                                 max_edge_um=args.max_edge_um, min_cluster=args.min_cluster)
        if args.interactions:
            export_interaction_graph(cells, out_edges, src_path=src,
                                     role_a=args.edge_roles[0], role_b=args.edge_roles[1],
                                     max_edge_um=args.max_edge_um, max_edges=args.max_edges)
        if args.measurements:
            export_cells_with_measurements(src, out_cells, max_edge_um=args.max_edge_um)
        print(f"  {slide.name}: {time.time() - t0:6.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
