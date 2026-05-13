"""Lightweight overlays for sanity-checking features at WSI / ROI level.

Everything works on the cell table alone (centroids in micrometres). If
``openslide`` is available and a slide path is given, plots are drawn over a
low-resolution thumbnail; otherwise over a white background. Matplotlib only.
"""
from __future__ import annotations

import numpy as np

from . import graph as G

# clean style: no grid, no greyed text, teal/orange accents
_TEAL, _ORANGE = "#1b9e8a", "#e07b39"
ROLE_COLORS = {
    "immune": _TEAL, "tumor": _ORANGE, "stroma": "#7a7a7a",
    "epithelial": "#9467bd", "endothelium": "#2c7fb8", "other": "#cccccc",
}


def _new_ax(ax, xy, slide_path=None):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 9))
    extent = None
    if slide_path is not None:
        try:
            import openslide

            sl = openslide.OpenSlide(str(slide_path))
            mpp = float(sl.properties.get(openslide.PROPERTY_NAME_MPP_X, 0.25))
            thumb_w = 2000
            scale = sl.dimensions[0] / thumb_w
            thumb = sl.get_thumbnail((thumb_w, int(sl.dimensions[1] / scale)))
            w_um, h_um = sl.dimensions[0] * mpp, sl.dimensions[1] * mpp
            ax.imshow(thumb, extent=(0, w_um, h_um, 0))
            extent = (0, w_um, h_um, 0)
        except Exception:
            pass
    ax.set_aspect("equal")
    ax.invert_yaxis() if extent is None else None
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.grid(False)
    return ax


def plot_cells(cells, *, by="role", ax=None, slide_path=None, s=1.0):
    """Scatter of cell centroids coloured by ``role`` (or any column)."""
    ax = _new_ax(ax, cells[["cx", "cy"]].to_numpy(), slide_path)
    if by == "role":
        for r, sub in cells.groupby("role", observed=True):
            ax.scatter(sub["cx"], sub["cy"], s=s, c=ROLE_COLORS.get(r, "#999999"), label=r, linewidths=0)
        ax.legend(markerscale=6, frameon=False, loc="upper right")
    else:
        sc = ax.scatter(cells["cx"], cells["cy"], s=s, c=cells[by], cmap="viridis", linewidths=0)
        ax.figure.colorbar(sc, ax=ax, shrink=0.7, label=by)
    return ax


def plot_graph(cells, *, max_edge_um=50.0, ax=None, slide_path=None, edge_alpha=0.15, max_edges=200_000):
    """Draw the pruned-Delaunay cell graph (subsampled if very large)."""
    from matplotlib.collections import LineCollection

    xy = cells[["cx", "cy"]].to_numpy(dtype=float)
    cg = G.build_graph(xy, max_edge_um=max_edge_um)
    ax = _new_ax(ax, xy, slide_path)
    e = cg.edges
    if len(e) > max_edges:
        e = e[np.random.default_rng(0).choice(len(e), max_edges, replace=False)]
    ax.add_collection(LineCollection(xy[e], colors="#444444", linewidths=0.2, alpha=edge_alpha))
    for r, sub in cells.groupby("role", observed=True):
        ax.scatter(sub["cx"], sub["cy"], s=1.0, c=ROLE_COLORS.get(r, "#999999"), label=r, linewidths=0)
    ax.legend(markerscale=6, frameon=False, loc="upper right")
    return ax


def plot_clusters(cells, role, *, max_edge_um=50.0, min_cluster=5, ax=None, slide_path=None):
    """Overlay convex hulls of a population's spatial clusters on all cells."""
    import shapely
    from matplotlib.patches import Polygon as MplPoly

    ax = plot_cells(cells, ax=ax, slide_path=slide_path, s=0.6)
    xy = cells.loc[cells["role"] == role, ["cx", "cy"]].to_numpy(dtype=float)
    for idx in G.clusters(xy, max_edge_um, min_cluster):
        if len(idx) < 3:
            continue
        h = shapely.convex_hull(shapely.MultiPoint(xy[idx]))
        if h.geom_type == "Polygon":
            ax.add_patch(MplPoly(np.asarray(h.exterior.coords), closed=True, fill=False,
                                 edgecolor=ROLE_COLORS.get(role, "k"), linewidth=1.4))
    ax.set_title(f"{role} clusters (n={len(G.clusters(xy, max_edge_um, min_cluster))})")
    return ax


def plot_heterogeneity(cells, feature="area", *, max_edge_um=50.0, ax=None, slide_path=None):
    """Colour each cell by the within-neighbourhood dispersion of ``feature``."""
    xy = cells[["cx", "cy"]].to_numpy(dtype=float)
    cg = G.build_graph(xy, max_edge_um=max_edge_um)
    v = cells[feature].to_numpy(dtype=float)
    med = np.nanmedian(v)
    nd = G.neighbor_dispersion(cg, np.where(np.isfinite(v), v, med if np.isfinite(med) else 0.0))
    ax = _new_ax(ax, xy, slide_path)
    vmax = np.nanpercentile(nd, 98)
    sc = ax.scatter(xy[:, 0], xy[:, 1], s=1.2, c=nd, cmap="inferno", vmax=vmax, linewidths=0)
    ax.figure.colorbar(sc, ax=ax, shrink=0.7, label=f"{feature} neighbourhood dispersion")
    return ax


def save(ax, path, dpi=600):
    ax.figure.tight_layout()
    ax.figure.savefig(path, dpi=dpi, bbox_inches="tight")
