"""Spatial cell graphs and the operations the feature extractors build on.

The base graph is a Delaunay triangulation of the cell centroids with edges
longer than ``max_edge_um`` removed. Delaunay is the natural neighbour graph:
it has no scale parameter (unlike a fixed radius or a distance-decay cutoff),
adapts to local density, and stays planar/sparse at whole-slide scale. The
length cap just severs spurious links across tissue gaps.

A learned (GNN) graph constructor is deliberately *not* used here: "the most
informative graph" only has meaning relative to a downstream label, which a
feature-extraction step does not have. The neighbourhood-aggregation helpers
below are the fixed, unlearned analogue of a mean-aggregator GNN layer -- they
give the same kind of message-passing summary without any training.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.spatial import Delaunay, cKDTree


@dataclass
class CellGraph:
    """A symmetric sparse cell graph plus the coordinates it was built on."""

    xy: np.ndarray            # (n, 2) micrometres
    adj: sp.csr_matrix        # (n, n) binary, no self-loops
    edge_len: np.ndarray      # (n_edges,) micrometres, aligned with ``edges``
    edges: np.ndarray         # (n_edges, 2) i<j

    @property
    def n(self) -> int:
        return self.xy.shape[0]

    @property
    def degree(self) -> np.ndarray:
        return np.asarray(self.adj.sum(1)).ravel()


def _delaunay_edges(xy: np.ndarray) -> np.ndarray:
    """Unique undirected edges (i<j) of the Delaunay triangulation of ``xy``.

    Degenerate inputs (<3 points, all collinear / coincident) yield no edges.
    """
    if len(xy) < 3:
        return np.empty((0, 2), dtype=np.int64)
    try:
        tri = Delaunay(xy)
    except Exception:  # QhullError on collinear / coincident points
        return np.empty((0, 2), dtype=np.int64)
    s = tri.simplices
    e = np.vstack([s[:, [0, 1]], s[:, [1, 2]], s[:, [0, 2]]])
    e.sort(axis=1)
    return np.unique(e, axis=0)


def build_graph(xy: np.ndarray, max_edge_um: float = 50.0) -> CellGraph:
    """Delaunay graph of ``xy`` (micrometres) with edges > ``max_edge_um`` dropped."""
    xy = np.ascontiguousarray(xy, dtype=np.float64)
    edges = _delaunay_edges(xy)
    if len(edges):
        d = np.hypot(*(xy[edges[:, 0]] - xy[edges[:, 1]]).T)
        keep = d <= max_edge_um
        edges, d = edges[keep], d[keep]
    else:
        d = np.empty(0)
    n = len(xy)
    adj = sp.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
    adj = (adj + adj.T).tocsr()
    adj.data[:] = 1.0
    return CellGraph(xy=xy, adj=adj, edge_len=d, edges=edges)


def clusters(xy: np.ndarray, max_edge_um: float, min_size: int) -> list[np.ndarray]:
    """Connected components of the pruned Delaunay graph on ``xy``.

    Returns a list of index arrays (into ``xy``), only those with >= ``min_size``
    members, largest first. Used to turn a single cell population into spatial
    aggregates the way the original spaTIL did.
    """
    if len(xy) < min_size:
        return []
    g = build_graph(xy, max_edge_um)
    k, labels = connected_components(g.adj, directed=False)
    out = []
    for c in range(k):
        idx = np.flatnonzero(labels == c)
        if len(idx) >= min_size:
            out.append(idx)
    out.sort(key=len, reverse=True)
    return out


def neighbor_mean(graph: CellGraph, values: np.ndarray, *, include_self: bool = True) -> np.ndarray:
    """Mean of ``values`` over each node's graph neighbourhood (vectorised).

    ``values`` may be 1-D ``(n,)`` or 2-D ``(n, k)``; isolated nodes keep their
    own value (or NaN if ``include_self`` is False).
    """
    A = graph.adj
    if include_self:
        A = A + sp.eye(graph.n, format="csr")
    deg = np.asarray(A.sum(1)).ravel()
    deg_safe = np.where(deg == 0, np.nan, deg)
    summed = A @ values
    if values.ndim == 1:
        return summed / deg_safe
    return summed / deg_safe[:, None]


def neighbor_dispersion(graph: CellGraph, values: np.ndarray) -> np.ndarray:
    """Within-neighbourhood standard deviation of ``values`` per node (incl. self)."""
    m1 = neighbor_mean(graph, values)
    m2 = neighbor_mean(graph, values ** 2)
    return np.sqrt(np.maximum(m2 - m1 ** 2, 0.0))


def onehot(labels, categories: list[str]) -> np.ndarray:
    """``(n, len(categories))`` indicator matrix for categorical labels."""
    codes = pd.Categorical(labels, categories=categories).codes
    out = np.zeros((len(codes), len(categories)), dtype=np.float64)
    valid = codes >= 0
    out[valid, codes[valid]] = 1.0
    return out


def knn_distances(xy: np.ndarray, query_xy: np.ndarray, k: int = 1) -> np.ndarray:
    """Distance(s) from each point in ``query_xy`` to its ``k`` nearest in ``xy``.

    Returns ``(m,)`` for ``k == 1`` else ``(m, k)``. Empty ``xy`` -> all inf.
    """
    if len(xy) == 0:
        return np.full(len(query_xy) if k == 1 else (len(query_xy), k), np.inf)
    tree = cKDTree(xy)
    d, _ = tree.query(query_xy, k=k)
    return d
