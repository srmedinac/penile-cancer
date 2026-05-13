"""WSI-level handcrafted features from a cell table.

Two interpretable families, both built on the pruned-Delaunay cell graph:

* ``het__*``  -- nuclear morphological heterogeneity (successor to the
  "cellular diversity" features): how variable nuclear shape is within local
  cell neighbourhoods, and how morphologically distinct neighbouring nuclei of
  different types are.
* ``spatil__*`` -- spatial organisation of immune cells relative to tumour and
  stroma (successor to spaTIL): per-population spatial clusters, cluster-cluster
  overlap, nearest-neighbour co-localisation, and infiltration of tumour by
  immune cells, including how *heterogeneous* that infiltration is across the
  slide.

Plus a small block of ``comp__`` / ``morph_global__`` / ``morph_tumor__``
context features (abundances and bulk morphometry) that any survival model
wants as covariates and that cost nothing to compute.

Every call returns the *same* flat ``dict[str, float]`` schema regardless of
what is present in the slide (missing populations -> NaN), so a folder of
slides stacks straight into a tidy feature table.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shapely
from scipy import stats as _stats

from . import graph as G
from .morphology import MORPH_COLUMNS

# morphology channels used for heterogeneity (a deliberately small, low-redundancy set)
HET_FEATS = ["area", "eccentricity", "circularity", "solidity", "major_axis"]
# morphology channels reported in bulk and for the tumour compartment
GLOBAL_FEATS = ["area", "eccentricity", "circularity", "solidity", "major_axis"]
TUMOR_FEATS = ["area", "eccentricity", "circularity"]

ROLES = ["immune", "tumor", "stroma", "epithelial", "endothelium", "other"]
SPATIL_ROLES = ["immune", "tumor", "stroma"]
PAIRS = [("immune", "tumor"), ("immune", "stroma"), ("tumor", "stroma")]

_EPS = 1e-9


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _stat_block(values: np.ndarray, prefix: str, *, with_skew: bool = False, lo: int = 10) -> dict[str, float]:
    """mean / std / low-percentile / p90 (optionally skew) of a vector, NaN-safe.

    ``lo`` is the lower percentile to report (default 10; use 50 for zero-inflated
    quantities whose 10th percentile would be structurally zero).
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    keys = ["mean", "std", f"p{lo}", "p90"] + (["skew"] if with_skew else [])
    if v.size == 0:
        return {f"{prefix}__{k}": np.nan for k in keys}
    out = {
        f"{prefix}__mean": float(v.mean()),
        f"{prefix}__std": float(v.std()),
        f"{prefix}__p{lo}": float(np.percentile(v, lo)),
        f"{prefix}__p90": float(np.percentile(v, 90)),
    }
    if with_skew:
        out[f"{prefix}__skew"] = float(_stats.skew(v)) if v.size > 2 else 0.0
    return out


def _shannon(p: np.ndarray, axis: int = -1) -> np.ndarray:
    """Shannon entropy (nats) of probability rows; rows summing to ~0 -> 0."""
    p = np.where(p > 0, p, 0.0)
    s = p.sum(axis=axis, keepdims=True)
    p = np.divide(p, s, out=np.zeros_like(p), where=s > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -np.sum(np.where(p > 0, p * np.log(p), 0.0), axis=axis)
    return h


def _hull(xy: np.ndarray):
    """Convex-hull polygon of a point set (>=3 pts) else None."""
    if len(xy) < 3:
        return None
    h = shapely.convex_hull(shapely.MultiPoint(xy))
    return h if h.geom_type == "Polygon" and h.area > 0 else None


def _tissue_area_mm2(xy: np.ndarray, bin_um: float = 64.0) -> float:
    """Support area as occupied-cell-grid count x bin area (robust, O(n))."""
    if len(xy) == 0:
        return np.nan
    b = np.floor(xy / bin_um).astype(np.int64)
    n_bins = len(np.unique(b[:, 0] + 1j * b[:, 1]))
    return n_bins * (bin_um ** 2) / 1e6


# --------------------------------------------------------------------------- #
# feature blocks
# --------------------------------------------------------------------------- #
def _composition(df: pd.DataFrame, tissue_mm2: float) -> dict[str, float]:
    n = len(df)
    role_counts = df["role"].value_counts()
    fracs = {r: float(role_counts.get(r, 0)) / max(n, 1) for r in ROLES}
    out = {"comp__n_cells": float(n), "comp__density_per_mm2": n / (tissue_mm2 + _EPS) if np.isfinite(tissue_mm2) else np.nan}
    out |= {f"comp__{r}_frac": fracs[r] for r in ROLES}
    out["comp__tumor_immune_ratio"] = fracs["tumor"] / (fracs["immune"] + _EPS)
    out["comp__role_entropy"] = float(_shannon(np.array([fracs[r] for r in ROLES])))
    return out


def _morph_global(df: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    for f in GLOBAL_FEATS:
        out |= _stat_block(df[f].to_numpy(), f"morph_global__{f}", with_skew=True)
    tum = df.loc[df["role"] == "tumor"]
    for f in TUMOR_FEATS:
        out |= _stat_block(tum[f].to_numpy() if len(tum) >= 20 else np.array([]), f"morph_tumor__{f}")
    return out


def _heterogeneity(df: pd.DataFrame, cg: G.CellGraph) -> dict[str, float]:
    out: dict[str, float] = {}
    cell_types = list(df["cell_type"].cat.categories) if hasattr(df["cell_type"], "cat") else sorted(df["cell_type"].unique())
    # within-neighbourhood morphological dispersion, and cross-type contrast on edges
    ei, ej = cg.edges[:, 0], cg.edges[:, 1]
    types = df["cell_type"].to_numpy()
    diff_type = types[ei] != types[ej]
    for f in HET_FEATS:
        vals = df[f].to_numpy(dtype=float)
        v = np.where(np.isfinite(vals), vals, np.nan)
        # neighbour_dispersion ignores NaNs poorly, so impute per-feature median for this op only
        med = np.nanmedian(v) if np.isfinite(np.nanmedian(v)) else 0.0
        nd = G.neighbor_dispersion(cg, np.where(np.isfinite(v), v, med))
        out |= _stat_block(nd, f"het__{f}_nbhd_disp")
        if len(cg.edges):
            contrast = np.abs(vals[ei] - vals[ej])[diff_type]
            out |= _stat_block(contrast, f"het__{f}_heterotypic_contrast")
        else:
            out |= _stat_block(np.array([]), f"het__{f}_heterotypic_contrast")
    # local cell-type entropy (composition of the immediate neighbourhood)
    oh_type = G.onehot(df["cell_type"].to_numpy(), cell_types)
    nbhd_type = G.neighbor_mean(cg, oh_type, include_self=False)
    out |= _stat_block(_shannon(nbhd_type, axis=1), "het__local_type_entropy", lo=50)
    # local role mixing: 1 - dominant role fraction in the neighbourhood
    oh_role = G.onehot(df["role"].to_numpy(), ROLES)
    nbhd_role = G.neighbor_mean(cg, oh_role, include_self=False)
    valid = np.isfinite(nbhd_role).all(1)
    out["het__local_role_mixing__mean"] = float((1.0 - nbhd_role[valid].max(1)).mean()) if valid.any() else np.nan
    out["het__heterotypic_edge_frac"] = float(diff_type.mean()) if len(cg.edges) else np.nan
    return out


def _role_xy(df: pd.DataFrame, role: str) -> np.ndarray:
    return df.loc[df["role"] == role, ["cx", "cy"]].to_numpy(dtype=float)


def _cluster_summary(xy: np.ndarray, cls: list[np.ndarray]) -> dict:
    """Counts / sizes / hulls / areas for a population's spatial clusters."""
    hulls, areas_mm2, sizes = [], [], []
    for idx in cls:
        h = _hull(xy[idx])
        hulls.append(h)
        areas_mm2.append((h.area / 1e6) if h is not None else np.nan)
        sizes.append(len(idx))
    return {"hulls": hulls, "areas_mm2": np.array(areas_mm2), "sizes": np.array(sizes), "clusters": cls}


def _spatil(df: pd.DataFrame, cg: G.CellGraph, *, max_edge_um: float, min_cluster: int, near_um: float) -> dict[str, float]:
    out: dict[str, float] = {}
    xy_all = df[["cx", "cy"]].to_numpy(dtype=float)
    role_arr = df["role"].to_numpy()
    xy = {r: _role_xy(df, r) for r in SPATIL_ROLES}
    cl = {r: G.clusters(xy[r], max_edge_um, min_cluster) for r in SPATIL_ROLES}
    summ = {r: _cluster_summary(xy[r], cl[r]) for r in SPATIL_ROLES}

    # ---- per-population cluster aggregates --------------------------------- #
    for r in SPATIL_ROLES:
        s, n_r = summ[r], len(xy[r])
        in_cluster = sum(len(c) for c in s["clusters"])
        out[f"spatil__{r}__n_clusters"] = float(len(s["clusters"]))
        out[f"spatil__{r}__clustered_frac"] = in_cluster / (n_r + _EPS)
        out[f"spatil__{r}__mean_cluster_size"] = float(np.mean(s["sizes"])) if len(s["sizes"]) else np.nan
        out[f"spatil__{r}__largest_cluster_size"] = float(s["sizes"].max()) if len(s["sizes"]) else 0.0
        out[f"spatil__{r}__mean_cluster_area_mm2"] = float(np.nanmean(s["areas_mm2"])) if len(s["areas_mm2"]) else np.nan
        with np.errstate(invalid="ignore", divide="ignore"):
            dens = s["sizes"] / (s["areas_mm2"] * 1e6 + _EPS)  # cells per um^2
        out[f"spatil__{r}__mean_cluster_density"] = float(np.nanmean(dens)) if len(dens) else np.nan
    # immune-aggregate compactness (round, dense aggregates ~ tertiary lymphoid structures)
    comp = []
    for h in summ["immune"]["hulls"]:
        if h is not None and h.area > 0:
            comp.append(h.length ** 2 / (4.0 * np.pi * h.area))
    out["spatil__immune_cluster_compactness__mean"] = float(np.mean(comp)) if comp else np.nan
    out["spatil__largest_immune_cluster_area_mm2"] = (
        float(summ["immune"]["areas_mm2"][np.argmax(summ["immune"]["sizes"])]) if len(summ["immune"]["sizes"]) else 0.0
    )

    # ---- nearest-neighbour co-localisation between populations ------------- #
    for a, b in PAIRS:
        d = G.knn_distances(xy[b], xy[a], k=1) if len(xy[a]) else np.array([])
        out |= {
            f"spatil__{a}_to_{b}__nndist_um__median": float(np.median(d)) if d.size and np.isfinite(d).any() else np.nan,
            f"spatil__{a}_to_{b}__nndist_um__p10": float(np.percentile(d[np.isfinite(d)], 10)) if np.isfinite(d).any() else np.nan,
            f"spatil__{a}_to_{b}__nndist_um__p90": float(np.percentile(d[np.isfinite(d)], 90)) if np.isfinite(d).any() else np.nan,
            f"spatil__{a}_near_{b}__frac": float((d <= near_um).mean()) if d.size else np.nan,
        }

    # ---- cluster-cluster spatial overlap ---------------------------------- #
    for a, b in PAIRS:
        ha = [h for h in summ[a]["hulls"] if h is not None]
        hb = [h for h in summ[b]["hulls"] if h is not None]
        if ha and hb:
            tree = shapely.STRtree(hb)
            inter_frac, ratios = 0, []
            for h in ha:
                hit = tree.query(h, predicate="intersects")
                if len(hit):
                    inter_frac += 1
                    inter_area = shapely.union_all([shapely.intersection(h, hb[k]) for k in hit]).area
                    ratios.append(inter_area / (h.area + _EPS))
            out[f"spatil__{a}cl_intersect_{b}cl__frac"] = inter_frac / len(ha)
            out[f"spatil__{a}cl_overlap_{b}cl__mean_ratio"] = float(np.mean(ratios)) if ratios else 0.0
        else:
            out[f"spatil__{a}cl_intersect_{b}cl__frac"] = np.nan
            out[f"spatil__{a}cl_overlap_{b}cl__mean_ratio"] = np.nan

    # ---- infiltration of tumour by immune cells --------------------------- #
    nt, ns = out["spatil__immune_near_tumor__frac"], out["spatil__immune_near_stroma__frac"]
    out["spatil__intratumoral_vs_stromal_immune"] = nt / (ns + _EPS) if np.isfinite(nt) and np.isfinite(ns) else np.nan
    # per-tumour-cell immune fraction in the immediate neighbourhood
    oh_role = G.onehot(role_arr, ROLES)
    nbhd_role = G.neighbor_mean(cg, oh_role, include_self=False)
    imm_col, tum_mask = ROLES.index("immune"), (role_arr == "tumor")
    tum_imm = nbhd_role[tum_mask, imm_col]
    out |= _stat_block(tum_imm, "spatil__tumor_local_immune_frac", lo=50)
    deg_imm = (cg.adj @ (role_arr == "immune").astype(float))
    out["spatil__tumor_adj_immune_frac"] = float((deg_imm[tum_mask] > 0).mean()) if tum_mask.any() else np.nan

    # ---- homophily and immune-tumour assortativity (edge-level) ----------- #
    for r in SPATIL_ROLES:
        m = role_arr == r
        if m.any() and len(cg.edges):
            same = nbhd_role[m, ROLES.index(r)]
            out[f"spatil__{r}_homophily"] = float(np.nanmean(same))
        else:
            out[f"spatil__{r}_homophily"] = np.nan
    if len(cg.edges):
        ei, ej = cg.edges[:, 0], cg.edges[:, 1]
        ri, rj = role_arr[ei], role_arr[ej]
        n_it = int(np.sum(((ri == "immune") & (rj == "tumor")) | ((ri == "tumor") & (rj == "immune"))))
        p_i = float((role_arr == "immune").mean())
        p_t = float((role_arr == "tumor").mean())
        expected = 2.0 * p_i * p_t * len(cg.edges)
        out["spatil__immune_tumor_assortativity"] = float(np.log((n_it + _EPS) / (expected + _EPS)))
    else:
        out["spatil__immune_tumor_assortativity"] = np.nan

    # ---- immune dispersion: observed vs random nearest-neighbour spacing --- #
    if len(xy["immune"]) >= 3:
        nn = G.knn_distances(xy["immune"], xy["immune"], k=2)[:, 1]
        tissue_um2 = _tissue_area_mm2(xy_all) * 1e6
        expected_nn = 0.5 / np.sqrt(len(xy["immune"]) / (tissue_um2 + _EPS)) if np.isfinite(tissue_um2) else np.nan
        out["spatil__immune_dispersion_index"] = float(np.mean(nn) / (expected_nn + _EPS)) if np.isfinite(expected_nn) else np.nan
    else:
        out["spatil__immune_dispersion_index"] = np.nan
    return out


# --------------------------------------------------------------------------- #
# orchestrator
# --------------------------------------------------------------------------- #
def extract_features(
    cells: pd.DataFrame,
    *,
    max_edge_um: float = 50.0,
    min_cluster: int = 5,
    near_um: float = 20.0,
) -> dict[str, float]:
    """Compute the full WSI feature vector from a cell table (see :func:`pathomics.io.load_cells`).

    Parameters
    ----------
    max_edge_um : Delaunay edges longer than this (um) are pruned before anything else.
    min_cluster : minimum cells for a spatial cluster to count.
    near_um : distance (um) defining "adjacent" for co-localisation / infiltration fractions.
    """
    df = cells.reset_index(drop=True)
    xy = df[["cx", "cy"]].to_numpy(dtype=float)
    cg = G.build_graph(xy, max_edge_um=max_edge_um)
    tissue_mm2 = _tissue_area_mm2(xy)

    out: dict[str, float] = {}
    out["meta__n_cells"] = float(len(df))
    out["meta__tissue_area_mm2"] = float(tissue_mm2)
    out["meta__mean_degree"] = float(cg.degree.mean()) if cg.n else np.nan
    out |= _composition(df, tissue_mm2)
    out |= _morph_global(df)
    out |= _heterogeneity(df, cg)
    out |= _spatil(df, cg, max_edge_um=max_edge_um, min_cluster=min_cluster, near_um=near_um)
    return out


def extract_features_roi(cells: pd.DataFrame, polygon, **kwargs) -> dict[str, float]:
    """Same features, restricted to cells whose centroid falls in ``polygon``.

    ``polygon`` is a shapely geometry in micrometres (matching ``cx``/``cy``).
    """
    pts = shapely.points(cells["cx"].to_numpy(), cells["cy"].to_numpy())
    inside = shapely.contains(polygon, pts)
    return extract_features(cells.loc[inside], **kwargs)
