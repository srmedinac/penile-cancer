"""``pathomics`` -- WSI-level handcrafted features from cell segmentations.

Two interpretable feature families on top of a pruned-Delaunay cell graph:
nuclear morphological heterogeneity (``diversity__*``) and immune spatial
organisation / spaTIL v2 (``spatil__*``), plus composition/morphometry
covariates. Cancer-agnostic: the only domain knowledge is the cell-type ->
role mapping in :data:`pathomics.io.DEFAULT_ROLES`.

Typical use::

    from pathomics import load_cells, extract_features
    cells = load_cells("histoplus_output/MPe08B/cell_masks.geojson", min_confidence=0.5)
    feats = extract_features(cells)

or batch a folder from the command line::

    python -m pathomics histoplus_output -o features.parquet --min-confidence 0.5
"""
from .io import DEFAULT_ROLES, load_cells
from .features import extract_features, extract_features_roi
from . import graph, morphology, viz

__all__ = [
    "load_cells", "DEFAULT_ROLES",
    "extract_features", "extract_features_roi",
    "graph", "morphology", "viz",
]
