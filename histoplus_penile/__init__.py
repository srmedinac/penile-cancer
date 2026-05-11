"""HistoPLUS cell segmentation + nuclear-diversity pipeline for the MDACC penile cohort.

Built on top of the Trident outputs (GrandQC tissue segmentation + 20x/512px patch
coordinates) so that HistoPLUS segments cells in *exactly* the tissue Trident kept,
without re-detecting tissue.

Modules
-------
trident_io       : read Trident patch-coordinate h5s, convert to HistoPLUS DeepZoom tiles.
extract_cells    : run the HistoPLUS CellViT segmentor on each slide -> cell_masks.json.
nuclear_diversity: turn the cell masks into per-patch and per-slide diversity tables.
"""
