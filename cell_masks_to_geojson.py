"""Convert histoplus cell_masks.json -> QuPath-compatible GeoJSON, one per slide.

Walks `histoplus_output/<slide>/cell_masks.json` and writes
`histoplus_output/<slide>/cell_masks.geojson` next to it.

Each tile entry has integer tile indices (x, y) and width/height.
Cell `coordinates` are tile-local; global pixel = tile_idx * tile_size + local.

Cells are written as QuPath `detection` objects whose `classification.name`
is the cell type. QuPath then lists every cell type in the Annotations-tab
class list, so you can pick a type there and "Select objects by classification"
to grab all cells of that type at once.

Input files are large (hundreds of MB to >1 GB), so the JSON is streamed in
with ijson and the GeoJSON is streamed out feature-by-feature.
"""
import json
from pathlib import Path

import ijson

ROOT = Path("/Users/srmedinac/Documents/PhD/penile/histoplus_output")

# Stable colors per cell type (RGB 0-255) for QuPath classifications.
COLORS = {
    "Fibroblasts":        [255, 165,   0],
    "Lymphocytes":        [  0, 128, 255],
    "Plasmocytes":        [128,   0, 255],
    "Macrophages":        [255,   0, 128],
    "Neutrophils":        [255, 255,   0],
    "Eosinophils":        [255,  64,  64],
    "Endothelial Cell":   [  0, 200, 200],
    "Epithelial":         [200, 200, 200],
    "Muscle Cell":        [128,  64,   0],
    "Cancer cell":        [220,  20,  60],
    "Apoptotic Body":     [ 80,  80,  80],
    "Red blood cell":     [180,   0,   0],
    "Minor Stromal Cell": [160, 120,  90],
    "Mitotic Figures":    [255,   0,   0],
}
DEFAULT_COLOR = [150, 150, 150]


def convert(src: Path, dst: Path) -> None:
    n_feat = 0
    n_skip = 0
    counts: dict[str, int] = {}

    with open(src, "rb") as fin, open(dst, "w") as fout:
        fout.write('{"type":"FeatureCollection","features":[')
        first = True
        for tile in ijson.items(fin, "cell_masks.item"):
            ox = int(tile["x"]) * int(tile["width"])
            oy = int(tile["y"]) * int(tile["height"])
            for cell in tile.get("masks", []):
                coords = cell["coordinates"]
                if len(coords) < 3:
                    n_skip += 1
                    continue
                ring = [[float(c[0]) + ox, float(c[1]) + oy] for c in coords]
                if ring[0] != ring[-1]:
                    ring.append(ring[0])
                ctype = cell.get("cell_type", "Unknown")
                counts[ctype] = counts.get(ctype, 0) + 1
                feat = {
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": [ring]},
                    "properties": {
                        "objectType": "detection",
                        "classification": {
                            "name": ctype,
                            "color": COLORS.get(ctype, DEFAULT_COLOR),
                        },
                        "measurements": {
                            "confidence": float(cell.get("confidence", 0.0))
                        },
                    },
                }
                fout.write("" if first else ",")
                json.dump(feat, fout)
                first = False
                n_feat += 1
        fout.write("]}")

    summary = ", ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
    print(f"  {dst.name}: {n_feat} cells (skipped {n_skip}), "
          f"{dst.stat().st_size / 1e6:.1f} MB")
    print(f"    types -> {summary}")


def main() -> None:
    srcs = sorted(ROOT.glob("*/cell_masks.json"))
    if not srcs:
        print(f"no cell_masks.json found under {ROOT}")
        return
    for src in srcs:
        dst = src.with_name("cell_masks.geojson")
        if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
            print(f"{src.parent.name}/  (geojson up to date, skipping)")
            continue
        print(f"{src.parent.name}/")
        convert(src, dst)


if __name__ == "__main__":
    main()
