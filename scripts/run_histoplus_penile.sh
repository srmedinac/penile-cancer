#!/bin/bash
# HistoPLUS cell segmentation + nuclear-diversity for the penile cohort, on top of Trident.
# Run AFTER scripts/run_trident_penile.sh has produced patch coords.
#
# Usage:
#   bash scripts/run_histoplus_penile.sh            # cells, then diversity
#   bash scripts/run_histoplus_penile.sh cells      # just HistoPLUS cell masks
#   bash scripts/run_histoplus_penile.sh diversity  # just the diversity tables
set -e

HF_TOKEN_LINE=$(grep -E '^export HF_TOKEN=' ~/.bashrc | tail -1); eval "$HF_TOKEN_LINE"; export HF_TOKEN
echo "HF_TOKEN set: $([ -n "$HF_TOKEN" ] && echo yes || echo NO)"
source /home/smedin7/miniconda3/etc/profile.d/conda.sh
conda activate histoplus

PROJ=/home/smedin7/Documents/penile
WSI_DIR=/tmp/penile_wsi_local
TRIDENT_DIR=$PROJ/trident_output
COORDS_SUBDIR=20x_512px_0px_overlap
OUT=$PROJ/histoplus_output

cd "$PROJ"
STAGE="${1:-all}"

run_cells() {
    python -m histoplus_penile.extract_cells \
        --wsi_dir "$WSI_DIR" \
        --trident_dir "$TRIDENT_DIR" \
        --coords_subdir "$COORDS_SUBDIR" \
        --out "$OUT" \
        --batch_size 8 \
        --skip_errors
}

run_diversity() {
    python -m histoplus_penile.nuclear_diversity \
        --histoplus_dir "$OUT" \
        --trident_dir "$TRIDENT_DIR" \
        --coords_subdir "$COORDS_SUBDIR" \
        --out "$OUT"
}

case "$STAGE" in
    cells)     run_cells ;;
    diversity) run_diversity ;;
    all)       run_cells; run_diversity ;;
    *) echo "unknown stage: $STAGE (use: cells|diversity|all)"; exit 1 ;;
esac

echo "Done. Cell masks: $OUT/<slide>/cell_masks.json | per-patch: $OUT/patch_diversity/<slide>.csv | per-slide: $OUT/slide_diversity.csv"
