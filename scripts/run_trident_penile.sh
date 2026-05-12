#!/bin/bash
# Trident pipeline for the MDACC penile cancer cohort (29 .svs WSIs).
#
# The 29 .svs live on a slow/contended Emory CIFS share. Trident's built-in
# --wsi_cache stager copies at ~1 MB/s on this mount, but a plain `cp` does
# ~70 MB/s — so we stage the slides locally with cp ourselves, then point
# Trident at the local copy (no --wsi_cache).
#
#   Stage 0  cp  -> stage all .svs to ${LOCAL_WSI}
#   Stage A  --task all  --patch_encoder conch_v15
#            -> tissue segmentation + QC contours/geojson
#            -> tissue patch coords @ 20x, 512px, 0px overlap
#            -> CONCH v1.5 patch features (.h5)
#   Stage B  --task feat --slide_encoder titan
#            -> TITAN slide-level features (.h5)
#
# Known-bad source files (truncated SVS, won't open in OpenSlide): MPe13LN.svs, MPe20P.svs.
# MPe3P-A9.svs has at least one corrupt JPEG tile (may fail in feat step).
#
# Non-interactive bash skips most of ~/.bashrc, so we pull HF_TOKEN out of it
# explicitly and activate the trident conda env by hand.
#
# Usage:
#   bash scripts/run_trident_penile.sh          # stage + A + B
#   bash scripts/run_trident_penile.sh stage    # just copy WSIs locally
#   bash scripts/run_trident_penile.sh A        # patch pass (seg+coords+CONCH)
#   bash scripts/run_trident_penile.sh B        # TITAN slide pass
set -e

# --- HF token (needed for CONCH / TITAN weight download) ---
HF_TOKEN_LINE=$(grep -E '^export HF_TOKEN=' ~/.bashrc | tail -1)
eval "$HF_TOKEN_LINE"
export HF_TOKEN
echo "HF_TOKEN set: $([ -n "$HF_TOKEN" ] && echo yes || echo NO)"

# --- conda env ---
source /home/smedin7/miniconda3/etc/profile.d/conda.sh
conda activate trident
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # reduce CUDA fragmentation on the 24 GB card

TRIDENT_DIR="/home/smedin7/trident"
SRC_WSI="/home/smedin7/g_drive/emory_datasets/rad_path_data/gu/penile/mdacc/curtis-pettaway/40x_Niki"
LOCAL_WSI="/tmp/penile_wsi_local"
JOB_DIR="/home/smedin7/Documents/penile/trident_output"

MAG=20
PATCH_SIZE=512
OVERLAP=0
# In --task all the seg model and the CONCH encoder share the GPU, so the seg
# batch must stay small (256 OOMs a 24 GB card). seg=16, feat=64.
SEG_BATCH=16
FEAT_BATCH=64

STAGE="${1:-all}"
cd "${TRIDENT_DIR}"

run_stage() {
    echo "=== Stage 0: staging .svs from CIFS share to ${LOCAL_WSI} ==="
    mkdir -p "${LOCAL_WSI}"
    for src in "${SRC_WSI}"/*.svs; do
        name=$(basename "${src}")
        dst="${LOCAL_WSI}/${name}"
        if [ -f "${dst}" ] && [ "$(stat -c%s "${dst}")" -eq "$(stat -c%s "${src}")" ]; then
            echo "  have ${name}"
        else
            echo "  copying ${name} ..."
            cp -p "${src}" "${dst}"
        fi
    done
    echo "  staged $(ls "${LOCAL_WSI}"/*.svs | wc -l) slides, $(du -sh "${LOCAL_WSI}" | cut -f1)"
}

run_A() {
    [ -d "${LOCAL_WSI}" ] || { echo "ERROR: ${LOCAL_WSI} missing — run 'stage' first"; exit 1; }
    echo "=== Stage A: seg + coords + CONCH v1.5 patch features (local WSIs) ==="
    python run_batch_of_slides.py \
        --task all \
        --wsi_dir "${LOCAL_WSI}" \
        --job_dir "${JOB_DIR}" \
        --wsi_ext .svs \
        --patch_encoder conch_v15 \
        --mag ${MAG} \
        --patch_size ${PATCH_SIZE} \
        --overlap ${OVERLAP} \
        --seg_batch_size ${SEG_BATCH} \
        --feat_batch_size ${FEAT_BATCH} \
        --skip_errors
}

run_B() {
    echo "=== Stage B: TITAN slide-level features ==="
    python run_batch_of_slides.py \
        --task feat \
        --wsi_dir "${LOCAL_WSI}" \
        --job_dir "${JOB_DIR}" \
        --wsi_ext .svs \
        --slide_encoder titan \
        --mag ${MAG} \
        --patch_size ${PATCH_SIZE} \
        --overlap ${OVERLAP} \
        --skip_errors
}

case "${STAGE}" in
    stage) run_stage ;;
    A)     run_A ;;
    B)     run_B ;;
    all)   run_stage; run_A; run_B ;;
    *) echo "Unknown stage: ${STAGE} (use: stage|A|B|all)"; exit 1 ;;
esac

echo ""
echo "========================================"
echo "Trident penile MDACC stage '${STAGE}' complete."
echo "Outputs under: ${JOB_DIR}"
echo "  QC contours:   ${JOB_DIR}/contours/  +  ${JOB_DIR}/contours_geojson/"
echo "  Patches:       ${JOB_DIR}/${MAG}x_${PATCH_SIZE}px_${OVERLAP}px_overlap/patches/"
echo "  CONCH feats:   ${JOB_DIR}/${MAG}x_${PATCH_SIZE}px_${OVERLAP}px_overlap/features_conch_v15/"
echo "  TITAN feats:   ${JOB_DIR}/${MAG}x_${PATCH_SIZE}px_${OVERLAP}px_overlap/slide_features_titan/"
echo "========================================"
