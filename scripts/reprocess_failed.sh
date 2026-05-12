#!/bin/bash
# Re-pull the slides that failed the first Trident/HistoPLUS run from the CIFS share,
# test whether the fresh copy is usable, then re-run both pipelines (both are resumable
# and skip slides that already have complete outputs).
#
# Failed first time round:
#   Trident   : MPe13LN.svs, MPe20P.svs  (truncated SVS — TIFF first-IFD past EOF)
#               MPe3P-A9.svs              (opens, but >=1 corrupt JPEG tile -> feat step dies)
#   HistoPLUS : MPe18LN                   (our bug — edge patch expanded to an out-of-grid
#                                          DeepZoom tile; fixed in trident_io.to_histoplus_coords)
#
# A re-pulled slide is moved into /tmp/penile_wsi_local only if OpenSlide can open it
# AND (for MPe3P-A9) the bytes differ from the copy we already have — otherwise the retry
# would just reproduce the same failure. Slides that are still broken are moved aside with
# a .broken suffix so the pipeline globs skip them cleanly.
#
# Usage:  bash scripts/reprocess_failed.sh
set -uo pipefail

PROJ=/home/smedin7/Documents/penile
SRC="/home/smedin7/g_drive/emory_datasets/rad_path_data/gu/penile/mdacc/curtis-pettaway/40x_Niki"
LOCAL=/tmp/penile_wsi_local
TRIDENT_OUT=$PROJ/trident_output
LOG=$PROJ/logs/reprocess_failed_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1

echo "############################################################"
echo "# reprocess_failed.sh   started $(date)"
echo "# log: $LOG"
echo "############################################################"

source /home/smedin7/miniconda3/etc/profile.d/conda.sh

opens_ok() {  # $1 = path ; prints OK/ FAIL line, returns 0 if openslide can open it
    conda activate trident
    python - "$1" <<'PY'
import sys, openslide
p = sys.argv[1]
try:
    o = openslide.OpenSlide(p)
    print(f"  OpenSlide: OK  dims={o.dimensions} levels={o.level_count}")
    o.close()
    sys.exit(0)
except Exception as e:
    print(f"  OpenSlide: FAIL  {e!r}")
    sys.exit(1)
PY
}

echo
echo "=== 1. re-pull failed source slides from CIFS share ==="
echo "    src: $SRC"
rm -fv "$LOCAL"/MPe3P-A9.svs.new   # stale partial from an earlier aborted re-download

for s in MPe13LN MPe20P MPe3P-A9; do
    echo
    echo "--- $s ---"
    if [ ! -f "$SRC/$s.svs" ]; then
        echo "  source file missing on share?! skipping"
        continue
    fi
    echo "  copying $s.svs ($(numfmt --to=iec $(stat -c%s "$SRC/$s.svs")) ) ..."
    t0=$SECONDS
    cp -p "$SRC/$s.svs" "$LOCAL/$s.svs.new"
    rc=$?
    echo "  cp rc=$rc  in $((SECONDS-t0))s   new size=$(stat -c%s "$LOCAL/$s.svs.new" 2>/dev/null)  src size=$(stat -c%s "$SRC/$s.svs")"
    [ $rc -ne 0 ] && { echo "  cp failed; leaving existing copy untouched"; rm -f "$LOCAL/$s.svs.new"; continue; }

    if [ -f "$LOCAL/$s.svs" ] && cmp -s "$LOCAL/$s.svs" "$LOCAL/$s.svs.new"; then
        same=yes; echo "  re-pulled bytes are IDENTICAL to the copy we already had"
    else
        same=no;  echo "  re-pulled bytes DIFFER from the existing copy (or there was none)"
    fi

    if opens_ok "$LOCAL/$s.svs.new"; then
        newok=yes
    else
        newok=no
    fi

    if [ "$newok" = yes ] && { [ "$same" = no ] || [ ! -f "$LOCAL/$s.svs" ]; }; then
        echo "  -> usable & new bytes: installing fresh copy, Trident will retry it"
        mv -f "$LOCAL/$s.svs.new" "$LOCAL/$s.svs"
        rm -f "$LOCAL/$s.svs.broken"
    elif [ "$newok" = yes ] && [ "$same" = yes ]; then
        echo "  -> opens, but byte-identical to the copy that already failed; not retrying"
        rm -f "$LOCAL/$s.svs.new"
        # for MPe3P-A9 the failure is a corrupt tile mid-WSI; keep the file in place so
        # the rest of the cohort still references it, but it will fail feat again.
    else
        echo "  -> still broken; moving aside as $s.svs.broken so the pipeline skips it"
        rm -f "$LOCAL/$s.svs.new"
        [ -f "$LOCAL/$s.svs" ] && mv -f "$LOCAL/$s.svs" "$LOCAL/$s.svs.broken"
    fi
done

echo
echo "=== 2. clear stale Trident lock files for the failed slides ==="
rm -fv "$TRIDENT_OUT"/contours/MPe13LN.jpg.lock \
       "$TRIDENT_OUT"/contours/MPe20P.jpg.lock \
       "$TRIDENT_OUT"/contours/MPe3P-A9.jpg.lock

echo
echo "=== 3. Trident stage A  (seg + coords + CONCH v1.5; resumable, skips the 26 done) ==="
bash "$PROJ/scripts/run_trident_penile.sh" A

echo
echo "=== 4. Trident stage B  (TITAN slide features; resumable) ==="
bash "$PROJ/scripts/run_trident_penile.sh" B

echo
echo "=== 5. HistoPLUS cell masks  (resumable; skips the 25 done, retries MPe18LN + anything recovered) ==="
bash "$PROJ/scripts/run_histoplus_penile.sh" cells

echo
echo "############################################################"
echo "# reprocess_failed.sh   done $(date)"
echo "############################################################"
echo
echo "Trident features present:"
ls "$TRIDENT_OUT"/20x_512px_0px_overlap/features_conch_v15/ | wc -l
echo "HistoPLUS cell_masks present:"
ls "$PROJ"/histoplus_output/*/cell_masks.json | wc -l
