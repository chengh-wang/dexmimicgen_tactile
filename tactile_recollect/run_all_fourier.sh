#!/usr/bin/env bash
# Orchestrate the full-dataset tactile extraction for all 3 Fourier (GR1) tasks.
# Waits for each raw HDF5 download to finish, then runs a resumable extraction.
# Safe to re-run: extraction skips demos already present in the output.
set -u
cd "$(dirname "$0")/.."                      # -> dexmimicgen repo root
export PYTHONPATH="$PWD"
PY=tactile_debug/.venv/bin/python
RAW=tactile_recollect/data/raw/generated
OUT=tactile_recollect/data/tactile
LOG=tactile_recollect/data/extract.log
mkdir -p "$OUT"

TASKS=(two_arm_can_sort_random two_arm_coffee two_arm_pouring)

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

for t in "${TASKS[@]}"; do
    src="$RAW/$t.hdf5"
    dst="$OUT/${t}_tactile.hdf5"
    log "waiting for download: $src"
    # wait until the final (renamed) file exists and its size is stable
    prev=-1
    while true; do
        if [ -f "$src" ]; then
            cur=$(stat -f%z "$src" 2>/dev/null || echo 0)
            if [ "$cur" = "$prev" ] && [ "$cur" -gt 0 ]; then break; fi
            prev=$cur
        fi
        sleep 20
    done
    log "download ready ($(du -h "$src" | cut -f1)); extracting -> $dst"
    $PY -m tactile_recollect.extract --dataset "$src" --out "$dst" --resume \
        >>"$LOG" 2>&1
    rc=$?
    log "extract $t finished rc=$rc"
done
log "ALL EXTRACTIONS COMPLETE"
