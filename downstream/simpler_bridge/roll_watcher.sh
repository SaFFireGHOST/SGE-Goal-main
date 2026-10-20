#!/bin/bash
# Poll the volume; auto-launch a GC rollout for each method as its goals land.
# Uses marker files (not bash arrays) to avoid issues with dotted method names.
source ~/miniconda3/etc/profile.d/conda.sh && conda activate lgrm
cd "$(dirname "$0")"
METHODS="gemini-3-pro-image gpt-image-1.5 flux-kontext-ours-klein flux2-klein-ours-klein susie clover"
MARK=/tmp/gc_rolled
mkdir -p "$MARK"
for i in $(seq 1 360); do   # up to ~12h
  alldone=1
  for m in $METHODS; do
    [ -f "$MARK/$m" ] && continue
    alldone=0
    if [ "$(modal volume ls simpler-eval-data "goals/$m" 2>/dev/null | grep -c widowx)" -ge 4 ]; then
      echo "[$(date +%H:%M)] GOALS READY -> rollout $m"
      nohup modal run modal_rollout_gc.py::run_method --method "$m" --n-eps 24 --chunk 6 \
        > "/tmp/gc_full_$m.log" 2>&1 &
      touch "$MARK/$m"
    fi
  done
  [ "$alldone" = "1" ] && { echo "all methods rolled"; break; }
  sleep 120
done
