#!/bin/bash
# Learned-IER downstream pipeline, run detached so it survives the SSH/VS Code session closing:
#   1. union model -> predicted M_i            (CALVIN-EVALS/sge_goal_union_ier)
#   2. Step 2 on predicted M_i -> M_f, IoU vs GT M_f
#   3. GT M_i folder -> Step 2 -> M_f, IoU vs GT M_f   (CALVIN-EVALS/sge_goal_gt_ier)
#   4. Step 3 on the GT-M_i folder
#   5. Step 3 on the union folder
# Every step skips cases already done, so rerunning resumes. Progress: run/logs/ier_pipeline.log
# Launch: setsid nohup bash run/run_ier_pipeline.sh > run/logs/ier_pipeline.log 2>&1 < /dev/null &
set -u
REPO=/home/sgeextn/SGE-Goal-Cleaned/SGE-Goal-main
R=$REPO/run
PY=/home/sgeextn/.conda/envs/sgegen/bin/python
SRC=$R/CALVIN-EVALS/sge_goal/calvin_abcd_test_data
UNION=$R/CALVIN-EVALS/sge_goal_union_ier/calvin_abcd_test_data
GTIER=$R/CALVIN-EVALS/sge_goal_gt_ier/calvin_abcd_test_data
CK=$R/ckpts/sge-goal-checkpoints/flux1-kontext-dev/calvin
FER_DIR=$REPO/sge_goal/mask_predictor/inference/flux_kontext_bridge_calvin
GOAL_DIR=$REPO/sge_goal/synthesis/inference/flux_kontext_bridge_calvin
OUT=$R/ier_results
mkdir -p $OUT $R/logs

stage() { echo; echo "===== [$(date '+%F %T')] $1 ====="; }
run() {  # run <log-name> <cmd...>; abort the pipeline if it fails
  local name=$1; shift
  "$@" > $R/logs/$name.log 2>&1
  local rc=$?
  echo "$name exit=$rc"
  if [ $rc -ne 0 ]; then echo "PIPELINE_FAILED at $name"; tail -20 $R/logs/$name.log; exit $rc; fi
}

stage "1/5 predict M_i with the union model"
run ier_predict_union bash $R/predict_ier.sh --checkpoint $R/ier_training/evf_union/best.pt \
  --data-dir $SRC --out-dir $UNION --skip-existing --gt-parquet "$R/calvin/hf_dataset/data/test-*.parquet"
tail -10 $R/logs/ier_predict_union.log

stage "2/5 Step 2 (final mask) on predicted union M_i"
run fer_union bash -c "cd $FER_DIR && $PY local_fer_mask_inference.py --data-dir $UNION --checkpoint-dir $CK/step2_final_mask --skip-existing --low-vram"
$PY $R/mask_iou_vs_gt.py --mask initial --dirs baseline_training_free=$SRC learned_union=$UNION --out $OUT/initial_mask_iou.json
$PY $R/mask_iou_vs_gt.py --mask final --dirs baseline_training_free=$SRC learned_union=$UNION --out $OUT/final_mask_iou_union.json

stage "3/5 Step 2 (final mask) on ground-truth M_i"
$PY $R/make_gt_ier_folder.py --src-dir $SRC --out-dir $GTIER
run fer_gt_ier bash -c "cd $FER_DIR && $PY local_fer_mask_inference.py --data-dir $GTIER --checkpoint-dir $CK/step2_final_mask --skip-existing --low-vram"
$PY $R/mask_iou_vs_gt.py --mask final --dirs baseline_training_free=$SRC learned_union=$UNION gt_initial=$GTIER --out $OUT/final_mask_iou.json

stage "4/5 Step 3 (goal image) on GT M_i + predicted M_f"
run goal_gt_ier bash -c "cd $GOAL_DIR && $PY local_flux_kontext_inference.py --data-dir $GTIER --checkpoint-dir $CK/step3_goal_image --skip-existing --low-vram"

stage "5/5 Step 3 (goal image) on learned union M_i + predicted M_f"
run goal_union bash -c "cd $GOAL_DIR && $PY local_flux_kontext_inference.py --data-dir $UNION --checkpoint-dir $CK/step3_goal_image --skip-existing --low-vram"

stage "done"
for d in $UNION $GTIER; do echo "$d: $(ls $d/*/edited_image.jpg 2>/dev/null | wc -l) goal images"; done
echo PIPELINE_DONE
