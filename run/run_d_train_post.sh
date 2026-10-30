#!/bin/bash
# Waits for build_d_train.py to finish, then runs the overlap + replay checks on all contexts and
# makes the Variant A train/val split. Detached: setsid nohup bash run/run_d_train_post.sh > run/logs/d_train_post.log 2>&1 < /dev/null &
R=/home/sgeextn/SGE-Goal-Cleaned/SGE-Goal-main/run
until grep -q "BUILD_DONE\|Traceback" $R/logs/build_d_train.log; do sleep 60; done
if ! grep -q BUILD_DONE $R/logs/build_d_train.log; then echo "BUILD_FAILED"; tail -20 $R/logs/build_d_train.log; exit 1; fi
source $R/eval_env.sh
export OMP_NUM_THREADS=1
cd $R/gr_mg_ws
/home/sgeextn/.conda/envs/gate0/bin/python $R/calvin/check_d_train.py --replay 100 2>&1 | grep -E "^overlap|^replay|Error|Traceback"
/home/sgeextn/.conda/envs/sgegen/bin/python -I $R/calvin/make_va_splits.py
du -sh $R/calvin/d_train
echo POST_DONE
