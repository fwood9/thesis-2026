#!/usr/bin/env bash

# Multi-seed repeatability sweep. Runs the whole pipeline once per seed, each
# writing to artifacts/seed_/, then aggregates into artifacts/mean/.


# build the fixed split once (uses random_state=10 inside stage2)
python stage1_features.py
python stage2_split.py

 SEEDS="4 8 9 10 30 64 67 88 117 149"   # 10 seeds

for s in $SEEDS; do
  echo "=================  SEED $s  ================="
  SEED=$s python stage3_train.py
  SEED=$s python stage4_calibrate.py
  SEED=$s python stage5_predict.py
  SEED=$s python stage6_importance.py
  SEED=$s python stage7_fairness.py
  SEED=$s python stage8_fairness_race_age.py
  SEED=$s python stage9_fairness_ci.py
  SEED=$s python stage10_smote_basic.py
  SEED=$s python stage10_smote_rerun.py
  SEED=$s python stage14_dir_rerun.py
  SEED=$s python stage15_dir_rerun_many_features.py
done

python aggregate_seeds.py
echo "done -> artifacts/mean/"
