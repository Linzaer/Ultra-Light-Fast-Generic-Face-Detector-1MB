#!/usr/bin/env bash
model_root_path="./models/train-version-RFB"
log_dir="$model_root_path/logs"
log="$log_dir/log"
mkdir -p "$log_dir"

python3 -u train.py \
  --datasets \
  ../SCUT_HEAD_Part_B \
  --validation_dataset \
  ../SCUT_HEAD_Part_B \
  --net \
  RFB \
  --num_epochs \
  500 \
  --scheduler \
  cosine \
  --t_max \
  500 \
  --lr \
  1e-2 \
  --batch_size \
  24 \
  --input_size \
  320 \
  --checkpoint_folder \
  ${model_root_path} \
  --num_workers \
  4 \
  --log_dir \
  ${log_dir} \
  --cuda_index \
  0 \
  --validation_epochs \
  20 \
  --optimizer_type \
  SGD \
  --debug_steps \
  20 \
  2>&1 | tee "$log"
