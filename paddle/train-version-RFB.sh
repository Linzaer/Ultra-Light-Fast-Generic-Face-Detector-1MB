#!/usr/bin/env bash
model_root_path="./models/train-version-RFB"
log_dir="$model_root_path/logs"
log="$log_dir/log"
mkdir -p "$log_dir"

python3 -u train.py \
  --datasets \
  ./data/wider_face_add_lm_10_10 \
  --validation_dataset \
  ./data/wider_face_add_lm_10_10 \
  --net \
  RFB \
  --num_epochs \
  200 \
  --milestones \
  "95,150" \
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
  2>&1 | tee "$log"
