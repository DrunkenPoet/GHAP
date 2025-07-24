#!/bin/bash

# Usage: bash run_from_pointcloud.sh <scene_name> <source_path> [<ckp_path>] [<sampling_ratio>]

scene_name=$1
source_path=$2
ckp_path=$3
sampling_ratio=${4:-0.2}  # default is 0.2 if not provided
# parameter checking
if [ -z "$scene_name" ] || [ -z "$source_path" ]; then
  echo "Usage: bash run.sh <scene_name> <source_path> [<ckp_path>] [<sampling_ratio>]"
  exit 1
fi

# build output path
directory1=./experiments/${scene_name}_${sampling_ratio}_from_pointcloud

# optional checkpoint flag
if [ -n "$ckp_path" ]; then
  ckpt_flag="--start_checkpoint $ckp_path"
else
  ckpt_flag=""
fi

# training
CUDA_VISIBLE_DEVICES=0 python train_and_prune.py \
    -s "$source_path" \
    -m "$directory1" \
    --sampling_ratio $sampling_ratio \
    --eval \
    --compact \
    --disable_viewer \
    $ckpt_flag \
    --iteration 35000 \
    --test_iterations 30001 30002 35000 \
    --save_iterations 35000 \
    --checkpoint_iterations 35000 \
    --sampling_iter 30001

# render & evaluate
python render.py -m "$directory1"
python metrics.py -m "$directory1"


