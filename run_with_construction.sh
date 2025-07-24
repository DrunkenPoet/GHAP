#!/bin/bash

# Usage: bash run.sh <scene_name> <source_path> [<sampling_ratio>]

scene_name=$1
source_path=$2
sampling_ratio=${3:-0.2}   # default is 0.2 if not provided

# parameter checking
if [ -z "$scene_name" ] || [ -z "$source_path" ]; then
  echo "Usage: bash run.sh <scene_name> <source_path>"
  exit 1
fi

# build output path
directory1=./experiments/${scene_name}_${sampling_ratio}_with_construction

# training
CUDA_VISIBLE_DEVICES=0 python train_and_prune.py \
    -s "$source_path" \
    -m "$directory1" \
    --sampling_ratio $sampling_ratio \
    --eval \
    --compact \
    --disable_viewer

# render & evaluate
python render.py -m "$directory1"
python metrics.py -m "$directory1"