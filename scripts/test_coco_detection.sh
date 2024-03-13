#!/bin/sh

use_captions=${1:-"True"}
prompt=${2:-0}
save_dir=$3
use_class_names=$4
root=${5:-"."}
data_path=${6:-"Coco_2017/filtered/"}

prompts=("None" "This is a photo of textures background" "This is a photo of intricately rich textures background" "This is a photo of colorful textures background" "This is a photo of distorted textures background" "this is a picture of a vivid red background" "this is a picture of a vivid green background" "this is a picture of a vivid blue background" "this is a picture of a vivid colorful background")
name_ext=("None" "texture_1" "texture_2" "texture_3" "texture_4" "red" "green" "blue" "colorful")


for expand_pixels in  0 3 6 9
do
  echo  "Running experiment with use_captions: $1, prompt: ${prompts[$prompt]}, expand_pixels: $expand_pixels"
  echo "Data path: $6"
  SAVE_DIR="${root}/${save_dir}_expand_mask_${expand_pixels}_use_cname_${use_class_names}_prompt_${name_ext[$prompt]}"
  echo "Saving to ${SAVE_DIR}"

  python obj_compose.py \
    --dataset coco \
    --data_path $data_path \
    --expand_mask_pixels $expand_pixels \
    --use_captions $use_captions \
    --prompt "${prompts[$prompt]}" \
    --use_class_name $use_class_names \
    --save_dir $SAVE_DIR \

  cd evaluations/detr-main

  python main.py \
    --batch_size 2 \
    --no_aux_loss --eval --resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth --coco_path "${SAVE_DIR}/dataset" \
    --output_dir "${SAVE_DIR}/detr_results"

  cd ../..

done

# bash scripts/test_coco_detection.sh <use captions: True/False> <prompt: 0-8> <save_dir : blip_caption> <use_class_names: True/False> <root: results> <data_path: Coco_2017/filtered/>