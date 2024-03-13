#!/bin/sh

data_path=${1:-"/home/hashmat/Downloads/Coco_2017/coco_classification/images"}
model_id=$2
output_directory=$3

# Define the model choices
model_choices=("resnet50" "vit_tiny_patch16_224" "vit_small_patch16_224" "swin_tiny_patch4_window7_224" "swin_small_patch4_window7_224" "densenet161" "resnet152")


python train.py $data_path \
--model "${model_choices[$model_id]}" \
--pretrained \
--num-classes 80 \
--log-wandb \
--experiment "${model_choices[$model_id]}" \
--output $output_directory
