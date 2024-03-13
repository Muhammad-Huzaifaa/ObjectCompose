#!/bin/sh

resume=$1
model_id=${2:-0}
data_root=${3:-"/home/hashmat/Code/Projects/InDomain_Adversaries/dataset_utils/coco/"}
output_directory=${4:-"./results_new"}

# Define the model choices
model_choices=("resnet50" "vit_tiny_patch16_224" "vit_small_patch16_224" "swin_tiny_patch4_window7_224" "swin_small_patch4_window7_224" "densenet161" "resnet152")
folder_names=("original" "blip_mask_0" "blip_mask_3" "blip_mask_6" "blip_mask_9" "blue_mask_0" "blue_mask_3" "blue_mask_6" "blue_mask_9" "red_mask_0" "red_mask_3" "red_mask_6" "red_mask_9" "green_mask_0" "green_mask_3" "green_mask_6" "green_mask_9" "colorful_mask_0" "colorful_mask_3" "colorful_mask_6" "colorful_mask_9" "texture_1_mask_0" "texture_1_mask_3" "texture_1_mask_6" "texture_1_mask_9" "texture_2_mask_0" "texture_2_mask_3" "texture_2_mask_6" "texture_2_mask_9" "texture_3_mask_0" "texture_3_mask_3" "texture_3_mask_6" "texture_3_mask_9" "texture_4_mask_0" "texture_4_mask_3" "texture_4_mask_6" "texture_4_mask_9" "adversarial_ensemble_coco_lr_5")



for folder in "${folder_names[@]}"; do
    data_root_="$data_root/$folder"
    python eval.py "$data_root_" \
    --model "${model_choices[$model_id]}" \
    --pretrained \
    --num-classes 80 \
    --experiment "${model_choices[$model_id]}_${folder}" \
    --output "$output_directory" \
    --resume "$resume"
done