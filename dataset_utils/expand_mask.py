import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot(img):

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


# Input and output paths
input_mask_folder = 'Coco_2017/filtered/masks'
output_mask_folder = 'Coco_2017/filtered/masks_expanded'
expansion_pixels = 5  # Adjust this based on your desired expansion

# Create the output folder if it doesn't exist
if not os.path.exists(output_mask_folder):
    os.makedirs(output_mask_folder)

# Loop through subdirectories and files in the input mask folder
for root, _, files in os.walk(input_mask_folder):
    for filename in files:
        mask_path = os.path.join(root, filename)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Invert the mask
        inverted_mask = cv2.bitwise_not(mask)

        # Create a structuring element (kernel) for dilation
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * expansion_pixels + 1, 2 * expansion_pixels + 1))

        # Perform dilation on the inverted mask
        expanded_inverted_mask = cv2.dilate(inverted_mask, kernel, iterations=1)

        # Invert the expanded mask back to get the desired result
        expanded_mask = cv2.bitwise_not(expanded_inverted_mask)

        # Get the relative path to the mask subfolder
        rel_path = os.path.relpath(root, input_mask_folder)

        # Create the corresponding subdirectory in the output directory
        output_subfolder = os.path.join(output_mask_folder, rel_path)
        if not os.path.exists(output_subfolder):
            os.makedirs(output_subfolder)

        # Save the expanded mask in the output subdirectory with the same filename and format
        output_mask_path = os.path.join(output_subfolder, filename)
        cv2.imwrite(output_mask_path, expanded_mask)

print("Mask expansion and saving completed.")