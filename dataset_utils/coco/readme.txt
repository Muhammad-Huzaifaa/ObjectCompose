1. First run check.py on coco 5k val set to get the dataset with segmentation boundaries on it.
2. Now filter the dataset to get the images with fewer number of objects and with the object mostly being in the foreground.
3. Run check2.py to save the segmentation masks of the filtered images.
4. Run modify_annotations.py to update the json file reflecting the changes in the dataset.
5. Run check_filtered_data.py to check if the changes have been reflected in the dataset.
6. Run create_dataset.py to create the dataset with the segmentation boundaries removed.