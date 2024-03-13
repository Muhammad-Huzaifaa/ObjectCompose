import os
import cv2
import glob
import json

from torchvision.io import read_image
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes

def get_image_name_from_id(data):
    id_to_image_name = {}
    for image_info in data["images"]:
        id_to_image_name[image_info["id"]] = image_info["file_name"]
    return id_to_image_name

def plot_image_from_path(path):
    img = read_image(path)
    img = transforms.ToPILImage()(img)
    plt.imshow(img)
    plt.show()

def get_class_names_from_id(data):
    class_id = {}
    for cat in data["categories"]:
        class_id[cat["id"]] = cat["name"]

    return class_id


def plot_box(image_path, masks):
    img = cv2.imread(image_path)

    if type(masks) == list:
        for mask in masks:
            mask = np.int32([mask])
            print(mask.shape)
            cv2.polylines(img, mask, isClosed=True, color=(0, 255, 0), thickness=2)
    else:
        mask = np.int32([masks])
        cv2.polylines(img, mask, isClosed=True, color=(0, 255, 0), thickness=2)

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('Image with Polygon Bounding Box')
    plt.show()


def plot_boxes(image_path, ann_info):
    img = cv2.imread(image_path)

    for ann in ann_info:
        masks = ann["mask"]
        if type(masks) == list:
            for mask in masks:
                mask = np.int32([mask])
                print(mask.shape)
                cv2.polylines(img, mask, isClosed=True, color=(0, 255, 0), thickness=2)
        else:
            mask = np.int32([masks])
            cv2.polylines(img, mask, isClosed=True, color=(0, 255, 0), thickness=2)

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('Image with Polygon Bounding Box')
    plt.show()

def save_annotated_image(image_path, ann_info):
    img = cv2.imread(image_path)
    height, width = image.shape[:2]
    mask_image = np.zeros((height, width), np.uint8)


    for ann in ann_info:
        masks = ann["mask"]
        if type(masks) == list:
            for mask in masks:
                mask = np.int32([mask])
                print(mask.shape)
                cv2.fillPoly(mask_image, mask, 255)
        else:
            mask = np.int32([masks])
            cv2.fillPoly(mask_image, mask, 255)

    # save image
    file_name = image_path.split("/")[-1]
    cv2.imwrite(f"./masks/{file_name}", mask_image)


image_paths = glob.glob("./filtered/images/*")
file = open("./archive/annotations_trainval2017/instances_val2017.json")
data = json.load(file)

filtered_image_names = [name.split("/")[-1] for name in image_paths]

id_to_image_name = get_image_name_from_id(data)
class_from_id = get_class_names_from_id(data)

image_names_crowd = []
seg_labels = {}
image_names_with_divided_masks = []


new_image_list = []

for image_info in data["images"]:
    image_name = image_info["file_name"]
    if image_name in filtered_image_names:
        new_image_list.append(image_info)

new_annotations = []

for ann_info in data["annotations"]:
    image_id = ann_info["image_id"]
    image_name = id_to_image_name[image_id]
    if image_name in filtered_image_names:
        new_annotations.append(ann_info)

data["images"] = new_image_list
data["annotations"] = new_annotations

with open("./newinstances_val2017.json", "w") as outfile:
    json.dump(data, outfile)

