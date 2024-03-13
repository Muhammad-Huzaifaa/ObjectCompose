import os
import cv2
import glob
import json
import shutil
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

    # save image
    file_name = image_path.split("/")[-1]
    cv2.imwrite(f"./data/{file_name}", img)

coco_path = "/home/hashmat/Downloads/Coco_2017/archive/val2017"
image_paths = glob.glob("/home/hashmat/Downloads/Coco_2017/filtered/images_with_masks/*")

save_folder = "/home/hashmat/Downloads/Coco_2017/filtered/images/"
for image_path in image_paths:
    filename = image_path.split("/")[-1]
    orig_image_path = os.path.join(coco_path, filename)
    save_image_path = os.path.join(save_folder, filename)
    shutil.copy(orig_image_path, save_image_path)


