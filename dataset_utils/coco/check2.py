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


for ann in data["annotations"]:
    image_id = ann["image_id"]
    image_name = id_to_image_name[image_id]
    category_id = ann["category_id"]
    iscrowd = ann["iscrowd"]


    if iscrowd == 1:
        image_names_crowd.append(image_name)
        continue


    arr = ann["segmentation"]

    if len(arr) > 1 :
        image_names_with_divided_masks.append(image_name)




    print(f"Image ID: {image_id}")
    print(f"Image Name: {image_name}")
    print(f"Category ID: {category_id}")
    print(f"Category Name: {class_from_id[category_id]}")
    print(f"Is Crowd: {iscrowd}")

    if image_id not in seg_labels:
        seg_labels[image_id] = []

    if len(arr) > 1:
        a = [np.asarray(x, np.int32).reshape((-1,2)) for x in arr]
        dic = {"mask": a, "category": category_id, "class_name": class_from_id[category_id]}

    elif len(arr) == 1:
        arr = np.asarray(arr, np.int32)
        arr = arr.reshape((-1, 2))
        dic = {"mask": arr, "category": category_id, "class_name": class_from_id[category_id]}

    else:
        print("Error")
        break

    seg_labels[image_id].append(dic)

seg_labels_ = {}

for k,v in seg_labels.items():
    image_name = id_to_image_name[k]
    if image_name in image_names_crowd:
        continue
    elif image_name in image_names_with_divided_masks:
        continue
    elif image_name in filtered_image_names:
        seg_labels_[k] = v
    else:
        print("Not Found")




#

for image_info in data["images"]:
    image_id = image_info["id"]
    image_name = image_info["file_name"]
    image_path = os.path.join("./archive/val2017", image_name)
    image = cv2.imread(image_path)
    if image_id not in seg_labels_:
        continue
    ann_info = seg_labels_[image_id]

    save_annotated_image(image_path, ann_info)



    if image_name in image_names_crowd:
        continue

