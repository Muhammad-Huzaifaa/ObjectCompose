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

def save_annotated_image(image_path, ann_info, class_id):
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
    path = f"/home/hashmat/Downloads/Coco_2017/coco_class/{class_id}"
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    full_path = os.path.join(path, file_name)
    mask_path= f"/home/hashmat/Downloads/Coco_2017/coco_class_masks/{class_id}"
    if not os.path.exists(mask_path):
        os.makedirs(mask_path, exist_ok=True)
    mask_full_path = os.path.join(mask_path, file_name)
    cv2.imwrite(mask_full_path, mask_image)
    cv2.imwrite(full_path, img)


image_paths = glob.glob("/home/hashmat/Downloads/Coco_2017/archive/val2017/*")
file = open("/home/hashmat/Downloads/Coco_2017/archive/annotations/instances_val2017.json")
data = json.load(file)

id_to_image_name = get_image_name_from_id(data)
class_from_id = get_class_names_from_id(data)

image_names_crowd = []
seg_labels = {}
image_names_with_divided_masks = []

area_id = {}
annotations = {}

# Save annotations with the largest area for each image
for ann in data["annotations"]:
    image_id = ann["image_id"]
    category_id = ann["category_id"]
    area = ann["area"]



    if image_id not in area_id:
        area_id[image_id] = 0


    if area > area_id[image_id]:
        area_id[image_id] = area
        annotations[image_id] = ann

annotations_ = {}

for ann in data["annotations"]:
    image_id = ann["image_id"]
    category_id = ann["category_id"]
    area = ann["area"]

    ann_ = annotations[image_id]
    category_id_ = ann_["category_id"]


    if image_id not in annotations_:
        annotations_[image_id] = []

    if category_id == category_id_:
        annotations_[image_id].append(ann)




# Generate masks for each image
for k,v in annotations_.items():

    image_id = k


    for a in v:
        if a["iscrowd"] == 1:
            iscrowd = True
        else:
            iscrowd = False
    image_name = id_to_image_name[image_id]




    if iscrowd == 1:
        image_names_crowd.append(image_name)
        continue

    for a in v:
        a_ = a["segmentation"]
        if len(a_) > 1:
            if image_name not in image_names_with_divided_masks:
                image_names_with_divided_masks.append(image_name)


    if image_id not in seg_labels:
        seg_labels[image_id] = []

    category_id = v[0]["category_id"]

    for a in v:
        arr = a["segmentation"]


        if len(arr) > 1:
            a = [np.asarray(x, np.int32).reshape((-1, 2)) for x in arr]
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
    else:
        seg_labels_[k] = v



for image_info in data["images"]:
    image_id = image_info["id"]
    image_name = image_info["file_name"]
    image_path = os.path.join("/home/hashmat/Downloads/Coco_2017/archive/val2017", image_name)
    image = cv2.imread(image_path)
    if image_id not in seg_labels_:
        continue
    ann_info = seg_labels_[image_id]
    class_id = ann_info[0]["class_name"]

    save_annotated_image(image_path, ann_info, class_id)



    if image_name in image_names_crowd:
        continue

