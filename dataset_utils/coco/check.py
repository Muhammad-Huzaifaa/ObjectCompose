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


image_paths = glob.glob("./archive/val2017/*")
file = open("./archive/annotations_trainval2017/instances_val2017.json")
data = json.load(file)

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
    else:
        seg_labels_[k] = v

#
# for ann in data["annotations"]:
#     image_id = ann["image_id"]
#     category_id = ann["category_id"]
#     iscrowd = ann["iscrowd"]
#
#     image_name = id_to_image_name[image_id]
#     class_name = class_from_id[category_id]
#     image_path = os.path.join("./archive/val2017", image_name)
#
#     if image_name in image_names_crowd:
#         print(f"ISCROWD This image {image_name} will be skipped")
#         continue
#
#     if image_name in image_names_with_divided_masks:
#         print(f"DIVIDED MASKS This image {image_name} will be skipped")
#         continue
#
#     if iscrowd == 1:
#         if id_to_image_name[image_id] not in image_names_crowd:
#             image_names_crowd.append(id_to_image_name[image_id])
#         continue
#
#     if image_id not in seg_labels:
#         seg_labels[image_id] = []
#
#     arr = ann["segmentation"]
#
#     print(f"Image ID: {image_id}")
#     print(f"Image Name: {id_to_image_name[image_id]}")
#     print(f"Category ID: {category_id}")
#     print(f"Category Name: {class_from_id[category_id]}")
#     print(f"Is Crowd: {iscrowd}")
#
#
#     if len(arr) > 1:
#
#         a = [ np.asarray(x, np.int32).reshape((-1,2)) for x in arr]
#
#         dic = {"mask": a, "category": category_id, "class_name": class_from_id[category_id]}
#
#         if id_to_image_name[image_id] not in image_names_with_divided_masks:
#
#             image_names_with_divided_masks.append(id_to_image_name[image_id])
#         continue
#     elif len(arr) == 1:
#         arr = np.asarray(arr, np.int32)
#         arr = arr.reshape((-1, 2))
#         dic = {"mask": arr, "category": category_id, "class_name": class_from_id[category_id]}
#         seg_labels[image_id].append(dic)
#
#     else:
#         print("Error")
#         break

    #plot_box(image_path, dic["mask"])


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

