import os

import requests
from PIL import Image

from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch
import argparse


def get_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default="coco", type=str, choices=["imagenet", "coco"],)
    parser.add_argument('--data_path', default="Coco_2017/archive/val2017", type=str,
                        help='Path to the images folder of the dataset')

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = get_parser()

    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    # by default `from_pretrained` loads the weights in float32
    # we load in float16 instead to save memory
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    if args.dataset == "imagenet":

        path = args.data_path
        dest = os.path.join("".join(path.split("/")[:-1]), "captions")
        os.makedirs(dest, exist_ok=True)
        folders = sorted(os.listdir(path))

        for folder in folders:
            folder_path = os.path.join(path, folder)
            dest_folder_path = os.path.join(dest, folder)
            os.makedirs(dest_folder_path, exist_ok=True)
            image_names = os.listdir(folder_path)
            for image_name in image_names:
                image_path = os.path.join(folder_path, image_name)
                dest_caption_path = os.path.join(dest_folder_path, f'{image_name.split(".")[0]}.txt')
                image = Image.open(image_path).convert('RGB')

                inputs = processor(image, return_tensors="pt").to(device, torch.float16)

                generated_ids = model.generate(**inputs, max_new_tokens=20)
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                print(generated_text)

                prompt = "this is a picture of"

                inputs = processor(image, text=prompt, return_tensors="pt").to(device, torch.float16)

                generated_ids = model.generate(**inputs, max_new_tokens=20)
                generated_text_ = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                print(f"{prompt} {generated_text_}")

                with open(dest_caption_path, "w") as f:
                    f.write(f"{prompt} {generated_text}")
                    f.write("\n")
                    f.write(generated_text)
                    f.write("\n")
                    f.write(f"{prompt} {generated_text_}")

    elif args.dataset == "coco":
        path = args.data_path
        dest = os.path.join("".join(path.split("/")[:-1]), "captions")
        os.makedirs(dest, exist_ok=True)
        image_names = os.listdir(path)
        for image_name in image_names:
            image_path = os.path.join(path, image_name)
            dest_caption_path = os.path.join(dest, f'{image_name.split(".")[0]}.txt')
            image = Image.open(image_path).convert('RGB')

            inputs = processor(image, return_tensors="pt").to(device, torch.float16)

            generated_ids = model.generate(**inputs, max_new_tokens=20)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            print(generated_text)

            prompt = "this is a picture of"

            inputs = processor(image, text=prompt, return_tensors="pt").to(device, torch.float16)

            generated_ids = model.generate(**inputs, max_new_tokens=20)
            generated_text_ = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            print(f"{prompt} {generated_text_}")

            with open(dest_caption_path, "w") as f:
                f.write(f"{prompt} {generated_text}")
                f.write("\n")
                f.write(generated_text)
                f.write("\n")
                f.write(f"{prompt} {generated_text_}")