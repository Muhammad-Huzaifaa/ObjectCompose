# grouped & sorted imports
import argparse
import os

from PIL import ImageFile
from torch.utils.data import DataLoader
from dataset import ImageFolder

ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from torchvision import transforms, datasets
# simplified functions

def save_img(img, file_dir):
    transforms.ToPILImage()(img.cpu()).save(file_dir)

def get_dataloader(dataset, batch_size, shuffle=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def dinov2_vit_base_patch14():
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_lc')
    model = model.to(DEVICE)
    return model

def dino_v2_vit_base_patch14_reg4():
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg_lc')
    model = model.to(DEVICE)
    return model

def dinov2_vit_small_patch14():
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_lc')
    model = model.to(DEVICE)
    return model

def dino_v2_vit_small_patch14_reg4():
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg_lc')
    model = model.to(DEVICE)
    return model

def dinov2_vit_large_patch14():
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_lc')
    model = model.to(DEVICE)
    return model

def dino_v2_vit_large_patch14_reg4():
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg_lc')
    model = model.to(DEVICE)
    return model






if __name__ == "__main__":


    args = argparse.ArgumentParser()
    args.add_argument("--dataset_path", default=r'/path/to/dataset', type=str)
    args.add_argument("--batch_size", default=64, type=int)
    args.add_argument("--save_dir", default="./DINOv2_Results", type=str)
    args = args.parse_args()


    transform = transforms.Compose([
        transforms.Resize(size=518, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size=(518, 518)),
        transforms.ToTensor(),
        transforms.Normalize(mean=torch.tensor([0.4850, 0.4560, 0.4060]), std=torch.tensor([0.2290, 0.2240, 0.2250]))
    ])


    # dataset and model creation
    dataset = ImageFolder(args.dataset_path, transform)
    dataloader = get_dataloader(dataset, args.batch_size)

    os.makedirs(args.save_dir, exist_ok=True)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ALL_CLASSES = sorted(os.listdir(args.dataset_path))


    # only on dinov2 models
    models = [dinov2_vit_base_patch14, dino_v2_vit_base_patch14_reg4, dinov2_vit_small_patch14, dino_v2_vit_small_patch14_reg4,
              dinov2_vit_large_patch14,
              dino_v2_vit_large_patch14_reg4]

    names = ['dinov2_vit_base_patch14', 'dino_v2_vit_base_patch14_reg4', 'dinov2_vit_small_patch14', 'dino_v2_vit_small_patch14_reg4',
             'dinov2_vit_large_patch14', 'dino_v2_vit_large_patch14_reg4']

    accuracy_per_model = []

    with torch.no_grad():
        for i, model in enumerate(models):
            model = model()
            correct = 0
            total = 0

            for batch_number, (images, labels) in enumerate(dataloader):
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                logits = model(images)
                _, predicted = torch.max(logits.data, 1)
                correct += (predicted == labels).sum().item()

                total += labels.size(0)

            accuracy = correct / total
            accuracy_per_model.append(accuracy)
            print(f"Model {names[i]} TF: {transforms}")
            with open(f'{args.save_dir}/dinov2_model_results.txt', 'a') as f:
                f.write(f"Model {names[i]} Accuracy: {accuracy}\n")

        average_accuracy = sum(accuracy_per_model) / len(models)
        print(f"Average Accuracy: {average_accuracy}")
        with open(f'{args.save_dir}/dinov2_model_results.txt', 'a') as f:
            f.write(f"Average Accuracy: {average_accuracy}\n")




