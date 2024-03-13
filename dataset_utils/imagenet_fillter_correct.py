# grouped & sorted imports
import os
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torch
import torch.nn as nn
import shutil
import timm
from timm import create_model

# Constants grouped
DATASET_PATH =  './2'
DESTINATION_FOLDER = "./2"
BATCH_SIZE = 256
DATA_SIZE = 256
CROP_SIZE = 224
NUM_WORKERS = 4

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ALL_CLASSES = sorted(os.listdir(DATASET_PATH))


class Normalize(nn.Module):
    def __init__(self, ms=None):
        super(Normalize, self).__init__()
        if ms == None:
            self.ms = [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)]
        else:
            self.ms = ms

    def forward(self, input):
        x = input.clone()
        for i in range(x.shape[1]):
            x[:, i] = (x[:, i] - self.ms[0][i]) / self.ms[1][i]
        return x

# simplified functions
def get_transforms():

    transform = Compose([
        Resize(DATA_SIZE),
        CenterCrop(CROP_SIZE),
        ToTensor()
    ])

    return transform

def save_img(img, file_dir):
    transforms.ToPILImage()(img.cpu()).save(file_dir)

def get_dataloader(dataset, batch_size, num_workers, shuffle=False):
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)


# dataset and model creation
dataset = datasets.ImageFolder(DATASET_PATH, get_transforms())
dataloader = get_dataloader(dataset, BATCH_SIZE, NUM_WORKERS)



def vit_small_patch16_224():
    model = create_model('vit_small_patch16_224', pretrained=True)
    mean, std = model.default_cfg['mean'], model.default_cfg['std']
    print(f"mean: {mean}, std: {std}")
    model = torch.nn.Sequential(Normalize(ms=[mean, std]), model)
    model = model.to(DEVICE)
    model.eval()
    return model

def resnet50():
    model = create_model('resnet50', pretrained=True)
    mean, std = model.default_cfg['mean'], model.default_cfg['std']
    print(f"mean: {mean}, std: {std}")
    model = torch.nn.Sequential(Normalize(ms=[mean, std]), model)
    model = model.to(DEVICE)
    model.eval()
    return model


def vit_tiny_patch16_224():
    model = create_model('vit_tiny_patch16_224', pretrained=True)
    mean, std = model.default_cfg['mean'], model.default_cfg['std']
    print(f"mean: {mean}, std: {std}")
    model = torch.nn.Sequential(Normalize(ms=[mean, std]), model)
    model = model.to(DEVICE)
    model.eval()
    return model

def resnet152d():
    model = create_model('resnet152d', pretrained=True)
    mean, std = model.default_cfg['mean'], model.default_cfg['std']
    print(f"mean: {mean}, std: {std}")
    model = torch.nn.Sequential(Normalize(ms=[mean, std]), model)
    model = model.to(DEVICE)
    model.eval()
    return model
def densenet161():
    model = create_model('densenet161', pretrained=True)
    mean, std = model.default_cfg['mean'], model.default_cfg['std']
    print(f"mean: {mean}, std: {std}")
    model = torch.nn.Sequential(Normalize(ms=[mean, std]), model)
    model = model.to(DEVICE)
    model.eval()
    return model

def swin_tiny_patch4_window7_224():
    model = create_model('swin_tiny_patch4_window7_224', pretrained=True)
    mean, std = model.default_cfg['mean'], model.default_cfg['std']
    print(f"mean: {mean}, std: {std}")
    model = torch.nn.Sequential(Normalize(ms=[mean, std]), model)
    model = model.to(DEVICE)
    model.eval()
    return model

def swin_small_patch4_window7_224():
    model = create_model('swin_small_patch4_window7_224', pretrained=True)
    mean, std = model.default_cfg['mean'], model.default_cfg['std']
    print(f"mean: {mean}, std: {std}")
    model = torch.nn.Sequential(Normalize(ms=[mean, std]), model)
    model = model.to(DEVICE)
    model.eval()
    return model



models = [vit_tiny_patch16_224, resnet50, vit_small_patch16_224, resnet152d, densenet161, swin_tiny_patch4_window7_224, swin_small_patch4_window7_224]




correct_per_model = 0
total = 0

with torch.no_grad():
    for batch_number, (images, labels) in enumerate(dataloader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        ensemble_predictions = []
        for model in models:
            model  = model()
            logits = model(images)
            _, predicted = torch.max(logits.data, 1)
            ensemble_predictions.append(predicted)
        ensemble_predictions = torch.stack(ensemble_predictions)

        correct_per_model += (ensemble_predictions == labels.view(1, -1).expand_as(ensemble_predictions)).sum(dim=1)

        total += labels.size(0)

        for idx in range(images.shape[0]):

            # if all the  predictions for this image are correct across all the models
            correct=True
            for prediction in ensemble_predictions:
                if prediction[idx] == labels[idx]:
                    print("...")
                else:
                    correct=False
                    break
            if not correct:
                continue
            else:
                source_class = ALL_CLASSES[labels[idx]]
                destination_folder = os.path.join(DESTINATION_FOLDER, source_class)

                if not os.path.exists(destination_folder):
                    os.makedirs(destination_folder)

                destination_path = os.path.join(destination_folder, f'{dataset.imgs[batch_number * BATCH_SIZE + idx][0].split("/")[-1]}')
                source_path = os.path.join(DATASET_PATH, source_class, dataset.imgs[batch_number * BATCH_SIZE + idx][0].split("/")[-1])
                print(source_path, destination_path)
                shutil.copy(source_path, destination_path)

        print(total)

print(f"Accuracy: {correct_per_model / total}")