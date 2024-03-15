# grouped & sorted imports
import os

import torch
import torch.nn as nn
from PIL import ImageFile
from torch.utils.data import DataLoader
from dataset import ImageFolder
from torchvision import transforms
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor
import argparse

from robustness import model_utils
from robustness.datasets import ImageNet

ImageFile.LOAD_TRUNCATED_IMAGES = True
from timm import create_model


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

def resnet50_adv():
    model = create_model('resnet50', pretrained=True)
    mean, std = model.default_cfg['mean'], model.default_cfg['std']
    print(f"mean: {mean}, std: {std}")
    model = torch.nn.Sequential(Normalize(ms=[mean, std]), model)
    model = model
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

def robust_model(model, chk_path):
    """
    Loads robust models from robustness package as well as there
    transformations
    """
    imagenet_ds = ImageNet('/')
    model, _ = model_utils.make_and_restore_model(arch=model, dataset=imagenet_ds,
                                                  resume_path=chk_path, parallel=False, add_custom_forward=True)

    return model


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--save_dir', default="output", type=str,
                        help='Where to save the adversarial examples, and other results')
    parser.add_argument('--data_path', default="path/to/dataset", type=str,
                        help='The clean images root directory')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size for the dataloader')


    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = get_parser()
    # Constants grouped
    DATASET_PATH = args.data_path
    BATCH_SIZE = args.batch_size
    SAVE_DIR = args.save_dir

    DATA_SIZE = 256
    CROP_SIZE = 224
    NUM_WORKERS = 4


    os.makedirs(SAVE_DIR, exist_ok=True)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ALL_CLASSES = sorted(os.listdir(DATASET_PATH))

    # dataset and model creation
    dataset = ImageFolder(DATASET_PATH, get_transforms())
    dataloader = get_dataloader(dataset, BATCH_SIZE, NUM_WORKERS)


    models = [vit_tiny_patch16_224, resnet50, vit_small_patch16_224, resnet152d, densenet161, swin_tiny_patch4_window7_224, swin_small_patch4_window7_224]
    names = ['vit_tiny_patch16_224', 'resnet50', 'vit_small_patch16_224', 'resnet152d', 'densenet161', 'swin_tiny_patch4_window7_224', 'swin_small_patch4_window7_224']



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
            print(f"Model {names[i]} Accuracy: {accuracy}")
            with open(f'{SAVE_DIR}/model_results.txt', 'a') as f:
                f.write(f"Model {names[i]} Accuracy: {accuracy}\n")

        average_accuracy = sum(accuracy_per_model) / len(models)
        print(f"Average Accuracy: {average_accuracy}")
        with open(f'{SAVE_DIR}/model_results.txt', 'a') as f:
            f.write(f"Average Accuracy: {average_accuracy}\n")



    pretrained_models_path={"eps_0": "pretrained_models/resnet50_l2_eps0.ckpt",
                            "eps0.5": "pretrained_models/resnet50_linf_eps0.5.ckpt",
                            "eps2.0": "pretrained_models/resnet50_linf_eps2.0.ckpt",
                            "eps4.0": "pretrained_models/resnet50_linf_eps4.0.ckpt",
                            "eps8.0": "pretrained_models/resnet50_linf_eps8.0.ckpt"}



    accuracy_per_model = []

    with torch.no_grad():
        for eps, path in pretrained_models_path.items():


            model = robust_model("resnet50", path)
            model.eval()
            model.to(DEVICE)

            correct = 0
            total = 0

            for i, (images, labels) in enumerate(dataloader):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                preds = model(images, with_image=False)
                _, pre = torch.max(preds, 1)
                correct += torch.sum(pre == labels)
                total += images.shape[0]
                print(total, end="\r")
            accuracy = 100 * correct / total
            accuracy_per_model.append(accuracy)
            print(f"Accuracy: {accuracy}")
            with open(f'{SAVE_DIR}/model_robust.txt', 'a') as f:
                f.write(f"ResNet-50 {eps} Accuracy: {accuracy}\n")

        average_accuracy = sum(accuracy_per_model) / len(pretrained_models_path)
        print(f"Average Accuracy: {average_accuracy}")
        with open(f'{SAVE_DIR}/model_robust.txt', 'a') as f:
            f.write(f"Average Accuracy across Models: {average_accuracy}\n")




    from stylised_models import deit_tiny_distilled_patch16_224, deit_small_distilled_patch16_224

    pretrained_models_path={
        "DeiT-S Distill": [deit_small_distilled_patch16_224, "pretrained_models/deit_s_sin_dist.pth"],
        "DeiT-T Distill": [deit_tiny_distilled_patch16_224, "pretrained_models/deit_t_sin_dist.pth"],
        "DeiT-T": ["deit_tiny_patch16_224","pretrained_models/deit_t_sin.pth"],
                            "DeiT-S": ["deit_small_patch16_224","pretrained_models/deit_s_sin.pth"],

    }



    accuracy_per_model = []

    with torch.no_grad():
        for model_name, values in pretrained_models_path.items():

            if "Distill" in model_name:
                model = values[0]()
            else:
                model = create_model(values[0], pretrained=False)


            checkpoint = torch.load(values[1], map_location='cpu')
            msg = model.load_state_dict(checkpoint['model'], strict=False)
            print(msg)
            model.eval()
            model.to(DEVICE)

            correct = 0
            total = 0

            for i, (images, labels) in enumerate(dataloader):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                preds = model(images)
                _, pre = torch.max(preds, 1)
                correct += torch.sum(pre == labels)
                total += images.shape[0]
                print(total, end="\r")
            accuracy = 100 * correct / total
            accuracy_per_model.append(accuracy)
            print(f"Accuracy: {accuracy}")
            with open(f'{SAVE_DIR}/model_style.txt', 'a') as f:
                f.write(f"Stylised {model} Accuracy: {accuracy}\n")

        average_accuracy = sum(accuracy_per_model) / len(pretrained_models_path)
        print(f"Average Accuracy: {average_accuracy}")
        with open(f'{SAVE_DIR}/model_style.txt', 'a') as f:
            f.write(f"Average Accuracy across Models: {average_accuracy}\n")