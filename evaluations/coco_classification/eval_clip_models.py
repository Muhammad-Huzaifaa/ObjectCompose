import os
import numpy as np
import torch
import clip
from tqdm.notebook import tqdm
import torchvision
import argparse

clip.available_models()


def set_seed(seed):
    """
    Function for setting the seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class DummyArgs:
    pass


def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lr, warmup_length, steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        assign_learning_rate(optimizer, lr)
        return lr

    return _lr_adjuster


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"



def zeroshot_classifier(classnames, templates, clip_model):
    with torch.no_grad():
        zeroshot_weights = []
        text_prompts = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates]
            text_prompts.append(texts)  # format with class
            texts = clip.tokenize(texts).cuda()  # tokenize
            class_embeddings = clip_model.encode_text(texts)  # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)

        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights, text_prompts


def evaluate_zs(model, zs_weights, dataloader):
    device = "cuda"
    logit_scale = 100.0
    top1 = AverageMeter("ACC@1", ":6.2f")
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ zs_weights).softmax(dim=-1)
            logits = similarity * logit_scale
            acc1 = accuracy(logits, labels, topk=(1,))
            top1.update(acc1[0].item(), len(labels))

    print(
        f"Zero-shot CLIP top-1 accuracy : {top1.avg:.2f}"
    )
    return top1.avg


def get_clip_model(name):
    clip_model, preprocess = clip.load(name)
    clip_model.eval()
    return clip_model, preprocess

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--save_dir', default="output", type=str,
                        help='Where to save the adversarial examples, and other results')
    parser.add_argument('--data_path', default="path/to/dataset", type=str,
                        help='The clean images root directory')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size for the dataloader')


    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = get_parser()
    model_names = ['RN50',
                   'RN101',
                   'RN50x4',
                   'RN50x16',
                   'ViT-B/32',
                   'ViT-B/16',
                   'ViT-L/14', ]


    data_path = args.data_path
    SAVE_DIR = args.save_dir
    os.makedirs(SAVE_DIR, exist_ok=True)

    class_name_to_id = {'airplane': 0,
                        'apple': 1,
                        'backpack': 2,
                        'banana': 3,
                        'baseball bat': 4,
                        'baseball glove': 5,
                        'bear': 6,
                        'bed': 7,
                        'bench': 8,
                        'bicycle': 9,
                        'bird': 10,
                        'boat': 11,
                        'book': 12,
                        'bottle': 13,
                        'bowl': 14,
                        'broccoli': 15,
                        'bus': 16,
                        'cake': 17,
                        'car': 18,
                        'carrot': 19,
                        'cat': 20,
                        'cell phone': 21,
                        'chair': 22,
                        'clock': 23,
                        'couch': 24,
                        'cow': 25,
                        'cup': 26,
                        'dining table': 27,
                        'dog': 28,
                        'donut': 29,
                        'elephant': 30,
                        'fire hydrant': 31,
                        'fork': 32,
                        'frisbee': 33,
                        'giraffe': 34,
                        'hair drier': 35,
                        'handbag': 36,
                        'horse': 37,
                        'hot dog': 38,
                        'keyboard': 39,
                        'kite': 40,
                        'knife': 41,
                        'laptop': 42,
                        'microwave': 43,
                        'motorcycle': 44,
                        'mouse': 45,
                        'orange': 46,
                        'oven': 47,
                        'parking meter': 48,
                        'person': 49,
                        'pizza': 50,
                        'potted plant': 51,
                        'refrigerator': 52,
                        'remote': 53,
                        'sandwich': 54,
                        'scissors': 55,
                        'sheep': 56,
                        'sink': 57,
                        'skateboard': 58,
                        'skis': 59,
                        'snowboard': 60,
                        'spoon': 61,
                        'sports ball': 62,
                        'stop sign': 63,
                        'suitcase': 64,
                        'surfboard': 65,
                        'teddy bear': 66,
                        'tennis racket': 67,
                        'tie': 68,
                        'toaster': 69,
                        'toilet': 70,
                        'toothbrush': 71,
                        'traffic light': 72,
                        'train': 73,
                        'truck': 74,
                        'tv': 75,
                        'umbrella': 76,
                        'vase': 77,
                        'wine glass': 78,
                        'zebra': 79}

    id_class_names = {v: k for k, v in class_name_to_id.items()}
    coco_classes = [id_class_names[i] for i in range(80)]

    imagenet_templates = [
        'a bad photo of a {}.',
        'a photo of many {}.',
        'a sculpture of a {}.',
        'a photo of the hard to see {}.',
        'a low resolution photo of the {}.',
        'a rendering of a {}.',
        'graffiti of a {}.',
        'a bad photo of the {}.',
        'a cropped photo of the {}.',
        'a tattoo of a {}.',
        'the embroidered {}.',
        'a photo of a hard to see {}.',
        'a bright photo of a {}.',
        'a photo of a clean {}.',
        'a photo of a dirty {}.',
        'a dark photo of the {}.',
        'a drawing of a {}.',
        'a photo of my {}.',
        'the plastic {}.',
        'a photo of the cool {}.',
        'a close-up photo of a {}.',
        'a black and white photo of the {}.',
        'a painting of the {}.',
        'a painting of a {}.',
        'a pixelated photo of the {}.',
        'a sculpture of the {}.',
        'a bright photo of the {}.',
        'a cropped photo of a {}.',
        'a plastic {}.',
        'a photo of the dirty {}.',
        'a jpeg corrupted photo of a {}.',
        'a blurry photo of the {}.',
        'a photo of the {}.',
        'a good photo of the {}.',
        'a rendering of the {}.',
        'a {} in a video game.',
        'a photo of one {}.',
        'a doodle of a {}.',
        'a close-up photo of the {}.',
        'a photo of a {}.',
        'the origami {}.',
        'the {} in a video game.',
        'a sketch of a {}.',
        'a doodle of the {}.',
        'a origami {}.',
        'a low resolution photo of a {}.',
        'the toy {}.',
        'a rendition of the {}.',
        'a photo of the clean {}.',
        'a photo of a large {}.',
        'a rendition of a {}.',
        'a photo of a nice {}.',
        'a photo of a weird {}.',
        'a blurry photo of a {}.',
        'a cartoon {}.',
        'art of a {}.',
        'a sketch of the {}.',
        'a embroidered {}.',
        'a pixelated photo of a {}.',
        'itap of the {}.',
        'a jpeg corrupted photo of the {}.',
        'a good photo of a {}.',
        'a plushie {}.',
        'a photo of the nice {}.',
        'a photo of the small {}.',
        'a photo of the weird {}.',
        'the cartoon {}.',
        'art of the {}.',
        'a drawing of the {}.',
        'a photo of the large {}.',
        'a black and white photo of a {}.',
        'the plushie {}.',
        'a dark photo of a {}.',
        'itap of a {}.',
        'graffiti of the {}.',
        'a toy {}.',
        'itap of my {}.',
        'a photo of a cool {}.',
        'a photo of a small {}.',
        'a tattoo of the {}.',
    ]
    coco_templates = ["a photo of a {}."]
    print(f"{len(coco_classes)} classes, {len(coco_templates)} templates")

    accuracy_per_model = []
    from coco_classification_dataset import CocoClassification
    for model_name in model_names:
        print(f"Model: {model_name}")
        clip_model, preprocess = get_clip_model(model_name)
        # imagenet_dataset = torchvision.datasets.ImageFolder(root=data_path,
        #                                                     transform=preprocess)
        coco_dataset =  CocoClassification(data_path, transform=preprocess)
        imagenet_loader = torch.utils.data.DataLoader(coco_dataset, batch_size=args.batch_size, num_workers=2)
        zeroshot_weights, text_prompts = zeroshot_classifier(coco_classes, coco_templates, clip_model)
        acc = evaluate_zs(clip_model, zeroshot_weights, imagenet_loader)
        accuracy_per_model.append(acc)
        print(f"Model {model_name} Accuracy: {acc}")
        with open(f'{SAVE_DIR}/model_results.txt', 'a') as f:
            f.write(f"Model {model_name} Accuracy: {acc}\n")

    average_accuracy = sum(accuracy_per_model) / len(model_names)
    print(f"Average Accuracy: {average_accuracy}")
    with open(f'{SAVE_DIR}/model_results.txt', 'a') as f:
        f.write(f"Average Accuracy: {average_accuracy}\n")