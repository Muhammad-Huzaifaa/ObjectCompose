from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch
from torchmetrics.functional.multimodal import clip_score
from dataset_utils.filtered_dataset import FilteredImageNetDataset
from torchvision import transforms
import argparse


def get_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('--clean_data_path', default="create_data/images", type=str,
                        help='Where to save the adversarial examples, and other results')
    parser.add_argument('--gen_data_path', default="create_data/images", type=str,
                        help='The clean images root directory')
    parser.add_argument('--batch_size', default=6, type=int,)

    args = parser.parse_args()

    return args

if __name__ == "__main__":

    args = get_parser()



    clean_data_path = args.clean_data_path
    gen_data_path = args.gen_data_path
    batch_size = args.batch_size


    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])


    clean_dataset = FilteredImageNetDataset(clean_data_path, transform=transform,caption=True)
    clean_dataloader = torch.utils.data.DataLoader(clean_dataset, batch_size=batch_size, shuffle=False, num_workers=1)


    gen_dataset = FilteredImageNetDataset(gen_data_path, transform=transform, caption=True)


    gen_dataloader = torch.utils.data.DataLoader(gen_dataset, batch_size=batch_size, shuffle=False, num_workers=1)


    avg_clean_clip_score = 0
    avg_gen_clip_score = 0
    avg_clip_score_of_clean_on_gen_captions = 0
    avg_clip_score_of_gen_on_clean_captions = 0


    # evaluate avergae clip scores and save them in a text file
    for i, (clean_batch, gen_batch) in enumerate(zip(clean_dataloader, gen_dataloader)):
        l = len(clean_batch[0])
        clean_img, _, clean_label, clean_caption, _, _, _ = clean_batch
        gen_img, _, gen_label, gen_caption, _, _, _ = gen_batch
        clean_caption = list(clean_caption)
        gen_caption = list(gen_caption)

        clean_clip_score = clip_score(clean_img, clean_caption, "openai/clip-vit-base-patch16")
        gen_clip_score = clip_score(gen_img, gen_caption, "openai/clip-vit-base-patch16")

        clip_score_of_clean_on_gen_captions = clip_score(clean_img, gen_caption, "openai/clip-vit-base-patch16")
        clip_score_of_gen_on_clean_captions = clip_score(gen_img, clean_caption, "openai/clip-vit-base-patch16")

        avg_clean_clip_score += clean_clip_score.detach()*l
        avg_gen_clip_score += gen_clip_score.detach()*l
        avg_clip_score_of_clean_on_gen_captions += clip_score_of_clean_on_gen_captions.detach()*l
        avg_clip_score_of_gen_on_clean_captions += clip_score_of_gen_on_clean_captions.detach()*l

        print(f"Batch {i} done")
    avg_clean_clip_score /= len(clean_dataloader)
    avg_gen_clip_score /= len(clean_dataloader)
    avg_clip_score_of_clean_on_gen_captions /= len(clean_dataloader)
    avg_clip_score_of_gen_on_clean_captions /= len(clean_dataloader)

    print(f"Average clean clip score: {avg_clean_clip_score}")
    print(f"Average gen clip score: {avg_gen_clip_score}")
    print(f"Average clip score of clean on gen captions: {avg_clip_score_of_clean_on_gen_captions}")
    print(f"Average clip score of gen on clean captions: {avg_clip_score_of_gen_on_clean_captions}")

    with open(f"{gen_data_path}/clip_scores.txt", "w") as f:
        f.write(f"Average clean clip score: {avg_clean_clip_score}\n")
        f.write(f"Average gen clip score: {avg_gen_clip_score}\n")
        f.write(f"Average clip score of clean on gen captions: {avg_clip_score_of_clean_on_gen_captions}\n")
        f.write(f"Average clip score of gen on clean captions: {avg_clip_score_of_gen_on_clean_captions}\n")

