# **ObjectCompose: Evaluating Resilience of Vision-Based Models on Object-to-Background Compositional Changes**

[Hashmat Shadab Malik*](https://github.com/HashmatShadab), 
[ Muhammad Huzaifa*](https://github.com/Muhammad-Huzaifaa),
[Muzammal Naseer](https://scholar.google.ch/citations?user=tM9xKA8AAAAJ&hl=en),
[Salman Khan](https://salman-h-khan.github.io),
and [Fahad Shahbaz Khan](https://scholar.google.es/citations?user=zvaeYnUAAAAJ&hl=en)

[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2403.04701)
[![Website](https://img.shields.io/badge/Project-Website-87CEEB)](https://muhammad-huzaifaa.github.io/ObjectCompose/)

[//]: # ([![Video]&#40;https://img.shields.io/badge/Video-Presentation-F9D371&#41;]&#40;https://drive.google.com/file/d/1ECkp_lbMj5Pz7RX_GgEvWWDHf5PUXlFd/view?usp=share_link&#41;)

[//]: # ([![slides]&#40;https://img.shields.io/badge/Poster-PDF-87CEEB&#41;]&#40;https://drive.google.com/file/d/1neYZca0KRIBCu5R6P78aQMYJa7R2aTFs/view?usp=share_link&#41;)
[//]: # ([![slides]&#40;https://img.shields.io/badge/Presentation-Slides-B762C1&#41;]&#40;https://drive.google.com/file/d/1wRgCs2uBO0p10FC75BKDUEdggz_GO9Oq/view?usp=share_link&#41;)

<hr />

# :rocket: News
* **(March XX, 2024)**
  * Pre-trained models and evaluation codes  are released.

<hr />

> **Abstract:** *Given the large-scale multi-modal training of recent vision-based models and their generalization capabilities, understanding the extent of their robustness is critical for their real-world deployment. In this work, we evaluate the resilience of current vision-based models against diverse object-to-background context variations. The majority of robustness evaluation methods have introduced synthetic datasets to induce changes to object characteristics (viewpoints, scale, color) or utilized image transformation techniques (adversarial changes, common corruptions) on real images to simulate shifts in distributions. Recent works have explored leveraging large language models and diffusion models to generate changes in the background. However, these methods either lack in offering control over the changes to be made or distort the object semantics, making them unsuitable for the task. Our method, on the other hand, can induce diverse object-to-background changes while preserving the original semantics and appearance of the object. To achieve this goal, we harness the generative capabilities of text-to-image, image-to-text, and image-to-segment models to automatically generate a broad spectrum of object-to-background changes. We induce both natural and adversarial background changes by either modifying the textual prompts or optimizing the latents and textual embedding of text-to-image models. This allows us to quantify the role of background context in understanding the robustness and generalization of deep neural networks. We produce various versions of standard vision datasets (ImageNet, COCO), incorporating either diverse and realistic backgrounds into the images or introducing color, texture, and adversarial changes in the background. We conduct extensive experiment to analyze the robustness of vision-based models against object-to-background context variations across diverse tasks.* 


## Contents

1) [Highlights](#Highlights) 
2) [Installation](#Installation)
3) [Dataset Preparation](#Dataset-Preparation)
4) [Background Generation](#Background-Generation)
5) [Datasets](#Datasets)
6) [Evaluation](#Evaluation)
7) [Results](#Results)

<hr>
<hr>





## Highlights

<sup>([top](#contents))</sup>

![main figure](assets/concept_fig.jpg)
>ObjectCompose consists of an inpainting-based diffusion model to generate the counterfactual background of an image. The object mask is obtained from a segmentation model (SAM) by providing the class label as an input prompt. The segmentation mask, along with the original image caption (generated via BLIP-2) is then processed through the diffusion model. For generating adversarial examples, both the latent and conditional embedding are optimized during the denoising process.



## Installation

<sup>([top](#contents))</sup>

   ```
   conda create -n objcomp python==3.11.3
   conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
   pip install -r requirements.txt
   pip install git+https://github.com/openai/CLIP.git
   pip install git+https://github.com/huggingface/transformers.git
   ```
`diffusers==0.18.2` is used in this codebase.



## Dataset Preparation

<sup>([top](#contents))</sup>

Please find data preparation instructions in  [`DataPrep.md`](DataPrep.md).







## Background Generation

<sup>([top](#contents))</sup>

**On ImageNet dataset, run the following command to generate the background changes:**
```
python obj_compose.py --dataset imagenet --data_path </path/to/image/folder> --save_dir </path/to/save/generated/dataset> --diffusion_steps <> --guidance <> --background_change <>  
```
This would generate the new dataset in the ImageNet format and save it in the save_dir. `background_change` can be set to `caption`, `class_name`, `prompt`

**On COCO dataset, run the following command to generate the background changes:**
```
python obj_compose.py --dataset coco --data_path <> --save_dir <> --diffusion_steps <> --guidance <> --prompt <>  
```
This would generate the new dataset in the COCO format and save it in the save_dir. `background_change` can be set to `caption`, `prompt`


- `--images_per_class`: This argument is used for the **ImageNet** and **COCO_Classification** datasets to specify the number of images per class.
- `--expand_mask_pixels`: This argument can be used for **all** the datasets to expand the size of the mask covering the object of interest.
- `--prompt`: This argument determines whether to use a manual prompt.

If you want to use a prompt and the class name together, pass the `--prompt` argument in the following format: `--prompt="A photo of a XXX in a red background"`
.`XXX` will be replaced by the class name.


##  Datasets
<sup>([top](#contents))</sup>

### Background Generation by ObjectCompose
 <table>
<tr>
<th>ImageNet-B</th>
<th>ImageNet-B_1k</th>
<th>COCO-DC (Classification)</th>
<th>COCO-DC (Detection)</th>
</tr>
<tr>
<td>

| Category     | Link |
|:-------------|:----:|
| Original     | [Link]()     |
| Class Name   | [Link]()     |
| BLiP Caption | [Link]()     |
| Color        | [Link]()     |
| Texture      | [Link]()     |
| Adversarial  | [Link]()     |

</td>
<td>

| Category     | Link |
|:-------------|:----:|
| Original     | [Link]()     |
| Class Name   | [Link]()     |
| BLiP Caption |  [Link]()    |
| Color        | [Link]()     |
| Texture      | [Link]()     |
| Adversarial  | [Link]()     |

</td>


<td>

| Category     | Link |
|:-------------|:----:|
| Original     | [Link](https://drive.google.com/drive/folders/1yHAPUZ3qyxQM_TRrAdUGx3Vkvd9bWxF5?usp=sharing)     |
| BLiP Caption | [Link](https://drive.google.com/drive/folders/1AI_rAwszy_3WiXc0O7WRz1aecthcTfVy?usp=sharing)     |
| Color        | [Link](https://drive.google.com/drive/folders/1XLMk2ewzZh59mTovpwzIMpzuBIsW4uej?usp=sharing)     |
| Texture      |  [Link](https://drive.google.com/drive/folders/1OkxgED2pGEGDZ7c-3CA_j5GUAfoT7yiL?usp=sharing)    |
| Adversarial  | [Link](https://drive.google.com/drive/folders/1A-43BlPSs3cV97Adczv74AJaPJScuj15?usp=sharing)     |

</td>
<td>

| Category     | Link |
|:-------------|:----:|
| Original     | [Link](https://drive.google.com/drive/folders/1p-ZXRXB4a92P0cUfZwGMu6L192as5BZZ?usp=sharing)     |
| BLiP Caption | [Link](https://drive.google.com/drive/folders/1IpAcP7-oi_Eb67WlotxiyzzJLx-1-Zxm?usp=sharing)     |
| Color        | [Link](https://drive.google.com/drive/folders/1u3-apoz-leI-xowGeZpQVUuoORMVPFRU?usp=sharing)     |
| Texture      | [Link](https://drive.google.com/drive/folders/12nmZCNv9Rsa_9zD5N08rm5NMo40XoqUz?usp=sharing)     |
| Adversarial  | [Link](https://drive.google.com/drive/folders/1w6D7rVEvejlrkfY61IXmlYkUPiYLJcr2?usp=sharing)     |

</td>
</tr>

</table>

### Background Generation by Other Methods on ImageNet-B_1k dataset

<table>
<tr valign="top">
<td>

**1. ImageNet-E**

| ImageNet-E                     |      Link       |
|:----------------------------|:---------------:|
| Original                    |    [Link]()     | 
| ImageNet-E ($`λ=-20`$)      |    [Link]()     |
| ImageNet-E ($`λ=20`$)       |    [Link]()     |
| ImageNet-E ($`λ_{adv}=20`$) |    [Link]()     |

</td>
<td>

**2. LANCE**

| LANCE                       |   Link   |
|:----------------------------|:--------:|
| Original                    | [Link]() | 
| LANCE                       | [Link]() |

</td>
</tr>
</table>


## Evaluation

<sup>([top](#contents))</sup>

### 1. Classification On imageNet Dataset

To evaluate classifier models on background changes generated on ImageNet-B and ImageNet-B_1k datasets, run the following command:

```python
python eval_classification_models.py --data_path </path/to/dataset> --batch_size <batch_size> --save_dir <folder/to/save/results>
```
This will evaluate the dataset on:
1. Naturally trained models
2. Adversarially trained models
3. Stylized models

Download the adversarial and stylized pretrained models from the respective links and save them in the `pretrained_models` folder. 
1. Adversarial Pretrained Models 
  
    Download from [here](https://huggingface.co/madrylab/robust-imagenet-models).
      - `resnet50_l2_eps0.ckpt`
      - `resnet50_linf_eps0.5.ckpt`
      - `resnet50_linf_eps2.0.ckpt`
      - `resnet50_linf_eps4.0.ckpt`
      - `resnet50_linf_eps8.0.ckpt`
2. Stylised Pretrained Models
        
   Download from [here](https://github.com/Muzammal-Naseer/IPViT/tree/main).
   - `deit_s_sin_dist.pth`
   - `deit_t_sin_dist.pth`
   - `deit_t_sin.pth`
   - `deit_s_sin.pth`

To evaluate CLIP models on background changes generated on ImageNet-B and ImageNet-B_1k datasets, run the following command:

```python
python eval_clip_models.py --data_path </path/to/dataset> --batch_size <batch_size> --save_dir <folder/to/save/results>
```
This will evaluate the dataset on across different CLIP models.


### 2. Classification On COCO-DC Dataset

Creating COCO classification dataset and Training/evaluating on classification models
1. Pass the image folder and annotation path in the dataset_utils/coco/coco_classification.py file
2. This would create two folders: one with images belonging to different classes and the other folder with masks of images belonging to different classes.

COCO-DC (classification) dataset for training classifiers can be downloaded from [here](https://drive.google.com/drive/folders/1WeLor6jhR7QG1BEvvvSgzRud62h5sdvR?usp=sharing)

After downloading the dataset, run the following command to train classification models:

```
cd evaluations/coco_classification

python train.py /path/to/images/folder --pretrained --num-classes 80
```
Pretrained weights of available models can be downloaded from [here](https://drive.google.com/drive/folders/1OLslQ2nhLxLW1lrhitd15vsmGrLP6qgH?usp=sharing)

- `resnet50`
- `vit_tiny_patch16_224`
- `vit_small_patch16_224`
- `swin_tiny_patch4_window7_224`
- `swin_small_patch4_window7_224`
- `densenet161`
- `resnet152`


To evaluate classifier models on background changes generated on COCO-DC(classification), run the following command:

```python
cd evaluations/coco_classification

python eval.py </path/to/dataset> --model <model_name> --pretrained --num_classes 80 --experiment <exp_name> --resume <path/to/ckpts> 
--output <folder/path/to/save/results>> --save_dir <folder/to/save/results>
```


To evaluate CLIP models on background changes generated on ImageNet-B and ImageNet-B_1k datasets, run the following command:

```python
cd evaluations/coco_classification
python eval_clip_models.py --data_path </path/to/dataset> --batch_size <batch_size> --save_dir <folder/to/save/results>
```
This will evaluate the dataset on across different CLIP models.


### 3. Detection On COCO-DC Dataset

Use the [DETR codebase](https://github.com/facebookresearch/detr) to evaluate the detection models on COCO-DC dataset. Run the following command:
```
python main.py --batch_size 2 --no_aux_loss --eval --resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth --coco_path /path/to/dataset --output_dir /path/to/output/dir 

```



### 4. Evaluating CLIP score on the generated dataset:

1. First generate captions for the generated dataset using test.ipynb and save them in the Captions folder.
2. Run eval_clip_models.py and set the data path for generated and original dataset.



## Results
<sup>([top](#contents))</sup>



Results across Transformers and CNN models on background variations across ImageNet-B_1k dataset. 

| **Datasets**                | **ViT-T** | **ViT-S** | **Swin-T** | **Swin-S** | **Res-50** | **Res-152** | **Dense-161** | **Average** |
|-----------------------------|--- |--- |--- |--- |--- |--- |--- |--- |
| Original                    | 95.5 | 97.5  |97.9| 98.3| 98.5  | 99.1 | 97.2|97.71 |
| ImageNet-E ($`λ=-20`$)      | 91.3 | 94.5   | 96.5 | 97.7 | 96.0 | 97.6 | 95.4 | 95.50 |
| ImageNet-E ($`λ=20`$)       | 90.4 | 94.5   | 95.9 | 97.4 | 95.4 | 97.4 | 95.0 | 95.19 |
| ImageNet-E ($`λ_{adv}=20`$) | 82.8 | 88.8   | 90.7 | 92.8 | 91.6 | 94.2 | 90.4 | 90.21 |
| LANCE                       | 80.0 | 83.8   | 87.6 | 87.7 | 86.1 | 87.4 | 85.1 | 85.38 |
| Class label                 | 90.5 | 94.0 | 95.1 | 95.4 | 96.7  | 96.5 | 94.7 | 94.70 |
| BLIP-2 Caption              | 85.5 | 89.1 | 91.9 | 92.1 | 93.9 | 94.5 | 90.6 | 91.08 |
| Color                       | 67.1 | 83.8 | 85.8 | 86.1 | 88.2  | 91.7 | 80.9 | **83.37** |
| Texture                     | 64.7 | 80.4 | 84.1 | 85.8 | 85.5 | 90.1 | 80.3 | **81.55** |
| Adversarial                 | 18.4 | 32.1 | 25.0 | 31.7 | 2.0  | 14.0 | 28.0 | **21.65** |





Results across CLIP models on background variations across ImageNet-B_1k dataset.

| **Datasets**                     | **ViT-B/32** | **ViT-B/16** | **ViT-L/14** | **Res50** | **Res101**| **Res50x4** | **Res50x16** | **Average** |
|-----------------------------|--- |--- |--- |--- |--- |--- |--- |--- |
| Original                     | 73.90 | 79.40 | 87.79 | 70.69 | 71.80 | 76.29 | 82.19 | 77.43   |
| ImageNet-E ($`λ=-20`$)       | 69.79 | 76.70 | 82.89 | 67.80 | 69.99| 72.70 | 77.00 | 73.83   |
| ImageNet-E ($`λ=20`$)        | 67.97 | 76.16 | 82.12 | 67.37 | 39.89 | 72.62 | 77.07 | 73.31   |
| ImageNet-E ($`λ_{adv}=-20`$) | 62.82 | 70.50 | 77.57 | 59.98 | 65.85 | 67.07 | 67.07 | 68.23   |
| LANCE                        | 54.99 | 54.19 | 57.48 | 58.05 | 60.02 | 60.39 | 73.37 | 59.78   |
| Class label                  | 78.49 | 83.66 | 81.58 | 76.60 | 77.00 | 82.09 | 84.50 | 80.55   |
| BLIP-2 Caption               | 68.79 | 72.29 | 71.49 | 65.20 | 68.40 | 71.20 | 75.40 | 70.40   |
| Color                        | 48.30 | 61.00 | 69.51 | 50.50 | 54.80 | 60.30 | 69.28 | **59.14**   |
| Texture                      | 49.60 | 62.41 | 58.88 | 51.69 | 53.20 | 60.79 | 67.49 | **57.71**   |
| Adversarial                  | 25.5 | 34.89 | 48.19 | 18.29 | 24.40 | 30.29 | 48.49 | **32.87**   |





## Citation
If you use our work, please consider citing:
```bibtex 
@article{malik2024objectcompose,
  title={ObjectCompose: Evaluating Resilience of Vision-Based Models on Object-to-Background Compositional Changes},
  author={Malik, Hashmat Shadab and Huzaifa, Muhammad and Naseer, Muzammal and Khan, Salman and Khan, Fahad Shahbaz},
  journal={arXiv preprint arXiv:2403.04701},
  year={2024}
}
```

<hr />

## Contact
Should you have any question, please create an issue on this repository or contact at muhammad.huzaifa@mbzuai.ac.ae

<hr />

## References
Our code is based on [diffusers](https://github.com/huggingface/diffusers) and [DiffAttack](https://github.com/windvchen/diffattack). We thank them for open-sourcing their codebase.