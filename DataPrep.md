# Data Preparation

## ImageNet-B
The dataset should be in the following format:

```
### ImageNet style


dataset
   images
        class1
             image1.jpeg
             image2.jpeg
             ...
        class2
             image1.jpeg
             image2.jpeg
             ...
        ...
   masks
        class1
             image1.jpeg
             image2.jpeg
             ...
         class2
             image1.jpeg
             image2.jpeg
             ...
         ...
    captions
        class1
             image1.txt
             image2.txt
             ...
         class2
             image1.txt
             image2.txt
             ...
         ...
```

## COCO-DC

```
dataset
   images or val2017       
         image1.jpeg
         image2.jpeg
         ...

   masks
         image1.jpeg
         image2.jpeg
         ...

    captions
         image1.txt
         image2.txt
         ...
    
    annotations
         instances_val2017.json

```


Captions can be generated for ImageNet-B and COCO-DC dataset using BLIP:
```
python dataset_utils/generate_captions_blip.py --dataset imagenet --data_path <>
```
A captions folder will be generated in the dataset folder.


To Download the ImageNet-B dataset go to this link [here](https://drive.google.com/drive/folders/1YMvJvUGSs96CS2XH4CqAnslUqZyiwnyG?usp=sharing)

To Download the ImageNet-B_1k dataset go to this link [here](https://drive.google.com/drive/folders/1NVF6ASPOxnM8cG_0CUJE4gPYp1QlZ5Tc?usp=sharing)

To Download the COCO-DC dataset go to this link [here](https://drive.google.com/drive/folders/1p-ZXRXB4a92P0cUfZwGMu6L192as5BZZ?usp=sharing)

To Download the COCO-DC(classification) dataset go to this link [here](https://drive.google.com/drive/folders/1yHAPUZ3qyxQM_TRrAdUGx3Vkvd9bWxF5?usp=sharing)
