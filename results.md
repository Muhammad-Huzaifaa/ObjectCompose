#### Default Diffusion Parameters:

| Inference Steps | Guidance | Strength |
|-----------------|----------|----------|
| 20              | 7.5      | 1.0      |

## Classification Results

1. ### Prompts Ablations
| Prompt                                                    | Classification Models | CLIP Models      | Adversarially Trained Models | Stylised ImageNet Models |
|-----------------------------------------------------------|-----------------------|------------------|------------------------------|--------------------------|
| "This is a picture of class name"             | 99.86% to 95.84%      | 82.23% to 82.92% | 91.21% to 88.32%             | 75.42% to 74.67%         |
| blip-2 Caption                                            | 99.86% to 92.13%      | 82.23% to 72.22% | 91.21% to 83.27%             | 75.42% to 68.80%         |
| "This is a photo of textures background"                  | 99.86% to 88.96%      | 82.23% to 66.67% | 91.21% to 75.76%             | 75.42% to 58.39%         |
| "This is a photo of intricately rich textures background" | 99.86% to 86.57% | 82.23% to 65.69% | 91.21% to 72.90%             | 75.42% to 52.54%         |
| "This is a photo of colorful textures background"         | 99.86% to 83.84%    | 82.23% to 60.78% | 91.21% to 65.35%             | 75.42% to 44.13%         |
| "This is a photo of distorted textures background"        | 99.86% to 80.23%   | 82.23% to 59.79% | 91.21% to 63.29%             | 75.42% to 41.90%         |
| "class Name" + against a vivid red background             | 99.86% to 90.98%   | 82.23% to 70.07% | 91.21% to 79.40%             | 75.42% to 62.97%         |
| "class Name" + against a vivid green background           | 99.86% to 91.69%   | 82.23% to 71.38% | 91.21% to 82.27%             | 75.42% to 65.28%         |
| "class Name" + against a vivid blue background            | 99.86% to 91.43%   | 82.23% to 71.05% | 91.21% to 82.19%             | 75.42% to 63.83%         |
| "class Name" + against a vivid colorful background        | 99.86% to 90.96%   | 82.23% to 69.79% | 91.21% to 79.82%             | 75.42% to 61.76%         |
| This is a photo of a vivid red background             | 99.86% to 87.72%   | 82.23% to 64.85% | 91.21% to 70.15%             | 75.42% to 54.19%         |
| This is a photo of a vivid green background           | 99.86% to 87.94%   | 82.23% to 65.49% | 91.21% to 74.24%             | 75.42% to 54.25%         |
| This is a photo of a vivid blue background            | 99.86% to 87.81%   | 82.23% to 65.03% | 91.21% to 72.01%             | 75.42% to 49.33%         |
| This is a photo of a vivid colorful background        | 99.86% to 85.19%   | 82.23% to 61.18% | 91.21% to 63.89%             | 75.42% to 45.08%         |




2. ### Adversarial Attack

| Attack           | Classification Models | CLIP Models      | Adversarially Trained Models | Stylised ImageNet Models |
|------------------|-----------------------|------------------|------------------------------|--------------------------|
| Latent Attack    | 99.86% to 35.1%       | 82.23% to 44.68% | 91.21% to 30.64%             | 75.42% to 22.62%         |
| Prompt Attack    | 99.86% to 37.17%      | 82.23% to 38.23% | 91.21% to 30.86%             | 75.42% to 20.05          |
| Ensemble  Attack | 99.86% to 21.65%      | 82.23% to 32.87% | 91.21% to 18.6%              | 75.42% to 11.9%          |

BLIP-2 captions are used as prompts for the adversarial setting

## Object Detection Results
