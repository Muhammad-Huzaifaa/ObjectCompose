import argparse
import os
import shutil
from typing import Any, Callable, Dict, List, Optional, Union

import PIL
import numpy as np
import torch
from torch import optim
from PIL import Image
from diffusers import DDIMScheduler
from diffusers.image_processor import VaeImageProcessor
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torchvision import transforms

from dataset_utils import imagenet_label
from dataset_utils.filtered_dataset import FilteredImageNetDataset, FilteredCOCODataset
from dataset_utils.filtered_dataset_coco_classification import FilteredCocoClassification
from distances import LpDistance
from pipelines import StableDiffusionInpaintPipeline
import eval_classification_models as classification_models
from utils import get_generator, encode_vae_image, get_timesteps, check_inputs, prepare_mask_and_masked_image, prepare_latents
from utils import view_images


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--save_dir', default="output", type=str,
                        help='Where to save the adversarial examples, and other results')
    parser.add_argument('--dataset', default="imagenet", type=str, choices=["imagenet", "coco_classification"])
    parser.add_argument('--data_path', default="/l/users/muhammad.huzaifa/866/dataset/filtered_images_resized", type=str,
                        help='The clean images root directory')
    parser.add_argument('--images_per_class', default=10000000, type=int)


    parser.add_argument('--pretrained_diffusion_path',
                        default="stabilityai/stable-diffusion-2-inpainting",
                        type=str,
                        help='Change the path to `stabilityai/stable-diffusion-2-inpainting` if want to use the pretrained model')
    parser.add_argument('--res', default=512, type=int, help='Input image resized resolution')
    parser.add_argument('--diffusion_steps', default=10, type=int, help='Total DDIM sampling steps')
    parser.add_argument('--start_step', default=6, type=int, help='Which DDIM step to start the attack')
    parser.add_argument('--attack_type', default="ensemble", type=str, choices=["text", "latent", "ensemble"])
    parser.add_argument('--guidance', default=7.5, type=float, help='guidance scale of diffusion models')
    parser.add_argument('--prompt', default="A picture of a XXX", type=str, help='prompt')
    parser.add_argument('--background_change', default="caption", type=str, choices=["class_name", "caption", "prompt"])

    parser.add_argument('--apply_mask', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='Whether to leverage pseudo mask for better imperceptibility')
    parser.add_argument('--expand_mask_pixels', default=None, type=int)

    parser.add_argument('--debug', default=False, type=lambda x: (str(x).lower() == 'true'))

    args = parser.parse_args()

    return args



def get_destination_folder(img_path, save_dir, dataset

):
    path = img_path
    image_name = os.path.basename(path)
    if dataset == "imagenet":
        folder_name = os.path.basename(os.path.dirname(path))
        destination_folder = os.path.join(save_dir, "dataset", folder_name)

    elif dataset == "coco":
        destination_folder = os.path.join(save_dir, "dataset", "images")

    if dataset == "coco_classification":
        folder_name = os.path.basename(os.path.dirname(path))
        destination_folder = os.path.join(save_dir, "dataset", folder_name)

    os.makedirs(destination_folder, exist_ok=True)

    return destination_folder, image_name




def prepare_mask_latents(
        vae, mask, masked_image, batch_size, height, width, dtype, device, generator, vae_scale_factor, do_classifier_free_guidance
):
    # resize the mask to latents shape as we concatenate the mask to the latents
    # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
    # and half precision
    mask = torch.nn.functional.interpolate(
        mask, size=(height // vae_scale_factor, width // vae_scale_factor)
    )
    mask = mask.to(device=device, dtype=dtype)

    masked_image = masked_image.to(device=device, dtype=dtype)
    masked_image_latents = encode_vae_image(vae, masked_image, generator=generator)

    # duplicate mask and masked_image_latents for each generation per prompt, using mps friendly method
    if mask.shape[0] < batch_size:
        if not batch_size % mask.shape[0] == 0:
            raise ValueError(
                "The passed mask and the required batch size don't match. Masks are supposed to be duplicated to"
                f" a total batch size of {batch_size}, but {mask.shape[0]} masks were passed. Make sure the number"
                " of masks that you pass is divisible by the total requested batch size."
            )
        mask = mask.repeat(batch_size // mask.shape[0], 1, 1, 1)
    if masked_image_latents.shape[0] < batch_size:
        if not batch_size % masked_image_latents.shape[0] == 0:
            raise ValueError(
                "The passed images and the required batch size don't match. Images are supposed to be duplicated"
                f" to a total batch size of {batch_size}, but {masked_image_latents.shape[0]} images were passed."
                " Make sure the number of images that you pass is divisible by the total requested batch size."
            )
        masked_image_latents = masked_image_latents.repeat(batch_size // masked_image_latents.shape[0], 1, 1, 1)

    mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
    masked_image_latents = (
        torch.cat([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
    )

    # aligning device to prevent device errors when concating it with the latent model input
    masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)
    return mask, masked_image_latents




def create_text_embeddings(text_encoder, input_ids, device):
    embeds = text_encoder(input_ids.to(device))
    return embeds[0]

def encode_prompt(
    tokenizer, text_encoder,
    prompt,
    batch_size,
    device,
    num_images_per_prompt,
    do_classifier_free_guidance,
    negative_prompt=None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
):
    """
    Encode the prompt into embeddings for image generation.

    Args:
        tokenizer (Any): Tokenizer used to encode the text prompts.
        text_encoder (Any): Text encoder model used to create embeddings.
        prompt (str): Prompt for image generation.
        batch_size (int): Number of images to generate per prompt.
        device (str): Device to perform the encoding on.
        num_images_per_prompt (int): Number of images to generate per prompt.
        do_classifier_free_guidance (bool): Whether to perform classifier-free guidance.
        negative_prompt (Optional[str]): Negative prompt for classifier-free guidance. Defaults to None.
        poembeds (Optional[torch.FloatTensor]): Precomputed embeddings for the prompt. Defaults to None.
        negative_prompt_embeds (Optional[torch.FloatTensor]): Precomputed embeddings for the negative prompt. Defaults to None.

    Returns:
        torch.FloatTensor: Encoded embeddings for the prompt.
    """
    text_inputs, uncond_inputs = None, None

    if prompt_embeds is None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids

    if do_classifier_free_guidance and negative_prompt_embeds is None:
        uncond_tokens = [""] * batch_size if negative_prompt is None else negative_prompt
        uncond_inputs = tokenizer(
            uncond_tokens,
            padding="max_length",
            max_length=tokenizer.model_max_length if text_inputs is None else text_inputs.input_ids.shape[1],
            truncation=True,
            return_tensors="pt",
        )

    # Create text embeddings from tokens
    if text_inputs is not None:
        prompt_embeds = create_text_embeddings(text_encoder, text_inputs.input_ids, device)
    if uncond_inputs is not None and do_classifier_free_guidance:
        negative_prompt_embeds = create_text_embeddings(text_encoder, uncond_inputs.input_ids, device)



    bs_embed, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)



    if do_classifier_free_guidance:
        # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
        seq_len = negative_prompt_embeds.shape[1]

        negative_prompt_embeds = negative_prompt_embeds.to(dtype=text_encoder.dtype, device=device)

        negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

    return prompt_embeds




@torch.enable_grad()
def inference(
        model,
        prompt: Union[str, List[str]] = None,
        image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        mask_image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        strength: float = 1.0,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        start_step: int = 7,
        attack: str = "latent",
        label = None,
        apply_mask: bool = False,
):
    device = model._execution_device
    classifier = classification_models.resnet50_adv().to(device).eval()
    classifier.requires_grad_(False)
    model.text_encoder.requires_grad_(False)
    model.vae.requires_grad_(False)
    model.unet.requires_grad_(False)

    # 0. Default height and width to unet
    height = height or model.unet.config.sample_size * model.vae_scale_factor
    width = width or model.unet.config.sample_size * model.vae_scale_factor
    print(f"Height : {height}   Width : {width}")

    do_classifier_free_guidance = guidance_scale > 1.0

    tokenizer = model.tokenizer
    text_encoder = model.text_encoder
    scheduler = model.scheduler
    unet = model.unet
    vae = model.vae


    """
    1. Check inputs
    """
    check_inputs(
        prompt,
        height,
        width,
        strength,
        callback_steps,
        negative_prompt,
        prompt_embeds,
        negative_prompt_embeds,
    )


    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    """
    2. Get the conditional and unconditional embeddings (2*len(prompt), seq_len, embed_dim) e.g., 2x77x1024
    """
    prompt_embeds = encode_prompt(tokenizer, text_encoder,
        prompt,
        batch_size,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
    )

    """
    3. set timesteps e.g if num_inference_steps = 20, if strength=1.0, then timesteps = 
    [951, 901, 851, 801, 751, 701, 651, 601, 551, 501, 451, 401, 351, 301, 251, 201, 151, 101,  51,   1]
    """
    scheduler.set_timesteps(num_inference_steps, device=device)

    timesteps, num_inference_steps = get_timesteps(scheduler,
        num_inference_steps=num_inference_steps, strength=strength, device=device
    )

    # at which timestep to set the initial noise (n.b. 50% if strength is 0.5)
    latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
    # create a boolean to check if the strength is set to 1. if so then initialise the latents with pure noise
    is_strength_max = strength == 1.0

    # 5. Preprocess mask and image
    mask, masked_image, init_image = prepare_mask_and_masked_image(
        image, mask_image, height, width, return_image=True
    )

    resized_mask = mask.clone().detach()
    resized_orig_image = init_image.clone().detach()
    resized_masked_image = masked_image.clone().detach()

    resized_mask = resized_mask.to(device=device)
    resized_orig_image = resized_orig_image.to(device=device)
    resized_masked_image = resized_masked_image.to(device=device)
    
    # 6. Prepare latent variables
    num_channels_latents = model.vae.config.latent_channels
    num_channels_unet = model.unet.config.in_channels
    return_image_latents = num_channels_unet == 4

    # if latents is none, then return a random latent vector
    latents_outputs = prepare_latents( vae, scheduler,
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        model.vae_scale_factor,
        latents,
        image=init_image,
        timestep=latent_timestep,
        is_strength_max=is_strength_max,
        return_noise=True,
        return_image_latents=return_image_latents,
    )

    if return_image_latents:
        latents, noise, image_latents = latents_outputs
    else:
        latents, noise = latents_outputs

    # 7. Prepare mask latent variables
    mask, masked_image_latents = prepare_mask_latents(vae,
        mask,
        masked_image,
        batch_size * num_images_per_prompt,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        model.vae_scale_factor,
        do_classifier_free_guidance,
    )
    init_image = init_image.to(device=device, dtype=masked_image_latents.dtype)
    init_image = encode_vae_image(vae, init_image, generator=generator)

    # 8. Check that sizes of mask, masked image and latents match
    if num_channels_unet == 9:
        # default case for runwayml/stable-diffusion-inpainting
        num_channels_mask = mask.shape[1]
        num_channels_masked_image = masked_image_latents.shape[1]
        if num_channels_latents + num_channels_mask + num_channels_masked_image != unet.config.in_channels:
            raise ValueError(
                f"Incorrect configuration settings! The config of `pipeline.unet`: {unet.config} expects"
                f" {model.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
                f" `num_channels_mask`: {num_channels_mask} + `num_channels_masked_image`: {num_channels_masked_image}"
                f" = {num_channels_latents + num_channels_masked_image + num_channels_mask}. Please verify the config of"
                " `pipeline.unet` or your `mask_image` or `image` input."
            )
    elif num_channels_unet != 4:
        raise ValueError(
            f"The unet {unet.__class__} should have either 4 or 9 input channels, not {unet.config.in_channels}."
        )

    # 9. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    extra_step_kwargs = model.prepare_extra_step_kwargs(generator, eta)

    # 10. Denoising loop
    num_warmup_steps = len(timesteps) - num_inference_steps * scheduler.order   
    
    for i, t in enumerate(timesteps[0:start_step-1]):
        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

        # concat latents, mask, masked_image_latents in the channel dimension
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        if num_channels_unet == 9:
            latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)

        # predict the noise residual
        noise_pred = unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=cross_attention_kwargs,
            return_dict=False,
        )[0]

        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
    
    
    latent = latents 
    if attack == "ensemble":
        latent.requires_grad_(True)
        prompt_embe = prompt_embeds[1].clone().detach().requires_grad_(True)  
        optimizer = optim.AdamW([prompt_embe,latent], lr=10e-2)
    elif attack == "latent":
        latent.requires_grad_(True)
        optimizer = optim.AdamW([latent], lr=10e-2)
    elif attack == "text":
        prompt_embe = prompt_embeds[1].clone().detach().requires_grad_(True)   
        optimizer = optim.AdamW([prompt_embe], lr=10e-2) 

    cross_entro = torch.nn.CrossEntropyLoss()
    
    for _, _ in enumerate(range(20)): 
        latents = latent 
        if attack != "latent":
            prompt_embeds = torch.cat([prompt_embeds[0].unsqueeze(0),prompt_embe.unsqueeze(0)])      
        for i, t in enumerate(timesteps[start_step-1:]):
        
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

            # concat latents, mask, masked_image_latents in the channel dimension
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            if num_channels_unet == 9:
                latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)

            # predict the noise residual
            noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False,
            )[0]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]    
        
        
        if num_channels_unet == 4:
            init_latents_proper = image_latents[:1]
            init_mask = mask[:1]

            if i < len(timesteps) - 1:
                noise_timestep = timesteps[i + 1]
                init_latents_proper = scheduler.add_noise(
                    init_latents_proper, noise, torch.tensor([noise_timestep])
                )

            latents = (1 - init_mask) * init_latents_proper + init_mask * latents

        # call the callback, if provided
        if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % model.scheduler.order == 0):
            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)

        if not output_type == "latent":
            image = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
            if apply_mask:
                image = (image.squeeze(0) * (resized_mask.squeeze(0)) + (1 - resized_mask.squeeze(0)) * resized_orig_image.squeeze(0)).unsqueeze(0)
            
            image, has_nsfw_concept = image, None  #
            # self.run_safety_checker(image, device, prompt_embeds.dtype)
            print("Not running safety checker")
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        out_image = (image / 2 + 0.5).clamp(0, 1).squeeze(0)
        
        out_image = TF.resize(out_image, 256)
        out_image = TF.center_crop(out_image, 224)
            
        pred = classifier(out_image.unsqueeze(0).to(device))
        loss = - cross_entro(pred, label) * 100
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

            
    with torch.no_grad():
        latents = latent
        if attack != "latent":
            prompt_embeds = torch.cat([prompt_embeds[0].unsqueeze(0),prompt_embe.unsqueeze(0)])     
        for i, t in enumerate(timesteps[start_step-1:]):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

            # concat latents, mask, masked_image_latents in the channel dimension
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            if num_channels_unet == 9:
                latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)

            # predict the noise residual
            noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False,
            )[0]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
            
            if num_channels_unet == 4:
                init_latents_proper = image_latents[:1]
                init_mask = mask[:1]

                if i < len(timesteps) - 1:
                    noise_timestep = timesteps[i + 1]
                    init_latents_proper = scheduler.add_noise(
                        init_latents_proper, noise, torch.tensor([noise_timestep])
                    )

                latents = (1 - init_mask) * init_latents_proper + init_mask * latents

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % model.scheduler.order == 0):
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)

    if not output_type == "latent":
        image = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
        if apply_mask:
                image = (image.squeeze(0) * (resized_mask.squeeze(0)) + (1 - resized_mask.squeeze(0)) * resized_orig_image.squeeze(0)).unsqueeze(0)
            
        image, has_nsfw_concept = image, None  #
        # self.run_safety_checker(image, device, prompt_embeds.dtype)
        print("Not running safety checker")
    else:
        image = latents
        has_nsfw_concept = None

    if has_nsfw_concept is None:
        do_denormalize = [True] * image.shape[0]
    else:
        do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]
            
    image = model.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)
    resized_mask = model.image_processor.postprocess(resized_mask, output_type=output_type,
                                                    do_denormalize=[False] * resized_mask.shape[0])
    resized_masked_image = model.image_processor.postprocess(resized_masked_image, output_type=output_type,
                                                            do_denormalize=do_denormalize)
    resized_orig_image = model.image_processor.postprocess(resized_orig_image, output_type=output_type,
                                                        do_denormalize=do_denormalize)    
            
    # Offload last model to CPU
    if hasattr(model, "final_offload_hook") and model.final_offload_hook is not None:
        model.final_offload_hook.offload()

    if not return_dict:
        return (image, has_nsfw_concept)

    return image, resized_mask, resized_masked_image, resized_orig_image


if __name__ == "__main__":

    args = get_parser()

    if args.prompt == "None":
        args.prompt = None
    if args.expand_mask_pixels == 0:
        args.expand_mask_pixels = None

    print(args)
    prompt= args.prompt
    guidance = args.guidance
    apply_mask = args.apply_mask
    diffusion_steps = args.diffusion_steps  # Total DDIM sampling steps.
    res = args.res  # Input image resized resolution.
    save_dir = args.save_dir  # Where to save the adversarial examples, and other results.
    os.makedirs(save_dir, exist_ok=True)
    batch_size = 1




    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Load Diffusion Model
    pretrained_diffusion_path = args.pretrained_diffusion_path
    ldm_stable = StableDiffusionInpaintPipeline.from_pretrained(pretrained_diffusion_path, resume_download=True).to(device)
    #you can use other scheduler
    ldm_stable.scheduler = DDIMScheduler.from_config(ldm_stable.scheduler.config)



    inpaint_images = []
    images = []


    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    transform_mask = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    to_pil = transforms.ToPILImage()
    if args.dataset == "imagenet":
        dataset = FilteredImageNetDataset(args.data_path, transform=transform, transform_mask=transform_mask,
                                           images_per_class=args.images_per_class, expansion_mask_pixels=args.expand_mask_pixels)
    elif args.dataset == "coco":
        dataset = FilteredCOCODataset(args.data_path, transform=transform, transform_mask=transform_mask,
                                       images_per_class=args.images_per_class, expansion_mask_pixels=args.expand_mask_pixels)
    elif args.dataset == "coco_classification":
        dataset = FilteredCocoClassification(args.data_path, transform=transform, transform_mask=transform_mask, split="val",
                                              images_per_class=args.images_per_class, expansion_mask_pixels=args.expand_mask_pixels)
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    L1 = LpDistance(1)
    L2 = LpDistance(2)
    Linf = LpDistance(float("inf"))

    for ind, (tmp_image,mask_img, target, caption, img_path, mask_pth, caption_path) in enumerate(dataloader):


        tmp_image = VaeImageProcessor.normalize(tmp_image)

        img_label = target.long().to(device)


        if args.background_change == "caption":
            prompt = caption[0]


        elif args.background_change == "class_name":
            if args.dataset not in ['imagenet', 'coco_classification']:
                raise ValueError("Dataset not defined")
            if args.dataset == "imagenet":
                prompt =  [args.prompt.replace('XXX', imagenet_label.refined_Label[label.item()]) for label in img_label] if args.prompt else [imagenet_label.refined_Label[label.item()] for label in img_label]
            elif args.dataset == "coco_classification":
                class_name = img_path[0].split("/")[-2]
                class_name = os.path.basename(os.path.dirname(img_path[0]))
                prompt = [args.prompt.replace("XXX", class_name)] if args.prompt else [class_name]

        elif args.background_change == "prompt":
            prompt = args.prompt

        else:
            raise ValueError("Background change not defined")

        print("prompt: ", prompt)

        """
        ldm_stable is the stable diffusion pipeline.
        prompt: ['class_name']
        tmp_image: Bx3x512x512 normalised [-1,1]
        mask_img: BX1X512x512 [0 - 1]
        generator: for deterministic behaviour
        num_inference_steps
        guidance
        """
        inpaint_image, resized_mask,  resized_masked_image, resized_orig_image = inference(model=ldm_stable, prompt=prompt, image=tmp_image, mask_image=mask_img, generator=get_generator(8888),
                           strength=1.0, num_inference_steps=diffusion_steps,
                           guidance_scale=guidance, num_images_per_prompt=1,
                           label = img_label, start_step=args.start_step, attack = args.attack_type,
                            output_type="np", apply_mask=apply_mask)

        resized_masked_image[resized_masked_image == 0.5] = 0
        perturbed_image = inpaint_image*(resized_mask)


        print("L1: {}\tL2: {}\tLinf: {}".format(L1(resized_orig_image, inpaint_image), L2(resized_orig_image, inpaint_image),
                                                Linf(resized_orig_image, inpaint_image)))

        diff_rel = inpaint_image - resized_orig_image
        diff_rel = (diff_rel - diff_rel.min()) / (diff_rel.max() - diff_rel.min()) * 255
        diff_abs = (np.abs(inpaint_image - resized_orig_image) * 255).astype(np.uint8)

        if args.debug:
            for i in range(batch_size):

                view_images(np.concatenate([resized_orig_image[[i], :]*255, resized_masked_image[[i], :]*255, inpaint_image[[i], :]*255, perturbed_image[[i], :]*255, diff_rel[[i], :].clip(0, 255), diff_abs[[i], :].clip(0, 255)]) ,
                            show=False,
                            save_path=args.save_dir + f"/images{ind}_{i}.png",
                            num_rows=2)


        destination_folder, image_name = get_destination_folder(img_path[0], save_dir, args.dataset)
        print("Destination Folder: ", destination_folder)
        os.makedirs(destination_folder, exist_ok=True)
        # save inpaint image
        inpaint_image = Image.fromarray(np.uint8(inpaint_image.squeeze(0)*255))
        inpaint_image.save(os.path.join(destination_folder, image_name))



    if args.dataset == "coco":
        print("******COCO*********")
        source_ann_file = os.path.join(args.data_path, "annotations", "instances_val2017.json")
        dest_ann_folder = os.path.join(save_dir,"dataset","annotations")
        os.makedirs(dest_ann_folder, exist_ok=True)
        dest_ann_file = os.path.join(dest_ann_folder, "instances_val2017.json")
        shutil.copyfile(source_ann_file, dest_ann_file)



