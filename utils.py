import numpy as np
import PIL
from PIL import Image
from diffusers.utils import randn_tensor


def view_images(images, num_rows=1, offset_ratio=0.02, save_path=None, show=False):

    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    if show:
        pil_img.show()
    if save_path is not None:
        pil_img.save(save_path)


def plot_grid(w, name="test.png"):
    import matplotlib.pyplot as plt
    import torchvision
    grid_img = torchvision.utils.make_grid(w)
    torchvision.utils.save_image(grid_img, name)
    plt.imshow(grid_img.permute(1,2,0).cpu())
    plt.show()

def normalize(images):
    """
    Normalize an image array to [-1,1].
    """
    return 2.0 * images - 1.0

def denormalize(images):
    """
    Denormalize an image array to [0,1].
    """
    return (images / 2 + 0.5).clamp(0, 1)

import torch
def get_generator(seed):
    return torch.Generator(device="cpu").manual_seed(seed)

def encode_vae_image(vae, image: torch.Tensor, generator: torch.Generator):
    if isinstance(generator, list):
        image_latents = [
            vae.encode(image[i: i + 1]).latent_dist.sample(generator=generator[i])
            for i in range(image.shape[0])
        ]
        image_latents = torch.cat(image_latents, dim=0)
    else:
        image_latents = vae.encode(image).latent_dist.sample(generator=generator)

    image_latents = vae.config.scaling_factor * image_latents

    return image_latents


def get_timesteps(scheduler, num_inference_steps, strength, device):
    """

    This method, get_timesteps, takes in four parameters: scheduler, num_inference_steps, strength, and device.

    scheduler: The scheduler object that contains the timesteps.
    num_inference_steps: The total number of inference steps.
    strength: The coefficient used to calculate the initial timestep.
    device: The device to run the calculations on.

    The method returns two values, timesteps and the number of remaining inference steps.

    The timesteps are calculated based on the scheduler's timesteps, using the initial timestep calculated using the strength coefficient. The initial timestep is the minimum between the product of num_inference_steps and strength, and num_inference_steps itself.

    The t_start is calculated as the maximum between num_inference_steps - init_timestep and 0. This ensures that the t_start does not exceed the total number of inference steps.

    The timesteps are obtained by slicing the scheduler's timesteps starting from t_start multiplied by the scheduler's order.

    Finally, the method returns the timesteps and the difference between the total number of inference steps and t_start.

    """
    # get the original timestep using init_timestep
    init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

    t_start = max(num_inference_steps - init_timestep, 0)
    timesteps = scheduler.timesteps[t_start * scheduler.order:]

    return timesteps, num_inference_steps - t_start

def check_inputs(
        prompt,
        height,
        width,
        strength,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
):
    """
    Check the inputs for the `check_inputs` method.

    Parameters:
    - `prompt`: The prompt to be used for processing. It can be a string or a list of strings.
    - `height`: The height of the input image. It should be divisible by scale factor.
    - `width`: The width of the input image. It should be divisible by scale factor.
    - `strength`: The strength of the processing. Should be a value between 0 and 1, inclusive.
    - `callback_steps`: The number of callback steps. Should be a positive integer.
    - `negative_prompt` (optional): The negative prompt to be used for processing. It can be a string or a list of strings.
    - `prompt_embeds` (optional): The embedding of the prompt. Should be a numpy array of shape (n, m).
    - `negative_prompt_embeds` (optional): The embedding of the negative prompt. Should be a numpy array of shape (n, m).

    Raises:
    - `ValueError`: If the value of `strength` is not in the range [0.0, 1.0].
    - `ValueError`: If `height` or `width` are not divisible by 8.
    - `ValueError`: If `callback_steps` is not a positive integer.
    - `ValueError`: If both `prompt` and `prompt_embeds` are provided.
    - `ValueError`: If neither `prompt` nor `prompt_embeds` are provided.
    - `ValueError`: If `prompt` is not a string or a list.
    - `ValueError`: If both `negative_prompt` and `negative_prompt_embeds` are provided.
    - `ValueError`: If the shapes of `prompt_embeds` and `negative_prompt_embeds` are not the same.

    """

    if strength < 0 or strength > 1:
        raise ValueError(f"The value of strength should in [0.0, 1.0] but is {strength}")

    if height % 8 != 0 or width % 8 != 0:
        raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

    if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
    ):
        raise ValueError(
            f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
            f" {type(callback_steps)}."
        )

    if prompt is not None and prompt_embeds is not None:
        raise ValueError(
            f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
            " only forward one of the two."
        )
    elif prompt is None and prompt_embeds is None:
        raise ValueError(
            "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
        )
    elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
        raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

    if negative_prompt is not None and negative_prompt_embeds is not None:
        raise ValueError(
            f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
            f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
        )

    if prompt_embeds is not None and negative_prompt_embeds is not None:
        if prompt_embeds.shape != negative_prompt_embeds.shape:
            raise ValueError(
                "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                f" {negative_prompt_embeds.shape}."
            )


def prepare_mask_and_masked_image(image, mask, height, width, return_image: bool = False):
    """
    Prepares a pair (image, mask) to be consumed by the Stable Diffusion pipeline. This means that those inputs will be
    converted to ``torch.Tensor`` with shapes ``batch x channels x height x width`` where ``channels`` is ``3`` for the
    ``image`` and ``1`` for the ``mask``.

    The ``image`` will be converted to ``torch.float32`` and normalized to be in ``[-1, 1]``. The ``mask`` will be
    binarized (``mask > 0.5``) and cast to ``torch.float32`` too.

    Args:
        image (Union[np.array, PIL.Image, torch.Tensor]): The image to inpaint.
            It can be a ``PIL.Image``, or a ``height x width x 3`` ``np.array`` or a ``channels x height x width``
            ``torch.Tensor`` or a ``batch x channels x height x width`` ``torch.Tensor``.
        mask (_type_): The mask to apply to the image, i.e. regions to inpaint.
            It can be a ``PIL.Image``, or a ``height x width`` ``np.array`` or a ``1 x height x width``
            ``torch.Tensor`` or a ``batch x 1 x height x width`` ``torch.Tensor``.


    Raises:
        ValueError: ``torch.Tensor`` images should be in the ``[-1, 1]`` range. ValueError: ``torch.Tensor`` mask
        should be in the ``[0, 1]`` range. ValueError: ``mask`` and ``image`` should have the same spatial dimensions.
        TypeError: ``mask`` is a ``torch.Tensor`` but ``image`` is not
            (ot the other way around).

    Returns:
        tuple[torch.Tensor]: The pair (mask, masked_image) as ``torch.Tensor`` with 4
            dimensions: ``batch x channels x height x width``.
    """

    if image is None:
        raise ValueError("`image` input cannot be undefined.")

    if mask is None:
        raise ValueError("`mask_image` input cannot be undefined.")

    if isinstance(image, torch.Tensor):
        if not isinstance(mask, torch.Tensor):
            raise TypeError(f"`image` is a torch.Tensor but `mask` (type: {type(mask)} is not")

        # Batch single image
        if image.ndim == 3:
            assert image.shape[0] == 3, "Image outside a batch should be of shape (3, H, W)"
            image = image.unsqueeze(0)

        # Batch and add channel dim for single mask
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)

        # Batch single mask or add channel dim
        if mask.ndim == 3:
            # Single batched mask, no channel dim or single mask not batched but channel dim
            if mask.shape[0] == 1:
                mask = mask.unsqueeze(0)

            # Batched masks no channel dim
            else:
                mask = mask.unsqueeze(1)

        assert image.ndim == 4 and mask.ndim == 4, "Image and Mask must have 4 dimensions"
        assert image.shape[-2:] == mask.shape[-2:], "Image and Mask must have the same spatial dimensions"
        assert image.shape[0] == mask.shape[0], "Image and Mask must have the same batch size"

        # Check image is in [-1, 1]
        if image.min() < -1 or image.max() > 1:
            raise ValueError("Image should be in [-1, 1] range")

        # Check mask is in [0, 1]
        if mask.min() < 0 or mask.max() > 1:
            raise ValueError("Mask should be in [0, 1] range")

        # Binarize mask
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1

        # Image as float32
        image = image.to(dtype=torch.float32)
    elif isinstance(mask, torch.Tensor):
        raise TypeError(f"`mask` is a torch.Tensor but `image` (type: {type(image)} is not")
    else:
        # preprocess image
        if isinstance(image, (PIL.Image.Image, np.ndarray)):
            image = [image]
        if isinstance(image, list) and isinstance(image[0], PIL.Image.Image):
            # resize all images w.r.t passed height an width
            image = [i.resize((width, height), resample=PIL.Image.LANCZOS) for i in image]
            image = [np.array(i.convert("RGB"))[None, :] for i in image]
            image = np.concatenate(image, axis=0)
        elif isinstance(image, list) and isinstance(image[0], np.ndarray):
            image = np.concatenate([i[None, :] for i in image], axis=0)

        image = image.transpose(0, 3, 1, 2)
        image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

        # preprocess mask
        if isinstance(mask, (PIL.Image.Image, np.ndarray)):
            mask = [mask]

        if isinstance(mask, list) and isinstance(mask[0], PIL.Image.Image):
            mask = [i.resize((width, height), resample=PIL.Image.LANCZOS) for i in mask]
            mask = np.concatenate([np.array(m.convert("L"))[None, None, :] for m in mask], axis=0)
            mask = mask.astype(np.float32) / 255.0
        elif isinstance(mask, list) and isinstance(mask[0], np.ndarray):
            mask = np.concatenate([m[None, None, :] for m in mask], axis=0)

        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5)

    # n.b. ensure backwards compatibility as old function does not return image
    if return_image:
        return mask, masked_image, image

    return mask, masked_image


def prepare_latents(
        vae,
        scheduler,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        vae_scale_factor,
        latents=None,
        image=None,
        timestep=None,
        is_strength_max=True,
        return_noise=False,
        return_image_latents=False,
):
    """

    Prepares the latents for diffusion models.

    :param vae: The VAE model used for encoding images into latents.
    :type vae: VAE object

    :param scheduler: The scheduler used for adding noise to latents during diffusion.
    :type scheduler: Scheduler object

    :param batch_size: The number of samples in each batch.
    :type batch_size: int

    :param num_channels_latents: The number of channels in the latents.
    :type num_channels_latents: int

    :param height: The height of the image.
    :type height: int

    :param width: The width of the image.
    :type width: int

    :param dtype: The dtype of the tensors.
    :type dtype: torch.dtype

    :param device: The device on which to perform the computation.
    :type device: str or torch.device

    :param generator: The random generator used for generating noise tensors.
                      Can be a single generator or a list of generators.
                      If a list of generators is provided, the length of the list must be equal to the batch size.
    :type generator: torch.Generator or list of torch.Generator

    :param vae_scale_factor: The scale factor used for downsampling the image before encoding.
    :type vae_scale_factor: int

    :param latents: The initial latents. If None, it will be initialized based on the is_strength_max parameter.
    :type latents: torch.Tensor or None

    :param image: The input image to encode into latents. Required if is_strength_max is False.
    :type image: PIL.Image or None

    :param timestep: The timestep at which diffusion is performed. Required if is_strength_max is False.
    :type timestep: int or None

    :param is_strength_max: Whether the strength parameter is set to its maximum value (i.e., 1.0).
                            If True, the initial latents will be initialized as pure noise.
                            If False, the initial latents will be initialized as a combination of image + noise.
    :type is_strength_max: bool

    :param return_noise: Whether to return the noise tensor used for initializing the latents.
    :type return_noise: bool

    :param return_image_latents: Whether to return the latents obtained from encoding the input image.
                                 Required if latents is None and is_strength_max is False.
    :type return_image_latents: bool

    :return: A tuple of tensors containing the latents and optionally the noise and image latents.
    :rtype: tuple
    """
    shape = (batch_size, num_channels_latents, height // vae_scale_factor, width // vae_scale_factor)
    if isinstance(generator, list) and len(generator) != batch_size:
        raise ValueError(
            f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
            f" size of {batch_size}. Make sure the batch size matches the length of the generators."
        )

    if (image is None or timestep is None) and not is_strength_max:
        raise ValueError(
            "Since strength < 1. initial latents are to be initialised as a combination of Image + Noise."
            "However, either the image or the noise timestep has not been provided."
        )

    if return_image_latents or (latents is None and not is_strength_max):
        image = image.to(device=device, dtype=dtype)
        image_latents = encode_vae_image(vae,image=image, generator=generator)

    if latents is None:
        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        # if strength is 1. then initialise the latents to noise, else initial to image + noise
        latents = noise if is_strength_max else scheduler.add_noise(image_latents, noise, timestep)
        # if pure noise then scale the initial latents by the  Scheduler's init sigma
        latents = latents * scheduler.init_noise_sigma if is_strength_max else latents
    else:
        noise = latents.to(device)
        latents = noise * scheduler.init_noise_sigma

    outputs = (latents,)

    if return_noise:
        outputs += (noise,)

    if return_image_latents:
        outputs += (image_latents,)

    return outputs
