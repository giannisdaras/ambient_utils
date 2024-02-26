import torch
from datasets import Dataset
from torchvision import transforms
import PIL
import inspect
from typing import List, Optional, Union
import copy
from diffusers.training_utils import EMAModel
from diffusers import UNet2DConditionModel
from diffusers import AutoencoderKL
from diffusers import StableDiffusionXLPipeline
from diffusers import EulerDiscreteScheduler
import numpy as np

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used,
            `timesteps` must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
                Custom timesteps used to support arbitrary spacing between timesteps. If `None`, then the default
                timestep spacing strategy of the scheduler is used. If `timesteps` is passed, `num_inference_steps`
                must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def timesteps_to_sigma(timesteps, alphas_cumprod):
    """Convert timesteps to sigmas."""
    sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
    sqrt_alpha_prod = sqrt_alpha_prod.flatten()
    sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
    return sqrt_one_minus_alpha_prod




def keep_subset_from_dataset(dataset, max_size):
    if max_size is None:
        return dataset
    else:
        total_size = dataset.num_rows
        num_shards = total_size // max_size
        if total_size % max_size != 0:
            num_shards += 1
        return dataset.shard(num_shards, index=0)                

def encode_prompt(captions, text_encoders, tokenizers):
    prompt_embeds_list = []

    with torch.no_grad():
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_inputs = tokenizer(
                captions,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            prompt_embeds = text_encoder(
                text_input_ids.to(text_encoder.device),
                output_hidden_states=True,
            )

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return {"prompt_embeds": prompt_embeds.cpu(), "pooled_prompt_embeds": pooled_prompt_embeds.cpu()}


def compute_time_ids(original_size, crops_coords_top_left, resolution=1024, device="cuda", weight_dtype=torch.float16):
    # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
    target_size = (resolution, resolution)
    add_time_ids = list(original_size + crops_coords_top_left + target_size)
    add_time_ids = torch.tensor([add_time_ids])
    add_time_ids = add_time_ids.to(device, dtype=weight_dtype)
    return add_time_ids



def _load_image(image_obj, device='cuda', resolution=None):
    if type(image_obj) == str:
        pil_image = PIL.Image.open(image_obj)
    elif type(image_obj) == PIL.Image.Image:
        pil_image = image_obj
    else:
        raise ValueError(f"Unrecognized image type: {type(image_obj)}")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((resolution, resolution)) if resolution is not None else transforms.Resize(pil_image.size)
    ])
    tensor_image = transform(pil_image)
    return torch.unsqueeze(tensor_image, 0).to(device)




def hot_sample(pipe, noisy_latent, steps, captions=None, normal_sampling_steps=50, stop_index=None, return_type="tensor", pipe_output_type="pil"):
    assert return_type in ["tensor", "pil"], f"Unrecognized return type: {return_type}"
    if captions is None:
        captions = noisy_latent.shape[0] * [""]
    # set inference timesteps
    pipe.scheduler.set_timesteps(normal_sampling_steps)
    inference_timesteps = pipe.scheduler.timesteps

    # find nearest timestep
    sampling_indices = (inference_timesteps.unsqueeze(1).cuda() - steps.unsqueeze(0).cuda()).abs().argmin(axis=0)


    sampled_images = []
    for index, latent, caption in zip(sampling_indices, noisy_latent, captions):
        initial_latent = torch.clone(latent.unsqueeze(0))
        def revert_callback(self, step, timestep, kwargs):
            if timestep > inference_timesteps[index]:
                # revert back change
                return {"latents": initial_latent}
            else:
                return {}

        uncond_img = pipe(prompt=caption, output_type=pipe_output_type, callback_on_step_end=revert_callback).images[0]
        if return_type == "tensor":
            sampled_images.append(_load_image(uncond_img))
        else:
            sampled_images.append(uncond_img)
    if return_type == "tensor":
        sampled_images = torch.cat(sampled_images) * 2 - 1
    return sampled_images


def broadcast_batch_tensor(batch_tensor):
    """ Takes a tensor of potential shape (batch_size) and returns a tensor of shape (batch_size, 1, 1, 1).
    """
    return batch_tensor.view(-1, 1, 1, 1)

def from_noise_pred_to_x0_pred_vp(noisy_input, noise_pred, sigma):
    sigma = broadcast_batch_tensor(sigma)
    return (noisy_input - sigma * noise_pred) / torch.sqrt(1 - sigma ** 2)


def run_unet(pipe, noisy_latent, steps, resolution=None, captions=None, return_noise=False):
    if isinstance(pipe.scheduler, EulerDiscreteScheduler):
        # see pipe.scheduler.scale_model_input
        divisor = np.array(((1 - pipe.scheduler.alphas_cumprod) / pipe.scheduler.alphas_cumprod) ** 0.5)[steps.cpu()]
        divisor = broadcast_batch_tensor(torch.tensor(divisor, device=noisy_latent.device))
        noisy_latent /= ((divisor **2 + 1) ** 0.5)

    if resolution is None:
        resolution = noisy_latent.shape[-2]
    if captions is None:
        captions = noisy_latent.shape[0] * [""]
    text_encoders = [pipe.text_encoder, pipe.text_encoder_2]

    tokenizers = [pipe.tokenizer, pipe.tokenizer_2]
    encoded_prompt = encode_prompt(captions, text_encoders, tokenizers)
    prompt_embeds = encoded_prompt["prompt_embeds"].cuda()
    pooled_prompt_embeds = encoded_prompt["pooled_prompt_embeds"]

    add_time_ids = torch.cat([compute_time_ids((resolution, resolution), (0, 0)) for _ in range(noisy_latent.shape[0])])
    unet_added_conditions = {"time_ids": add_time_ids.cuda(), "text_embeds": pooled_prompt_embeds.cuda()}
    noise_pred = pipe.unet(noisy_latent, steps, prompt_embeds, added_cond_kwargs=unet_added_conditions).sample
    if return_noise:
        return noise_pred
    
    sigma = timesteps_to_sigma(steps, pipe.scheduler.alphas_cumprod.to(steps.device))
    clean = from_noise_pred_to_x0_pred_vp(noisy_latent, noise_pred, sigma.to(noise_pred.device))
    return clean

def sample_with_early_stop(pipe, denoising_end, prompts, num_inference_steps=50, return_latent=False, **pipe_kwargs):
    noisy_latents = pipe(prompts, **pipe_kwargs, denoising_end=denoising_end, num_inference_steps=num_inference_steps, output_type="latent").images

    # find final timestep
    discrete_timestep_cutoff = int(round(pipe.scheduler.config.num_train_timesteps - (denoising_end * pipe.scheduler.config.num_train_timesteps)))
    timesteps = retrieve_timesteps(pipe.scheduler, num_inference_steps)[0]
    num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
    final_timestep = timesteps[:num_inference_steps + 1][-1].long()

    clean = run_unet(pipe, noisy_latents, torch.tensor([final_timestep], device=pipe.device), captions=prompts, return_noise=False)
    if return_latent:
        return clean
    else:
        images = pipe.vae.decode(clean.to(pipe.vae.dtype) / pipe.vae.scaling_factor).sample
        return images


def load_model(input_dir, vae_path, sdxl_path, trained_with_lora=False, use_ema=False, device="cuda", weight_dtype=torch.float16):
    vae = AutoencoderKL.from_pretrained(vae_path, subfolder=None, torch_dtype=weight_dtype)
    if trained_with_lora or input_dir is None:
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            sdxl_path,
            vae=vae,
            torch_dtype=weight_dtype,
        )
    else:
        if use_ema:
            unet = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel, torch_dtype=weight_dtype)
        else:
            unet = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet", torch_dtype=weight_dtype)
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            sdxl_path,
            vae=vae,
            unet=unet,
            torch_dtype=weight_dtype,
        )
    if trained_with_lora and input_dir is not None:
        pipeline.load_lora_weights(input_dir)
    pipeline.set_progress_bar_config(leave=False)
    pipeline.set_progress_bar_config(disable=True)
    return pipeline.to(device)