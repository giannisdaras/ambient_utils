import torch
from datasets import Dataset
from torchvision import transforms
import PIL

def timesteps_to_sigma(timesteps, alphas_cumprod):
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


def compute_time_ids(original_size, crops_coords_top_left, resolution=512, device="cuda", weight_dtype=torch.float16):
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
        uncond_img = pipe(prompt=caption, latents=latent.unsqueeze(0) / pipe.scheduler.init_noise_sigma, 
            warm_index=int(index), stop_index=stop_index, output_type=pipe_output_type).images[0]
        if return_type == "tensor":
            sampled_images.append(_load_image(uncond_img))
        else:
            sampled_images.append(uncond_img)
    if return_type == "tensor":
        sampled_images = torch.cat(sampled_images) * 2 - 1
    return sampled_images



def run_unet(pipe, noisy_latent, steps, resolution=None, captions=None, return_noise=False):
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
    alphas_cumprod = pipe.scheduler.alphas_cumprod[steps.cpu()].cuda()[:, None, None, None]
    denoised_latent = (1 / torch.sqrt(alphas_cumprod) ) * (noisy_latent + (1 - alphas_cumprod) * noise_pred)
    return denoised_latent