import torch
from datasets import Dataset

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

