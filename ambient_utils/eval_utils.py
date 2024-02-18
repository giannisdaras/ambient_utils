import warnings
import torch
import numpy as np
import scipy
from ambient_utils import dist
from tqdm import tqdm
from ambient_utils.dataset_utils import ImageFolderDataset
import sys
import os
import pickle
from ambient_utils.url_utils import open_url


def calculate_inception_stats(
    image_path, num_expected=None, seed=0, max_batch_size=64,
    num_workers=3, prefetch_factor=2, device=torch.device('cuda'), 
    distributed=False):

    # Load Inception-v3 model.
    # This is a direct PyTorch translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
    detector_kwargs = dict(return_features=True)
    inception_kwargs = dict(no_output_bias=True) # Match the original implementation by not applying bias in the softmax layer.
    feature_dim = 2048
    
    # hack to add torch utils to path
    dir_path = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(dir_path)
    dist.print0("Loading detector")
    with open_url(detector_url, verbose=(dist.get_rank() == 0)) as f:
        detector_net = pickle.load(f).to(device)
    dist.print0("Detector loaded.")

    # List images.
    dataset_obj = ImageFolderDataset(path=image_path, max_size=num_expected, random_seed=seed, normalize=False)

    # Other ranks follow.
    if distributed and dist.get_rank() == 0:
        torch.distributed.barrier()

    if distributed:
        # Divide images into batches.
        num_batches = ((len(dataset_obj) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
        all_batches = torch.arange(len(dataset_obj)).tensor_split(num_batches)
        rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]
        
        data_loader = torch.utils.data.DataLoader(dataset_obj, batch_sampler=rank_batches, num_workers=0)
    else:
        data_loader = torch.utils.data.DataLoader(dataset_obj, batch_size=max_batch_size, num_workers=num_workers, 
                                                  prefetch_factor=prefetch_factor, pin_memory=True)
        rank_batches = torch.arange(len(data_loader))
    iter_loader = iter(data_loader)


    # Accumulate statistics.
    dist.print0(f'Calculating statistics for {len(dataset_obj)} images...')
    mu = torch.zeros([feature_dim], dtype=torch.float64, device=device)
    sigma = torch.zeros([feature_dim, feature_dim], dtype=torch.float64, device=device)

    all_features = []

    for _ in tqdm(range(len(rank_batches))):

        images = next(iter_loader)['image']

        if images.shape[0] == 0:
            break
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])

        # fid 
        features = detector_net(images.to(device), **detector_kwargs).to(torch.float64)
        mu += features.sum(0)
        sigma += features.T @ features

        # inception
        inception_features = torch.clamp(detector_net(images.to(device), **inception_kwargs), min=1e-6, max=1.0)
        all_features.append(inception_features.to(torch.float64))

    # print(f"Rank {dist.get_rank()} finished processing.")
    all_features = torch.cat(all_features, dim=0).reshape(-1, inception_features.shape[-1]).to(torch.float64)
    dist.print0("Features computed locally.")
    if distributed:
        dist.print0("Wait for all others to finish before gathering...")
        torch.distributed.barrier()
        dist.print0("Gathering process started...")
        all_features_list = [torch.ones_like(all_features) for _ in range(dist.get_world_size())]
        torch.distributed.all_gather(all_features_list, all_features)
    else:
        all_features_list = [all_features]
    all_features_gathered = torch.cat(all_features_list, dim=0)
    
    gen_probs = all_features_gathered.reshape(-1, all_features.shape[-1]).cpu().numpy()
    dist.print0(f"{gen_probs.shape}, {gen_probs.min()}, {gen_probs.max()}")
    dist.print0("Computing KL...")
    kl = gen_probs * (np.log(gen_probs) - np.log(np.mean(gen_probs, axis=0, keepdims=True)))
    kl = np.mean(np.sum(kl, axis=1))
    dist.print0("KL computed...")
    inception_score = np.mean(np.exp(kl))
    dist.print0(f"Inception score: {inception_score}")

    if distributed:
        torch.distributed.all_reduce(mu)
        torch.distributed.all_reduce(sigma)
    mu /= len(dataset_obj)
    sigma -= mu.ger(mu) * len(dataset_obj)
    sigma /= len(dataset_obj) - 1
    return mu.cpu().numpy(), sigma.cpu().numpy(), inception_score


def calculate_fid_from_inception_stats(mu, sigma, ref_path):
    with open_url(ref_path) as f:
        ref = dict(np.load(f))
    mu_ref, sigma_ref = ref['mu'], ref['sigma']
    m = np.square(mu - mu_ref).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma, sigma_ref), disp=False)
    fid = m + np.trace(sigma + sigma_ref - s * 2)
    return float(np.real(fid))