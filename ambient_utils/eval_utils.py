import warnings
import torch
import numpy as np
import scipy
import dist

def calculate_inception_stats(
    image_path, num_expected=None, seed=0, max_batch_size=64,
    num_workers=3, prefetch_factor=2, device=torch.device('cuda')):

    # Load Inception-v3 model.
    # This is a direct PyTorch translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    dist.print0('Loading Inception-v3 model...')
    detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
    detector_kwargs = dict(return_features=True)
    inception_kwargs = dict(no_output_bias=True) # Match the original implementation by not applying bias in the softmax layer.
    feature_dim = 2048
    with dnnlib.util.open_url(detector_url, verbose=(dist.get_rank() == 0)) as f:
        detector_net = pickle.load(f).to(device)

    # List images.
    dist.print0(f'Loading images from "{image_path}"...')
    dataset_obj = dataset.ImageFolderDataset(path=image_path, max_size=num_expected, random_seed=seed, normalize=False)

    if num_expected is not None and len(dataset_obj) < num_expected:
        raise click.ClickException(f'Found {len(dataset_obj)} images, but expected at least {num_expected}')
    if len(dataset_obj) < 2:
        raise click.ClickException(f'Found {len(dataset_obj)} images, but need at least 2 to compute statistics')

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # Divide images into batches.
    num_batches = ((len(dataset_obj) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.arange(len(dataset_obj)).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]
    
    data_loader = torch.utils.data.DataLoader(dataset_obj, batch_sampler=rank_batches, num_workers=0)
    iter_loader = iter(data_loader)


    # Accumulate statistics.
    dist.print0(f'Calculating statistics for {len(dataset_obj)} images...')
    mu = torch.zeros([feature_dim], dtype=torch.float64, device=device)
    sigma = torch.zeros([feature_dim, feature_dim], dtype=torch.float64, device=device)

    all_features = []

    for _ in tqdm.tqdm(range(len(rank_batches))):
        images, _labels, _, _ = next(iter_loader)

        torch.distributed.barrier()
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


    all_features = torch.cat(all_features, dim=0).reshape(-1, inception_features.shape[-1]).to(torch.float64)
    dist.print0("Features computed locally.")
    dist.print0("Wait for all others to finish before gathering...")
    torch.distributed.barrier()
    dist.print0("Gathering process started...")

    all_features_list = [torch.ones_like(all_features) for _ in range(dist.get_world_size())]
    torch.distributed.all_gather(all_features_list, all_features)
    all_features_gathered = torch.cat(all_features_list, dim=0)
    
    gen_probs = all_features_gathered.reshape(-1, all_features.shape[-1]).cpu().numpy()
    dist.print0(f"{gen_probs.shape}, {gen_probs.min()}, {gen_probs.max()}")
    dist.print0("Computing KL...")
    kl = gen_probs * (np.log(gen_probs) - np.log(np.mean(gen_probs, axis=0, keepdims=True)))
    kl = np.mean(np.sum(kl, axis=1))
    dist.print0("KL computed...")
    inception_score = np.mean(np.exp(kl))
    dist.print0(f"Inception score: {inception_score}")

    # Calculate grand totals.
    torch.distributed.all_reduce(mu)
    torch.distributed.all_reduce(sigma)
    mu /= len(dataset_obj)
    sigma -= mu.ger(mu) * len(dataset_obj)
    sigma /= len(dataset_obj) - 1
    return mu.cpu().numpy(), sigma.cpu().numpy(), inception_score


def calculate_fid_from_inception_stats(mu, sigma, mu_ref, sigma_ref):
    m = np.square(mu - mu_ref).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma, sigma_ref), disp=False)
    fid = m + np.trace(sigma + sigma_ref - s * 2)
    return float(np.real(fid))

def calc(image_path, ref_path, num_expected, seed, batch, num_rows=8, num_cols=8, image_size=32, with_wandb=True):
    """Calculate Inception/FID for a given set of images."""
    assert num_rows * num_cols <= num_expected, "You need to save more images."
    dist.print0("Starting FID calculation...")
    dist.print0(f'Loading dataset reference statistics from "{ref_path}"...')
    ref = None
    if dist.get_rank() == 0:
        with dnnlib.util.open_url(ref_path) as f:
            ref = dict(np.load(f))
    
    try:
        checkpoint_index = int(image_path.split('/')[-1])
    except:
        checkpoint_index = 0
        # raise warning that we could not find the checkpoint
        warnings.warn("Could not find the checkpoint")
    
    mu, sigma, inception = calculate_inception_stats(image_path=image_path, num_expected=num_expected, seed=seed, max_batch_size=batch)
    dist.print0(f'Calculating FID for {image_path}...')
    if dist.get_rank() == 0:
        fid = calculate_fid_from_inception_stats(mu, sigma, ref['mu'], ref['sigma'])
        dist.print0(f"FID: {fid:g}")

    torch.distributed.barrier()
    if dist.get_rank() == 0 and with_wandb:
        wandb.log({"FID": fid, "Inception": inception, "image_grid": wandb.Image(grid_image)}, step=checkpoint_index, commit=True)
    dist.print0("Computed FID and logged it.")