import torch
from ambient_utils.utils import broadcast_batch_tensor, ambient_sqrt


def get_mean_loss(loss, mask=None):
    """ Computes the mean loss over the batch.
        Args:
            loss: (batch_size, num_channels, height, width)
            mask: (batch_size, num_channels, height, width) or (batch_size)
        Returns:
            mean_loss: scalar tensor.
    """
    if mask is None:
        return loss.mean()
        
    if len(mask.shape) == 1:
        mask = mask.repeat([1, loss.shape[1], loss.shape[2], loss.shape[3]])    
    
    if mask.sum() == 0:
        return loss.sum() * mask.sum()
    loss = loss.sum() / mask.sum()
    return loss


def from_noise_pred_to_x0_pred_vp(noisy_input, noise_pred, sigma):
    sigma = broadcast_batch_tensor(sigma)
    return (noisy_input - sigma * noise_pred) / torch.sqrt(1 - sigma ** 2)

def from_x0_pred_to_noise_pred_vp(noisy_input, x0_pred, sigma):
    sigma = broadcast_batch_tensor(sigma)
    return (noisy_input - x0_pred * torch.sqrt(1 - sigma ** 2)) / sigma


def from_x0_pred_to_xnature_pred_vp_to_vp(x0_pred, noisy_input, current_sigma, desired_sigma):
    current_sigma, desired_sigma = [broadcast_batch_tensor(x) for x in [current_sigma, desired_sigma]]
    scaling_coeff = ambient_sqrt((1 - desired_sigma**2) / (1 - current_sigma ** 2))
    noise_coeff = ambient_sqrt(desired_sigma ** 2 - (scaling_coeff ** 2) * current_sigma ** 2)
    return ((noise_coeff / desired_sigma) ** 2 * (ambient_sqrt(1 - desired_sigma ** 2) * x0_pred - noisy_input) + noisy_input) / scaling_coeff

def from_x0_pred_to_xnature_pred_ve_to_ve(x0_pred, noisy_input, current_sigma, desired_sigma):
    current_sigma, desired_sigma = [broadcast_batch_tensor(x) for x in [current_sigma, desired_sigma]]
    return (1 - (desired_sigma / current_sigma) ** 2) * x0_pred + ((desired_sigma / current_sigma) ** 2) * noisy_input