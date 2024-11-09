import torch
from typing import Callable
from ambient_utils.utils import batch_vmap

def get_classifier_trajectory(
    model: torch.nn.Module,
    input: torch.Tensor,
    scheduler: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    diffusion_times: torch.Tensor,
    device: str = 'cuda',
    batch_size: int = 1,
) -> torch.Tensor:
    """
    Get the trajectory of the classifier for a given input and diffusion times.
    
    Args:
        model: The classifier model
        input: Input tensor to classify
        scheduler: Function that takes two tensor arguments and returns a tensor
        diffusion_times: Tensor of diffusion timesteps
        device: Device to run model on
        batch_size: Number of inputs to process in parallel at once.
        
    Returns:
        torch.Tensor: Model output predictions
    """
    model.to(device)
    model.eval()
    predictions = []
    with torch.no_grad():
        def process_t(t):
            updated_input = scheduler(input, t)
            output = model(updated_input, t.unsqueeze(0))
            return output.cpu()

        predictions = batch_vmap(process_t, diffusion_times, batch_size=batch_size)
    return predictions


if __name__ == '__main__':
    pass