import torch
from typing import Callable

def get_classifier_trajectory(
    model: torch.nn.Module,
    input: torch.Tensor,
    scheduler: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    diffusion_times: torch.Tensor,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Get the trajectory of the classifier for a given input and diffusion times.
    
    Args:
        model: The classifier model
        input: Input tensor to classify
        scheduler: Function that takes two tensor arguments and returns a tensor
        diffusion_times: Tensor of diffusion timesteps
        device: Device to run model on
        
    Returns:
        torch.Tensor: Model output predictions
    """
    model.to(device)
    model.eval()
    predictions = []
    with torch.no_grad():
        for t in diffusion_times:
            input = scheduler(input, t)
            output = model(input)
            predictions.append(output.cpu())
    return torch.stack(predictions)


if __name__ == '__main__':
    pass