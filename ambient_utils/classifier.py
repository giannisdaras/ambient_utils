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
    model_output_type: str = 'logits',
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
        model_output_type: The type of output that the model returns.
    Returns:
        torch.Tensor: Model output predictions
    """
    model.to(device)
    model.eval()
    predictions = []

    def process_t(t):
        updated_input = scheduler(input, t)
        output = model(updated_input, t.unsqueeze(0))
        if model_output_type == 'logits':
            probs = torch.softmax(output, dim=1)[:, 0]
        elif model_output_type == 'probs':
            probs = output
        return probs.cpu()

    vmapped_fn = torch.func.vmap(process_t, randomness="same", chunk_size=batch_size)
    with torch.no_grad():
        predictions = vmapped_fn(diffusion_times)
    return predictions


def analyze_classifier_trajectory(
    # trajectory of probabilities. Here 1 means that the image is fake.
    trajectory: torch.Tensor,
    diffusion_times: torch.Tensor,
    epsilon: float = 0.01,
) -> None:
    # find the first time at which 0.5 - trajectory + epsilon becomes positive.
    # this essentially finds the first confusion time.
    # if no such time exists, return the last time.
    # the higher the epsilon, the easier the misclassification.
    confusion_indices = (0.5 - trajectory + epsilon) > 0
    if confusion_indices.any():
        first_confusion = confusion_indices.nonzero(as_tuple=True)[0][0]
    else:
        first_confusion = diffusion_times.size(0) - 1

    return_dict = {
        "first_confusion": diffusion_times[first_confusion],
    }
    return return_dict


if __name__ == '__main__':
    pass