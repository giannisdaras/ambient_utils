import torch
from ambient_utils.utils import broadcast_batch_tensor, ambient_sqrt, load_image, image_to_numpy, image_from_numpy, ensure_tensor, ensure_dimensions
from scipy.ndimage import gaussian_filter
import PIL
import tempfile
import numpy as np
import imagecorruptions

def add_extra_noise_from_vp_to_vp(noisy_input, current_sigma, desired_sigma):
    """
        Adds extra noise to the input to move from current_sigma to desired_sigma.
        Args:
            noisy_input: input to add noise to
            current_sigma: current noise level
            desired_sigma: desired noise level
        Returns:
            extra_noisy: noisy input with extra noise
            noise_realization: the noise realization that was added
            done: True if the desired noise level was reached, False otherwise
    """
    scaling_coeff = ambient_sqrt((1 - desired_sigma**2) / (1 - current_sigma ** 2))
    noise_coeff = ambient_sqrt(desired_sigma ** 2 - (scaling_coeff ** 2) * current_sigma ** 2)
    noise_realization = torch.randn_like(noisy_input)
    scaling_coeff, noise_coeff, current_sigma, desired_sigma = [broadcast_batch_tensor(x) for x in [scaling_coeff, noise_coeff, current_sigma, desired_sigma]]
    extra_noisy = scaling_coeff * noisy_input + noise_coeff * noise_realization
    # when we are trying to move to a lower noise level, just do nothing
    extra_noisy = torch.where(current_sigma > desired_sigma, noisy_input, extra_noisy)
    return extra_noisy, noise_realization, (current_sigma <= desired_sigma)[:, 0, 0, 0]


def get_box_mask(image_shape, survival_probability, device='cuda'):
    """Creates a mask with a box of size survival_probability * image_shape[1] somewhere randomly in the image.
        The mask can overflow outside of the image.
        Args:
            image_shape: (batch_size, num_channels, height, width)
            survival_probability: probability of a pixel being unmasked
            device: device to use for the mask
        Returns:
            mask: (batch_size, num_channels, height, width)
    """
    batch_size = image_shape[0]
    num_channels = image_shape[1]
    height = image_shape[2]
    width = image_shape[3]

    # create a mask with the same size as the image
    mask = torch.zeros((batch_size, num_channels, height, width), device=device)

    # decide where to place the box randomly -- set the box at a different location for each image in the batch
    box_start_row = torch.randint(0, height, (batch_size, 1, 1), device=device)
    box_start_col = torch.randint(0, width, (batch_size, 1, 1), device=device)
    box_height = torch.ceil(torch.tensor((1 - survival_probability) * height)).int()
    box_width = torch.ceil(torch.tensor((1 - survival_probability) * width)).int()
    
    box_start_row_expanded = box_start_row.view(batch_size, 1, 1, 1)
    box_start_col_expanded = box_start_col.view(batch_size, 1, 1, 1)

    rows = torch.arange(height, device=device).view(1, 1, -1, 1).expand_as(mask)
    cols = torch.arange(width, device=device).view(1, 1, 1, -1).expand_as(mask)

    inside_box_rows = (rows >= box_start_row_expanded) & (rows < (box_start_row_expanded + box_height))
    inside_box_cols = (cols >= box_start_col_expanded) & (cols < (box_start_col_expanded + box_width))

    inside_box = inside_box_rows & inside_box_cols
    mask[inside_box] = 1.0
    
    return 1 - mask



def get_box_mask_that_fits(image_shape, survival_probability, device='cuda'):
    """Creates a mask with a box of size survival_probability * image_shape[1] somewhere randomly in the image.
        Args:
            image_shape: (batch_size, num_channels, height, width)
            survival_probability: probability of a pixel being unmasked
            device: device to use for the mask
        Returns:
            mask: (batch_size, num_channels, height, width)
    """
    assert survival_probability >= 0.5, "survival_probability must be >= 0.5 for the mask to fit."
    batch_size = image_shape[0]
    num_channels = image_shape[1]
    height = image_shape[2]
    width = image_shape[3]

    # create a mask with the same size as the image
    mask = torch.zeros((batch_size, num_channels, height, width), device=device)

    mask_width = torch.ceil(torch.tensor((1 - survival_probability) * width)).int()
    mask_height = torch.ceil(torch.tensor((1 - survival_probability) * height)).int()

    # decide where to place the box randomly -- set the box at a different location for each image in the batch
    box_start_row = torch.randint(mask_height, height - mask_height, (batch_size, 1, 1), device=device)
    box_start_col = torch.randint(mask_width, width - mask_width, (batch_size, 1, 1), device=device)
    box_height = torch.ceil(torch.tensor((1 - survival_probability) * height)).int()
    box_width = torch.ceil(torch.tensor((1 - survival_probability) * width)).int()
    
    box_start_row_expanded = box_start_row.view(batch_size, 1, 1, 1)
    box_start_col_expanded = box_start_col.view(batch_size, 1, 1, 1)

    rows = torch.arange(height, device=device).view(1, 1, -1, 1).expand_as(mask)
    cols = torch.arange(width, device=device).view(1, 1, 1, -1).expand_as(mask)

    inside_box_rows = (rows >= box_start_row_expanded) & (rows < (box_start_row_expanded + box_height))
    inside_box_cols = (cols >= box_start_col_expanded) & (cols < (box_start_col_expanded + box_width))

    inside_box = inside_box_rows & inside_box_cols
    mask[inside_box] = 1.0
    
    return 1 - mask


@ensure_tensor
@ensure_dimensions
def apply_blur(image, sigma):
    device = image.device
    return torch.tensor(gaussian_filter(image[0].cpu(), sigma=(0, sigma, sigma))).to(device).unsqueeze(0)

@ensure_tensor
@ensure_dimensions
def apply_mask(image, masking_probability):
    device = image.device
    mask = (torch.rand(image.shape[2:]) > masking_probability).unsqueeze(0).repeat(image.shape[1], 1, 1)
    return image * mask.unsqueeze(0).to(device)

@ensure_tensor
@ensure_dimensions
def apply_jpeg_compression(image, quality):
    device = image.device
    image = image.cpu()
    image = image_to_numpy(image[0])
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
        PIL.Image.fromarray(image).save(temp_file.name, quality=quality)
        image = load_image(temp_file.name, device=device) * 2 - 1
    return image
    

@ensure_tensor
@ensure_dimensions
def apply_motion_blur(image, kernel_size, angle):
    device = image.device
    image = image.cpu()
    image = PIL.Image.fromarray(image_to_numpy(image[0]))
    image = image.filter(PIL.ImageFilter.GaussianBlur(radius=kernel_size))
    return image_from_numpy(np.array(image)).to(device)


@ensure_tensor
@ensure_dimensions
def apply_pixelate(image, pixel_size):
    device = image.device
    image = image.cpu()
    original_size = image.shape[-2:]
    image = PIL.Image.fromarray(image_to_numpy(image[0]))
    image = image.resize((pixel_size, pixel_size), PIL.Image.Resampling.NEAREST)
    image = image.resize((original_size[0], original_size[1]), PIL.Image.Resampling.NEAREST)
    return image_from_numpy(np.array(image)).to(device)


@ensure_tensor
@ensure_dimensions
def apply_saturation(image, saturation_level):
    device = image.device
    image = image.cpu()
    image = image_to_numpy(image[0])
    image = np.clip(image * saturation_level, 0, 255)
    return image_from_numpy(image).to(device)

@ensure_tensor
@ensure_dimensions
def apply_color_shift(image, shift):
    device = image.device
    image = image.cpu()
    image = image_to_numpy(image[0])
    image = np.clip(image + shift, 0, 255)
    return image_from_numpy(image).to(device)

@ensure_tensor
@ensure_dimensions
def apply_imagecorruptions(image, corruption_name, severity):
    # names: ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
    device = image.device
    image = image.cpu()
    image = image_to_numpy(image[0])
    image = np.clip(imagecorruptions.corrupt(image, corruption_name=corruption_name, severity=severity), 0, 255)
    return image_from_numpy(image).to(device)


