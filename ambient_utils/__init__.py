import torch
from torchvision import transforms
import PIL
import numpy as np

def load_image(image_obj, device='cuda', resolution=None):
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

def save_image(images, image_path):
    image_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    if image_np.shape[2] == 1:
        PIL.Image.fromarray(image_np[:, :, 0], 'L').save(image_path)
    else:
        PIL.Image.fromarray(image_np, 'RGB').save(image_path)


def save_images(images, image_path, num_rows=None, num_cols=None):
    if num_rows is None:
        num_rows = int(np.sqrt(images.shape[0]))
    if num_cols is None:
        num_cols = int(np.ceil(images.shape[0] / num_rows))
    
    # TODO(giannisdaras): only works with square grids
    image_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    image_size = images.shape[-2]
    grid_image = PIL.Image.new('RGB', (num_cols * image_size, num_rows * image_size))
    for i in range(num_rows):
        for j in range(num_cols):
            index = i * num_cols + j
            img = PIL.Image.fromarray(image_np[index])
            grid_image.paste(img, (i * image_size, j * image_size))
    grid_image.save(image_path)


def tile_image(batch_image, n, m=None):
    if m is None:
        m = n
    assert n * m == batch_image.size(0)
    channels, height, width = batch_image.size(1), batch_image.size(2), batch_image.size(3)
    batch_image = batch_image.view(n, m, channels, height, width)
    batch_image = batch_image.permute(2, 0, 3, 1, 4)  # n, height, n, width, c
    batch_image = batch_image.contiguous().view(channels, n * height, m * width)
    return batch_image


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