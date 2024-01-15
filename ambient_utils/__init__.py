import torch
from torchvision import transforms
import PIL
import numpy as np
import wandb
import imageio
from . import dataset_utils, geom_utils, diffusers_utils
import math

# vae = AutoencoderKL.from_pretrained(vae_path, subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None, revision=args.revision, variant=args.variant,)
# vae = vae.cuda().to(torch.float32)
# image_pred = vae.decode(xn_pred / vae.config.scaling_factor, return_dict=False)[0]
# ambient_utils.save_image(image_pred[0], "image_pred.png")

def broadcast_batch_tensor(batch_tensor):
    """ Takes a tensor of potential shape (batch_size) and returns a tensor of shape (batch_size, 1, 1, 1).
    """
    return batch_tensor.view(-1, 1, 1, 1)

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


def ambient_sqrt(x):
    """
        Computes the square root of x if x is positive, and 1 otherwise.
    """
    return torch.where(x < 0, torch.ones_like(x), x.sqrt())


def from_noise_pred_to_x0_pred_vp(noisy_input, noise_pred, sigma):
    sigma = broadcast_batch_tensor(sigma)
    return (noisy_input - sigma * noise_pred) / torch.sqrt(1 - sigma ** 2)


def from_x0_pred_to_xnature_pred_vp_to_vp(x0_pred, noisy_input, current_sigma, desired_sigma):
    current_sigma, desired_sigma = [broadcast_batch_tensor(x) for x in [current_sigma, desired_sigma]]
    scaling_coeff = ambient_sqrt((1 - desired_sigma**2) / (1 - current_sigma ** 2))
    noise_coeff = ambient_sqrt(desired_sigma ** 2 - (scaling_coeff ** 2) * current_sigma ** 2)
    return ((noise_coeff / desired_sigma) ** 2 * (ambient_sqrt(1 - desired_sigma ** 2) * x0_pred - noisy_input) + noisy_input) / scaling_coeff


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

def save_image(images, image_path, save_wandb=False, down_factor=None, wandb_down_factor=None, 
               caption=None, font_size=40, text_color=(255, 255, 255)):
    image_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    if image_np.shape[2] == 1:
        pil_image = PIL.Image.fromarray(image_np[:, :, 0], 'L')
    else:
        pil_image = PIL.Image.fromarray(image_np, 'RGB')
    if down_factor is not None:
        pil_image = pil_image.resize((pil_image.size[0] // down_factor, pil_image.size[1] // down_factor))
    
    if caption is not None:
        draw = PIL.ImageDraw.Draw(pil_image)
        # use LaTeX bold font
        font = PIL.ImageFont.truetype("cmr10.ttf", font_size)
        # make bold
        draw.text((0, 0), caption, text_color, font=font)

    pil_image.save(image_path)

    if save_wandb:
        if wandb_down_factor is not None:
            # resize for speed
            pil_image = pil_image.resize((pil_image.size[0] // wandb_down_factor, pil_image.size[1] // wandb_down_factor))
        wandb.log({"images/" + image_path.split("/")[-1]: wandb.Image(pil_image)})


def find_closest_factors(number):
    sqrt_number = int(math.sqrt(number))
    
    n = sqrt_number
    m = number // n
    
    while n * m != number:
        n += 1
        m = number // n

    return m, n

def save_images(images, image_path, num_rows=None, num_cols=None, save_wandb=False, down_factor=None, wandb_down_factor=None, 
                captions=None, font_size=40, text_color=(255, 255, 255), draw_horizontal_arrow=False, draw_vertical_arrow=False):
    if num_rows is None and num_cols is None:
        num_rows = int(np.sqrt(images.shape[0]))    
        num_cols = int(np.ceil(images.shape[0] / num_rows))
    elif num_rows is None and num_cols is not None:
        num_rows = int(np.ceil(images.shape[0] / num_cols))
    elif num_rows is not None and num_cols is None:
        num_cols = int(np.ceil(images.shape[0] / num_rows))
    
    if num_rows * num_cols != images.shape[0]:
        num_rows, num_cols = find_closest_factors(images.shape[0])
    
    image_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    image_size = images.shape[-2]
    grid_image = PIL.Image.new('RGB', (num_cols * image_size, num_rows * image_size))
    for i in range(num_rows):
        for j in range(num_cols):
            index = i * num_cols + j
            img = PIL.Image.fromarray(image_np[index])
            grid_image.paste(img, (j * image_size, i * image_size))
            if captions is not None:
                draw = PIL.ImageDraw.Draw(grid_image)
                # use LaTeX bold font
                font = PIL.ImageFont.truetype("cmr10.ttf", font_size)
                draw.text((j * image_size, i * image_size), captions[index], text_color, font=font)
    if down_factor is not None:
        grid_image = grid_image.resize((grid_image.size[0] // down_factor, grid_image.size[1] // down_factor))
    
    if draw_horizontal_arrow:
        draw = PIL.ImageDraw.Draw(grid_image)
        # draw it on the top of the image
        draw.line((0, 0, grid_image.size[0], 0), fill=(255, 0, 0), width=5)

    if draw_vertical_arrow:
        draw = PIL.ImageDraw.Draw(grid_image)
        # draw it on the left of the image
        draw.line((0, 0, 0, grid_image.size[1]), fill=(255, 0, 0), width=5)

    grid_image.save(image_path)

    if save_wandb:
        if wandb_down_factor is not None:
            # resize for speed
            grid_image = grid_image.resize((grid_image.size[0] // wandb_down_factor, grid_image.size[1] // wandb_down_factor))
        wandb.log({"images/" + image_path.split("/")[-1]: wandb.Image(grid_image)})




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




def create_video_from_frames(frames, video_path, fps=25):
    """
        Creates a video from frames.
        Args:
            frames: (batch_size, num_frames, num_channels, height, width)
            video_path: path to save the video
            fps: frames per second
    """
    def _create_video_from_frames(frames, video_path, fps=25):
        frames = (frames * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        imageio.mimwrite(video_path, frames, fps=fps)
    batch_size = frames.shape[0]
    tiled_frames = [tile_image(frame, n=batch_size, m=1) for frame in frames.permute(1, 0, 2, 3, 4)]
    tiled_frames = torch.stack(tiled_frames, dim=0)
    _create_video_from_frames(tiled_frames, video_path, fps=fps)


def get_rel_methods(obj, keyword):
    """Returns all methods/properties of obj that contain keyword in their name.
    """
    return [attr for attr in dir(obj) if keyword in attr]

    