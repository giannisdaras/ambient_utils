import torch
import numpy as np
import PIL
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import torchvision.transforms as transforms
import wandb
import math
import os
import s3fs
from typing import Any

def get_rel_methods(obj, keyword):
    """Returns all methods/properties of obj that contain keyword in their name.
    """
    return [attr for attr in dir(obj) if keyword in attr]

def broadcast_batch_tensor(batch_tensor):
    """ Takes a tensor of potential shape (batch_size) and returns a tensor of shape (batch_size, 1, 1, 1).
    """
    return batch_tensor.view(-1, 1, 1, 1)


def ambient_sqrt(x):
    """
        Computes the square root of x if x is positive, and 1 otherwise.
    """
    return torch.where(x < 0, torch.ones_like(x), x.sqrt())

def load_image(image_obj, device='cuda', resolution=None):
    if type(image_obj) == str:
        pil_image = PIL.Image.open(image_obj)
    elif type(image_obj) == PIL.Image.Image:
        pil_image = image_obj
    else:
        raise ValueError(f"Unrecognized image type: {type(image_obj)}")
    if resolution is not None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((resolution, resolution))
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
    tensor_image = transform(pil_image)
    return torch.unsqueeze(tensor_image, 0).to(device)

def save_image(images, image_path, save_wandb=False, down_factor=None, wandb_down_factor=None, 
               caption=None, font_size=40, text_color=(255, 255, 255), image_type="RGB"):
    image_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    if image_np.shape[2] == 1:
        pil_image = PIL.Image.fromarray(image_np[:, :, 0], 'L')
    else:
        pil_image = PIL.Image.fromarray(image_np, image_type)
    if down_factor is not None:
        pil_image = pil_image.resize((pil_image.size[0] // down_factor, pil_image.size[1] // down_factor))
    
    if caption is not None:
        draw = PIL.ImageDraw.Draw(pil_image)
        # use LaTeX bold font
        font = PIL.ImageFont.truetype("cmr10.ttf", font_size)
        # make bold
        draw.text((0, 0), caption, text_color, font=font)

    pil_image.save(image_path)

    if save_wandb and wandb.run is not None:
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


def image_to_numpy(image):
    return (image * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()

def image_from_numpy(image):
    return torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(torch.float32) / 127.5 - 1

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

    if save_wandb and wandb.run is not None:
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


_dnnlib_cache_dir = None





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




    

def color_image_border(images, color_mask, border_thickness=5):
    """
        Colors the border of the image for which color_mask is True with the given color.
        Args:
            images: (batch_size, num_channels, height, width)
            color_mask: (batch_size,)
        Returns:
            colored_images: (batch_size, num_channels, height, width)
    """
    assert images.shape[0] == color_mask.shape[0]
    height = images.shape[2]
    width = images.shape[3]
    colored_images = images.clone()
    
    expanded_color_mask = 1 - color_mask[:, None, None, None].to(images.device)
    colored_images[:, 0, :border_thickness, :] = expanded_color_mask[:, 0, :border_thickness, :] * colored_images[:, 0, :border_thickness, :] + (1 - expanded_color_mask[:, 0, :border_thickness, :]) * 1
    colored_images[:, 0, height - border_thickness - 1:, :] = expanded_color_mask[:, 0] * colored_images[:, 0, height - border_thickness - 1:, :] + (1 - expanded_color_mask[:, 0]) * 1
    colored_images[:, 0, :, :border_thickness] = expanded_color_mask[:, 0, :, :border_thickness] * colored_images[:, 0, :, :border_thickness] + (1 - expanded_color_mask[:, 0, :, :border_thickness]) * 1
    colored_images[:, 0, :, width - border_thickness - 1:] = expanded_color_mask[:, 0] * colored_images[:, 0, :, width - border_thickness - 1:] + (1 - expanded_color_mask[:, 0]) * 1
    return colored_images


def stylize_plots():
    sns.set(style="whitegrid")


class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]

def expand_vars(args):
    if not isinstance(args, dict):
        args = EasyDict(vars(args))
    else:
        args = EasyDict(args)
    for key, value in args.items():
        if isinstance(value, str) and "$" in value:
            args[key] = os.path.expandvars(value)
    return args  

def set_seed(seed=42):
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def is_file(filename):
    if not filename.startswith('s3://'):
        return os.path.isfile(filename)
    else:
        s3 = s3fs.S3FileSystem(anon=False)
        return s3.isfile(filename)


def pad_image(image, mode='reflect', height_patch=14, width_patch=14):
    # Get the input image shape
    batch, channels, height, width = image.shape
    
    # Calculate the padding required for height and width
    padding_height = (height_patch - (height % height_patch)) % height_patch
    padding_width = (width_patch - (width % width_patch)) % width_patch
    
    # Determine the padding on each side
    top_padding = padding_height // 2
    bottom_padding = padding_height - top_padding
    left_padding = padding_width // 2
    right_padding = padding_width - left_padding
    
    # Apply the padding to the image tensor
    padding = (left_padding, right_padding, top_padding, bottom_padding)
    padded_image = torch.nn.functional.pad(image, padding, mode=mode)

    return padded_image


def batch_vmap(fn, inputs, batch_size, randomness="different"):
    """
    Applies `fn` to `inputs` in parallel, in batches of size `batch_size` to avoid OOM issues.

    Args:
        fn (Callable): The function to vectorize using `torch.vmap`.
        inputs (Tensor): The inputs to process, where each entry along the first dimension is a separate input.
        batch_size (int): The number of inputs to process in parallel at once.

    Returns:
        Tensor: The concatenated results of applying `fn` to `inputs` in parallel in batches.
    """
    results = []
    for i in range(0, inputs.size(0), batch_size):
        batch_inputs = inputs[i:i + batch_size]
        # Apply `torch.vmap(fn)` over this batch
        batch_results = torch.vmap(fn, randomness=randomness)(batch_inputs)
        results.append(batch_results)
    
    return torch.cat(results, dim=0)