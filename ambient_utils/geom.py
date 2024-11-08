import torch
import numpy as np

def keep_center(image, keep_ratio=0.5, keep_original_size=False):
    """
        Keeps the center of the image, discarding the rest.
        Args:
            image: (batch_size, num_channels, height, width)
            keep_ratio: ratio of the image to keep
            keep_original_size: if True, the image will be upscaled to the original size
        Returns:
            image: 
                (batch_size, num_channels, height * keep_ratio, width * keep_ratio) if keep_original_size is False
                (batch_size, num_channels, height, width) if keep_original_size is True
    """
    height = image.shape[2]
    width = image.shape[3]
    new_height = int(height * keep_ratio)
    new_width = int(width * keep_ratio)
    start_row = (height - new_height) // 2
    start_col = (width - new_width) // 2
    if keep_original_size:
        return torch.nn.functional.interpolate(image[:, :, start_row:start_row + new_height, start_col:start_col + new_width], size=(height, width))
    else:
        return image[:, :, start_row:start_row + new_height, start_col:start_col + new_width]

def shift_image_with_real_data_left_or_right(ambient_image, real_image, keep_ratio, shift_ratio, shift_left, return_shifted=False):
    """
        Shifts the real image to the left or right and returns the result.
        Args:
            ambient_image: (batch_size, num_channels, ambient_height, ambient_width)
            real_image: (batch_size, num_channels, real_height, real_width)
            keep_ratio: ratio of center crop of ambient image
            shift_ratio: (1,) or (batch_size,). ratio of the image to shift
            shift_left: (1,) or (batch_size,). if True, the image will be shifted left, otherwise right
    """
    def _shift_single_image(ambient_image, real_image, keep_ratio, shift_ratio=0.5, shift_left=True):
        # round shift_ratio to three decimal places
        shift_ratio = round(shift_ratio, 3)
        if shift_ratio == 0:
            if return_shifted:
                return ambient_image, real_image
            return ambient_image
        if shift_ratio < 0:
            shift_ratio = -shift_ratio
            shift_left = not shift_left
        _, _, ambient_height, ambient_width = ambient_image.shape
        _, _, real_height, real_width = real_image.shape
        num_pixels_to_shift = int(shift_ratio * real_width)

        assert ambient_width <= real_width - 2 * num_pixels_to_shift, "The real image should have enough data to shift the ambient image."
        assert int(real_height * keep_ratio) == ambient_height, "Keep ratio is not set correctly."
        assert int(real_width * keep_ratio) == ambient_width, "Keep ratio is not set correctly."

        if shift_left:
            # shift real image to the left
            shifted_real_image = real_image[:, :, :, :-num_pixels_to_shift]
            # pad to its original size
            shifted_real_image = torch.nn.functional.pad(shifted_real_image, (num_pixels_to_shift, 0, 0, 0))
        else:
            # shift real image to the right
            shifted_real_image = real_image[:, :, :, num_pixels_to_shift:]
            # pad to its original size
            shifted_real_image = torch.nn.functional.pad(shifted_real_image, (0, num_pixels_to_shift, 0, 0))   
        shifted = keep_center(shifted_real_image, keep_ratio)
        cloned_ambient_image = torch.clone(ambient_image)
        if shift_left:
            cloned_ambient_image[:, :, :, -shifted.shape[3]:] = shifted
        else:
            cloned_ambient_image[:, :, :, :shifted.shape[3]] = shifted
        return cloned_ambient_image, shifted_real_image
    
    if type(shift_ratio) == float:
        shift_ratio = torch.tensor(ambient_image.shape[0] * [shift_ratio], device=ambient_image.device)
    if type(shift_left) == bool:
        shift_left = torch.tensor(ambient_image.shape[0] * [shift_left], device=ambient_image.device)
    cloned_ambient_image = torch.clone(ambient_image)
    cloned_real_image = torch.clone(real_image)
    for i in range(ambient_image.shape[0]):
        cloned_ambient_image[i], cloned_real_image[i] = _shift_single_image(ambient_image[i].unsqueeze(0), real_image[i].unsqueeze(0), keep_ratio, float(shift_ratio[i]), shift_left[i])
    if return_shifted:
        return cloned_ambient_image, cloned_real_image
    return cloned_ambient_image


def shift_image_with_real_data_up_or_down(ambient_image, real_image, keep_ratio, shift_ratio, shift_up, return_shifted=False):
    """
        Shifts the real image up or down and returns the result.
        Args:
            ambient_image: (batch_size, num_channels, ambient_height, ambient_width)
            real_image: (batch_size, num_channels, real_height, real_width)
            keep_ratio: ratio of center crop of ambient image
            shift_ratio: (batch_size,) ratio of the image to shift
            shift_up: (batch_size,) if True, the image will be shifted up, otherwise down
    """
    def _shift_single_image(ambient_image, real_image, keep_ratio, shift_ratio=0.5, shift_up=True):
        # round shift_ratio to three decimal places
        shift_ratio = round(shift_ratio, 3)
        if shift_ratio == 0:
            if return_shifted:
                return ambient_image, real_image
            return ambient_image
        if shift_ratio < 0:
            shift_ratio = -shift_ratio
            shift_up = not shift_up
        _, _, ambient_height, ambient_width = ambient_image.shape
        _, _, real_height, real_width = real_image.shape
        num_pixels_to_shift = int(shift_ratio * real_height)

        assert ambient_height <= real_height - 2 * num_pixels_to_shift, "The real image should have enough data to shift the ambient image."
        assert int(real_height * keep_ratio) == ambient_height, "Keep ratio is not set correctly."
        assert int(real_width * keep_ratio) == ambient_width, "Keep ratio is not set correctly."

        if shift_up:
            # shift real image to the left
            shifted_real_image = real_image[:, :, :-num_pixels_to_shift, :]
            # pad to its original size
            shifted_real_image = torch.nn.functional.pad(shifted_real_image, (0, 0, num_pixels_to_shift, 0))
        else:
            # shift real image to the right
            shifted_real_image = real_image[:, :, num_pixels_to_shift:, :]
            # pad to its original size
            shifted_real_image = torch.nn.functional.pad(shifted_real_image, (0, 0, 0, num_pixels_to_shift))   
        shifted = keep_center(shifted_real_image, keep_ratio)
        cloned_ambient_image = torch.clone(ambient_image)
        if shift_up:
            cloned_ambient_image[:, :, -shifted.shape[2]:, :] = shifted
        else:
            cloned_ambient_image[:, :, :shifted.shape[2], :] = shifted
        return cloned_ambient_image, shifted_real_image

    if type(shift_ratio) == float:
        shift_ratio = torch.tensor(ambient_image.shape[0] * [shift_ratio], device=ambient_image.device)
    if type(shift_up) == bool:
        shift_up = torch.tensor(ambient_image.shape[0] * [shift_up], device=ambient_image.device)

    cloned_ambient_image = torch.clone(ambient_image)
    cloned_real_image = torch.clone(real_image)
    for i in range(ambient_image.shape[0]):
        cloned_ambient_image[i], cloned_real_image[i] = _shift_single_image(ambient_image[i].unsqueeze(0), real_image[i].unsqueeze(0), keep_ratio, float(shift_ratio[i]), shift_up[i])
    
    if return_shifted:
        return cloned_ambient_image, cloned_real_image
    return cloned_ambient_image


def animate_image_left_to_right(image, keep_ratio=0.7, shift_span=0.1, num_steps=10):
    """
        Animates the image from left to right.
        Args:
            image: (batch_size, num_channels, height, width)
            keep_ratio: ratio of the image to keep
            shift_span: ratio of the image to shift
            num_steps: number of steps to animate
        Returns:
            images: (num_steps, batch_size, num_channels, height, width)
    """
    batch_size = image.shape[0]
    num_channels = image.shape[1]
    height = image.shape[2]
    width = image.shape[3]
    images = []
    centered_image = keep_center(image, keep_ratio)
    # move left
    for i in range(num_steps):
        shift_ratio = shift_span * ((i + 1) / num_steps)
        images.append(shift_image_with_real_data_left_or_right(centered_image, image, keep_ratio, shift_ratio=shift_ratio, shift_left=True))
    
    # move back to the center
    images = images + images[::-1] + [centered_image]
    
    right_movement_images = []
    # move right
    for i in range(num_steps):
        shift_ratio = shift_span * ((i + 1) / num_steps)
        right_movement_images.append(shift_image_with_real_data_left_or_right(centered_image, image, keep_ratio, shift_ratio=shift_ratio, shift_left=False))
    
    images = images + right_movement_images + right_movement_images[::-1] + [centered_image]

    images = torch.stack(images)
    images = images.permute(1, 0, 2, 3, 4)
    return images


def animate_image_up_to_down(image, keep_ratio=0.7, shift_span=0.1, num_steps=10):
    """
        Animates the image from left to right.
        Args:
            image: (batch_size, num_channels, height, width)
            keep_ratio: ratio of the image to keep
            shift_span: ratio of the image to shift
            num_steps: number of steps to animate
        Returns:
            images: (num_steps, batch_size, num_channels, height, width)
    """
    images = []
    centered_image = keep_center(image, keep_ratio)
    # move up
    for i in range(num_steps):
        shift_ratio = shift_span * ((i + 1) / num_steps)
        images.append(shift_image_with_real_data_up_or_down(centered_image, image, keep_ratio, shift_ratio=shift_ratio, shift_up=True))
    
    # move back to the center
    images = images + images[::-1] + [centered_image]
    
    down_movement_images = []
    # move down
    for i in range(num_steps):
        shift_ratio = shift_span * ((i + 1) / num_steps)
        down_movement_images.append(shift_image_with_real_data_up_or_down(centered_image, image, keep_ratio, shift_ratio=shift_ratio, shift_up=False))

    images = images + down_movement_images + down_movement_images[::-1] + [centered_image]
    images = torch.stack(images)
    images = images.permute(1, 0, 2, 3, 4)
    return images


def animate_image_rotation_around_center(image, keep_ratio=0.7, rotation_radius=0.1, num_steps=10):
    """
        Animates the image from left to right.
        Args:
            image: (batch_size, num_channels, height, width)
            keep_ratio: ratio of the image to keep
            rotation_radius: ratio of the image to rotate
            num_steps: number of steps to animate
        Returns:
            images: (num_steps, batch_size, num_channels, height, width)
    """
    images = []
    centered_image = keep_center(image, keep_ratio)
    # move left
    for i in range(num_steps):
        rotation_angle = 2 * np.pi * ((i + 1) / num_steps)
        left_shift_ratio = float(rotation_radius * np.sin(rotation_angle))
        top_shift_ratio = float(rotation_radius * np.cos(rotation_angle))
        left_shifted, shifted_image = shift_image_with_real_data_left_or_right(centered_image, image, keep_ratio, shift_ratio=left_shift_ratio, shift_left=True, return_shifted=True)
        images.append(shift_image_with_real_data_up_or_down(left_shifted, shifted_image, keep_ratio, shift_ratio=top_shift_ratio, shift_up=False))
    images = torch.stack(images)
    images = images.permute(1, 0, 2, 3, 4)
    return images


def move_random_direction(image, keep_ratio=0.7, max_radius=0.1, return_shift_params=True):
    """
        Moves the image in a random direction.
        Args:
            image: (batch_size, num_channels, height, width)
            keep_ratio: ratio of the image to keep
            max_radius: ratio of the image to shift
            num_steps: number of steps to animate
    """
    centered_image = keep_center(image, keep_ratio)
    random_angles = np.random.uniform(0, 2 * np.pi, size=(image.shape[0],))
    random_radii = np.random.uniform(0, max_radius, size=(image.shape[0],))
    left_shift_ratio = torch.tensor(random_radii * np.sin(random_angles), device=image.device)
    top_shift_ratio = torch.tensor(random_radii * np.cos(random_angles), device=image.device)
    left_shifted, real_shifted = shift_image_with_real_data_left_or_right(centered_image, image, keep_ratio, shift_ratio=left_shift_ratio, shift_left=True, return_shifted=True)
    final_shifted = shift_image_with_real_data_up_or_down(left_shifted, real_shifted, keep_ratio, shift_ratio=top_shift_ratio, shift_up=False)
    if return_shift_params:
        return final_shifted, left_shift_ratio, top_shift_ratio
    return final_shifted

