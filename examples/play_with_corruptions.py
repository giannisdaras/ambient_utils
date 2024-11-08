import ambient_utils as ambient

def main():
    image = ambient.load_image("example_image.jpg", device="cpu") * 2 - 1
    ambient.save_image(image[0], "example_image_loaded.jpg")


    # ambient.save_image(ambient.apply_blur(image, 1.0)[0], "example_image_blurred.jpg")
    # ambient.save_image(ambient.apply_mask(image, 0.1)[0], "example_image_masked.jpg")
    # ambient.save_image(ambient.apply_jpeg_compression(image, 40)[0], "example_image_jpeg_compressed.jpg")
    # ambient.save_image(ambient.apply_motion_blur(image, 4.0, 0.0)[0], "example_image_motion_blurred.jpg")
    # ambient.save_image(ambient.apply_pixelate(image, 48)[0], "example_image_pixelated.jpg")
    # ambient.save_image(ambient.apply_saturation(image, 2.0)[0], "example_image_oversaturated.jpg")
    # ambient.save_image(ambient.apply_saturation(image, 0.4)[0], "example_image_undersaturated.jpg")
    # ambient.save_image(ambient.apply_color_shift(image, 20)[0], "example_image_color_shifted.jpg")

    ambient.save_image(ambient.apply_imagecorruptions(image, "gaussian_noise", 5)[0], "example_image_gaussian_noise.jpg")
    ambient.save_image(ambient.apply_imagecorruptions(image, "shot_noise", 5)[0], "example_image_shot_noise.jpg")
    ambient.save_image(ambient.apply_imagecorruptions(image, "impulse_noise", 5)[0], "example_image_impulse_noise.jpg")
    ambient.save_image(ambient.apply_imagecorruptions(image, "defocus_blur", 5)[0], "example_image_defocus_blur.jpg")
    ambient.save_image(ambient.apply_imagecorruptions(image, "motion_blur", 5)[0], "example_image_motion_blur.jpg")
    ambient.save_image(ambient.apply_imagecorruptions(image, "zoom_blur", 5)[0], "example_image_zoom_blur.jpg")
    ambient.save_image(ambient.apply_imagecorruptions(image, "snow", 5)[0], "example_image_snow.jpg")
    ambient.save_image(ambient.apply_imagecorruptions(image, "frost", 5)[0], "example_image_frost.jpg")
    ambient.save_image(ambient.apply_imagecorruptions(image, "fog", 5)[0], "example_image_fog.jpg")

if __name__ == "__main__":
    main()