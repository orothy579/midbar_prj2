# 이건 안됨
import cv2
import numpy as np


def improve_color_image(image_path, h=10, h_color=10, template_window_size=5, search_window_size=11):
    """
    Improves a color image by reducing noise using non-local means denoising.

    Args:
        image_path (str): Path to the input color image.
        h (int): Filter strength parameter for luminance. Higher values remove more noise.
        h_color (int): Filter strength parameter for color components.
        template_window_size (int): Size of the template window (should be odd).
        search_window_size (int): Size of the search window (should be odd).

    Returns:
        numpy.ndarray: The denoised color image.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not open or find the image at {image_path}")
        return None

    # Apply non-local means denoising for color images
    denoised_image = cv2.fastNlMeansDenoisingColored(
        img, None, h, h_color, template_window_size, search_window_size)
    return denoised_image


if __name__ == '__main__':
    # Replace 'your_color_image.png' with the actual path to your color image file
    color_image_path = '/Users/lch/development/image_transformer/image/test_30_f1.jpg'
    improved_image = improve_color_image(color_image_path)

    if improved_image is not None:
        # Display the original and improved images
        original_image = cv2.imread(color_image_path)
        cv2.imshow('Original Color Image', original_image)
        cv2.imshow('Improved Color Image', improved_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
