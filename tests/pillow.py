from PIL import Image, ImageFilter


def reduce_flicker_pillow_blur(image_path, radius=2):
    try:
        img = Image.open(image_path)
        blurred_img = img.filter(ImageFilter.GaussianBlur(radius))
        return blurred_img
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
        return None


if __name__ == '__main__':
    flickering_image = '/Users/lch/development/image_transformer/image/test_30_f1.jpg'
    deflickered_image = reduce_flicker_pillow_blur(flickering_image)
    if deflickered_image:
        Image.open(flickering_image).show('Original Image')
        deflickered_image.show('Deflickered Image (Pillow Blur)')
