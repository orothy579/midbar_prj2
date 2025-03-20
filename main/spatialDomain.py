import cv2


def reduce_flicker_gaussian_blur_opencv(image_path, kernel_size=(5, 5)):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return None
    blurred_img = cv2.GaussianBlur(img, kernel_size, 0)
    return blurred_img


def reduce_flicker_median_blur_opencv(image_path, kernel_size=5):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return None
    blurred_img = cv2.medianBlur(img, kernel_size)
    return blurred_img


def reduce_flicker_non_local_means_opencv(image_path, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return None
    denoised_img = cv2.fastNlMeansDenoisingColored(
        img, None, h, hColor, templateWindowSize, searchWindowSize)
    return denoised_img


if __name__ == '__main__':
    flickering_image = '/Users/lch/development/image_transformer/image/test_30_f1.jpg'
    gaussian_blurred_image = reduce_flicker_gaussian_blur_opencv(
        flickering_image)
    median_blurred_image = reduce_flicker_median_blur_opencv(flickering_image)
    non_local_means_denoised_image = reduce_flicker_non_local_means_opencv(
        flickering_image)

    if gaussian_blurred_image is not None:
        cv2.imshow('Original Image', cv2.imread(flickering_image))
        cv2.imshow('Gaussian Blurred Image', gaussian_blurred_image)
    if median_blurred_image is not None:
        cv2.imshow('Median Blurred Image', median_blurred_image)
    if non_local_means_denoised_image is not None:
        cv2.imshow('Non-Local Means Denoised Image',
                   non_local_means_denoised_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
