from skimage import io, restoration
import matplotlib.pyplot as plt


def reduce_flicker_tv_skimage(image_path, weight=0.1):
    img = io.imread(image_path)
    denoised_img = restoration.denoise_tv_chambolle(
        img, weight=weight, channel_axis=-1)
    return (denoised_img * 255).astype('uint8')


def reduce_flicker_bilateral_skimage(image_path, sigma_color=0.05, sigma_spatial=15):
    img = io.imread(image_path)
    denoised_img = restoration.denoise_bilateral(
        img, sigma_color=sigma_color, sigma_spatial=sigma_spatial, channel_axis=-1)
    return (denoised_img * 255).astype('uint8')


if __name__ == '__main__':
    flickering_image = '/Users/lch/development/image_transformer/image/test_30_f1.jpg'
    tv_denoised_image = reduce_flicker_tv_skimage(flickering_image)
    bilateral_denoised_image = reduce_flicker_bilateral_skimage(
        flickering_image)

    if tv_denoised_image is not None:
        plt.subplot(121), plt.imshow(io.imread(flickering_image))
        plt.title('Original Image')
        plt.subplot(122), plt.imshow(tv_denoised_image)
        plt.title('Deflickered Image (TV)')
        plt.show()

    if bilateral_denoised_image is not None:
        plt.subplot(121), plt.imshow(io.imread(flickering_image))
        plt.title('Original Image')
        plt.subplot(122), plt.imshow(bilateral_denoised_image)
        plt.title('Deflickered Image (Bilateral)')
        plt.show()
