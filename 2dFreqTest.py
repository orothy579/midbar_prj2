import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


def show_2d_spectrum(img_path):
    """
    주어진 이미지를 푸리에 변환한 뒤,
    2D 스펙트럼(magnitude)을 log 스케일 + 정규화하여 시각화.
    """
    # 1) 이미지 로드 (그레이)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Image load failed!")
        return

    # 2) DFT (복소수) + shift
    float_img = np.float32(img)
    dft_img = cv2.dft(float_img, flags=cv2.DFT_COMPLEX_OUTPUT)
    # 2D shift 주파수가 이미지 중앙에 오도록 재배치 ( 0, 0 주파수가 이미지 중앙에 오도록)
    dft_shifted = np.fft.fftshift(dft_img, axes=[0, 1])

    # 3) 스펙트럼 크기(magnitude) 계산 & log 변환
    planes = cv2.split(dft_shifted)  # [real, imag]
    mag = cv2.magnitude(planes[0], planes[1])
    mag += 1.0  # log 값이 0 이 되는 것을 방지
    mag_log = np.log(mag)

    # 4) 0~1 사이로 Normalize (정규화)
    mag_log_norm = cv2.normalize(mag_log, None, 0, 1, cv2.NORM_MINMAX)

    h, w = mag_log_norm.shape

    # 5) 2D 스펙트럼 시각화
    plt.figure(figsize=(8, 8))
    plt.imshow(mag_log_norm, cmap='gray',
               extent=(-w/2, w/2, h/2, -h/2))
    plt.title("2D Frequency Spectrum (Normalized)")
    plt.colorbar(label="Normalized log magnitude")
    plt.show()


if __name__ == "__main__":
    # test_30_f1.jpg 파일에 대해 2D 스펙트럼 시각화
    show_2d_spectrum(
        "/Users/lch/development/image_transformer/image/test_30_f1.jpg")
