import cv2
import numpy as np
import matplotlib.pyplot as plt


def show_frequency_profile(image_path):
    # 1) 이미지 로드 (그레이스케일)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Image load failed!")
        return

    # 2) float32로 변환
    float_img = np.float32(img)

    # 3) DFT (2채널: 실수부 + 허수부)
    dft_img = cv2.dft(float_img, flags=cv2.DFT_COMPLEX_OUTPUT)

    # 4) 스펙트럼을 중앙에 맞춰 shift
    dft_shifted = np.fft.fftshift(dft_img, axes=[0, 1])

    # 5) 실수부/허수부 분리 후, 스펙트럼 크기(Magnitude) 계산
    planes = cv2.split(dft_shifted)  # planes[0] = 실수부, planes[1] = 허수부
    mag = cv2.magnitude(planes[0], planes[1])

    # 6) 로그 스케일 변환 (log(1 + magnitude))
    mag += 1
    mag = np.log(mag)

    # 7) 시각화를 위해 0~1 범위로 정규화
    mag_norm = cv2.normalize(mag, None, 0, 1, cv2.NORM_MINMAX)

    # 8) 스펙트럼의 "중심 열(column)"을 추출
    center_col = mag_norm.shape[1] // 2
    freq_profile = mag_norm[:, center_col]  # 모든 행(row)에 대해 중앙 열만 뽑음

    # 9) 그래프 그리기
    plt.plot(freq_profile)
    plt.title("Frequency Domain (Center Column Profile)")
    plt.xlabel("Row index (in frequency domain)")
    plt.ylabel("Normalized log magnitude")
    plt.show()

    plt.imshow(img, cmap='gray'), plt.title('Original Image')


# 예시 사용
if __name__ == "__main__":
    show_frequency_profile(
        "/Users/lch/development/image_transformer/image/test_30_f1.jpg")  # 실제 이미지 경로 넣어주세요
