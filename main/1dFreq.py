import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


def find_banding_frequency(img_path, distance, prominence):
    """
    - img_path: 분석할 이미지 경로
    - distance: 인접 peak 최소 거리 (픽셀 단위)
    - prominence: peak가 최소한 이 정도 높이는 되어야 한다는 파라미터
    """

    # 1) 이미지 로드 (그레이)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Image load failed!")
        return None

    # 2) DFT (복소수) + shift
    float_img = np.float32(img)
    dft_img = cv2.dft(float_img, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shifted = np.fft.fftshift(dft_img, axes=[0, 1])  # 2D shift

    # 3) 스펙트럼 크기(magnitude) 계산 & log 변환
    planes = cv2.split(dft_shifted)  # [real, imag]
    mag = cv2.magnitude(planes[0], planes[1])
    mag += 1.0
    mag_log = np.log(mag)

    # 4) 0~1 사이로 Normalize (정규화)
    mag_log_norm = cv2.normalize(mag_log, None, 0, 1, cv2.NORM_MINMAX)

    # 5) 중앙 열(center column) 프로파일 추출
    h, w = mag_log_norm.shape
    center_col = w // 2  # 가운데 열 인덱스
    freq_profile = mag_log_norm[:, center_col]

    # 6) peak 찾기
    peaks, properties = find_peaks(freq_profile,
                                   distance=distance,
                                   prominence=prominence)

    # 7) 시각화 & 어떤 peak이 나왔는지 확인
    plt.figure(figsize=(8, 5))
    plt.plot(freq_profile, label='Frequency Profile (Center Column)')
    plt.plot(peaks, freq_profile[peaks], "x", label='Detected Peaks')
    plt.title("Center Column Spectrum Profile (Normalized)")
    plt.xlabel("Row index (frequency domain)")
    plt.ylabel("Normalized log Magnitude (0~1)")
    plt.legend()
    plt.show()

    # 8) DC(중심) 인덱스
    center_y = h // 2

    # 9) DC로부터 얼마나 떨어진 피크가 있는지 검사
    banding_candidates = []
    for p in peaks:
        dist_from_center = abs(p - center_y)
        if dist_from_center > 0:  # 0이면 DC 자체
            banding_candidates.append((p, dist_from_center, freq_profile[p]))

    # 10) 가장 높은 peak를 선택
    if len(banding_candidates) > 0:
        banding_candidates.sort(key=lambda x: x[2], reverse=True)
        top_peak = banding_candidates[0]
        peak_index, k, peak_value = top_peak
        print(
            f"Detected banding peak at row={peak_index}, distance={k}, magnitude={peak_value}")
        return k
    else:
        print("No significant banding peak found.")
        return None


# 실제 사용 예:
if __name__ == "__main__":
    k_value = find_banding_frequency(
        "/Users/lch/development/image_transformer/result/hsv.jpg", distance=5, prominence=0.05)
    k_value1 = find_banding_frequency(
        "/Users/lch/development/image_transformer/image/test_30_f1.jpg", distance=5, prominence=0.05
    )
    print("Estimated k =", k_value)
