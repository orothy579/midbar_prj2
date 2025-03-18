import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


def find_banding_frequency(img_path, distance=5, prominence=0.1):
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

    # 4) 중앙 열(center column) 프로파일 추출
    h, w = mag_log.shape
    center_col = w // 2  # 가운데 열 인덱스
    freq_profile = mag_log[:, center_col]

    # 5) peak 찾기
    #    find_peaks 함수: (데이터, 최소 간격 등등) → peak 인덱스 반환
    peaks, properties = find_peaks(freq_profile,
                                   distance=distance,
                                   prominence=prominence)

    # 6) 시각화 & 어떤 peak이 나왔는지 확인
    plt.figure(figsize=(8, 5))
    plt.plot(freq_profile, label='Frequency Profile (Center Column)')
    plt.plot(peaks, freq_profile[peaks], "x", label='Detected Peaks')
    plt.title("Center Column Spectrum Profile")
    plt.xlabel("Row index (frequency domain)")
    plt.ylabel("log Magnitude")
    plt.legend()
    plt.show()

    # 7) DC(중심) 인덱스
    center_y = h // 2

    # 8) DC로부터 얼마나 떨어진 피크가 있는지 검사
    #    (가장 가까운 피크 or 가장 강한 피크를 밴딩 주파수로 가정 가능)
    banding_candidates = []
    for p in peaks:
        dist_from_center = abs(p - center_y)
        if dist_from_center > 0:  # 0이면 DC 자체
            banding_candidates.append((p, dist_from_center, freq_profile[p]))

    # 9) dist_from_center(= k)가 가장 작은 or 가장 높은 peak를 골라봄
    #    (예: 가장 강한 peak)
    if len(banding_candidates) > 0:
        # 정렬 기준: 진폭이 가장 큰 애를 우선 or DC와의 거리 등등
        # 여기서는 진폭(세 번째 튜플 값)을 기준으로 내림차순
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
        "test_60_f1.jpg", distance=5, prominence=0.2)
    print("Estimated k =", k_value)
