import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


def remove_horizontal_banding_automatic(img_path, peak_distance, peak_prominence, radius):
    """
    - img_path: 분석할 이미지 파일 경로 (그레이스케일에 적합)
    - peak_distance: 인접 peak 간 최소 거리 (픽셀 단위)
    - peak_prominence: peak 최소 두드러짐 정도
    - radius: 노치 필터 반경
    """

    # 1) 이미지 로드 (그레이스케일)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Image load failed:", img_path)
        return None

    # 2) DFT (2채널 복소수) + fftshift
    float_img = np.float32(img)
    dft_img = cv2.dft(float_img, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shifted = np.fft.fftshift(dft_img, axes=[0, 1])

    h, w = img.shape
    center_y, center_x = h // 2, w // 2  # (row, col)

    # 3) 스펙트럼(진폭) 계산 & log
    real_part, imag_part = cv2.split(dft_shifted)
    magnitude = cv2.magnitude(real_part, imag_part)
    magnitude += 1.0
    mag_log = np.log(magnitude)

    # 4) 스펙트럼 정규화 (시각화용)
    mag_log_norm = cv2.normalize(mag_log, None, 0, 1, cv2.NORM_MINMAX)

    # [Optional] 스펙트럼 시각화 (Before Filtering)
    plt.figure(figsize=(6, 5))
    plt.imshow(mag_log_norm, cmap='gray')
    plt.title("Spectrum (Before)")
    plt.colorbar(label="Normalized log Magnitude")
    plt.show()

    # 5) 수평 밴딩 → 세로축 주파수가 강함
    #    -> '중앙 열' 1D 프로파일을 뽑아서 peak를 찾는다
    center_col = center_x  # 가운데 열
    freq_profile = mag_log_norm[:, center_col]

    # 6) peak 찾기 (find_peaks)
    #    prominence / distance 등 파라미터를 조절해서
    #    의미있는 봉우리만 골라냄
    peaks, properties = find_peaks(freq_profile,
                                   distance=peak_distance,
                                   prominence=peak_prominence)

    # 7) DC(중심) 인덱스 주변은 제외 (dist=0)
    banding_candidates = []
    for p in peaks:
        dist_from_center = abs(p - center_y)
        if dist_from_center > 0:  # 0이면 DC 자체
            # 피크의 실제 log-mag 값
            candidate_val = freq_profile[p]
            banding_candidates.append((p, dist_from_center, candidate_val))

    # 8) 후보가 없으면 → No banding found
    if len(banding_candidates) == 0:
        print("[INFO] No significant horizontal banding peak found.")
        return img

    # 만약 여러 개가 나오면, 전부 제거 (원하면 상위 몇 개만 제거 가능)
    # 아래는 일단 '전부 제거' 가정
    # banding_candidates를 진폭 내림차순 정렬 후, for문 돌릴 수 있음
    # 여기서는 그냥 그대로 사용
    # banding_candidates.sort(key=lambda x: x[2], reverse=True)

    # 9) 노치 필터 생성
    # dft_shifted shape: (h, w, 2)
    # => 2채널 (실수, 허수) 이므로, 노치도 동일 shape
    notch_mask = np.ones((h, w, 2), np.float32)

    for (peak_index, dist_k, peak_val) in banding_candidates:
        # center_col = center_x
        # peak_index = row index
        y_up = peak_index
        y_down = 2 * center_y - peak_index  # 대칭점 (스펙트럼이 대칭적)
        # 원래는 +/-k 위치이므로, y_up = center_y + k, y_down = center_y - k

        # 노치(원) 그리기
        cv2.circle(notch_mask, (center_x, y_up), radius, (0, 0), -1)
        cv2.circle(notch_mask, (center_x, y_down), radius, (0, 0), -1)

    # 10) 노치 적용
    dft_shifted *= notch_mask

    # 11) 스펙트럼 시각화 (After)
    real2, imag2 = cv2.split(dft_shifted)
    mag2 = cv2.magnitude(real2, imag2)
    mag2_log = np.log(mag2 + 1)
    mag2_norm = cv2.normalize(mag2_log, None, 0, 1, cv2.NORM_MINMAX)

    plt.figure(figsize=(6, 5))
    plt.imshow(mag2_norm, cmap='gray')
    plt.title("Spectrum (After)")
    plt.colorbar(label="Normalized log Magnitude")
    plt.show()

    # 12) 역변환(iDFT)
    dft_ishift = np.fft.ifftshift(dft_shifted, axes=[0, 1])
    recovered = cv2.idft(dft_ishift, flags=cv2.DFT_REAL_OUTPUT | cv2.DFT_SCALE)

    # 13) 정규화하여 8비트 변환
    recovered_norm = cv2.normalize(recovered, None, 0, 255, cv2.NORM_MINMAX)
    recovered_img = np.uint8(recovered_norm)

    return recovered_img


if __name__ == "__main__":
    # 실제 테스트
    filtered = remove_horizontal_banding_automatic(
        "/Users/lch/development/image_transformer/image/test_60_f1_3.jpg",
        peak_distance=5,
        peak_prominence=0.02,
        radius=10
    )

    if filtered is not None:
        plt.imshow(filtered, cmap='gray')
        plt.title("Banding Removed (Automatic Notch)")
        plt.show()
