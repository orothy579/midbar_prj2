import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


def remove_horizontal_banding_single_channel(gray_img,
                                             peak_distance=5,
                                             peak_prominence=0.05,
                                             radius=5):
    """
    단일 채널(그레이)에서 수평 밴딩을 자동 제거.
    DFT -> (중앙 열 프로파일 + find_peaks) -> Notch -> iDFT
    """

    float_img = np.float32(gray_img)
    # 1) DFT + Shift
    dft_img = cv2.dft(float_img, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shifted = np.fft.fftshift(dft_img, axes=[0, 1])

    h, w = gray_img.shape
    center_y, center_x = h // 2, w // 2

    # 2) 스펙트럼 Magnitude & log
    planes = cv2.split(dft_shifted)
    magnitude = cv2.magnitude(planes[0], planes[1])
    magnitude += 1.0
    mag_log = np.log(magnitude)

    # 3) 중앙 열(세로 주파수) 프로파일 추출
    freq_profile = mag_log[:, center_x]

    # 4) 피크 탐색 (find_peaks)
    peaks, properties = find_peaks(freq_profile,
                                   distance=peak_distance,
                                   prominence=peak_prominence)

    # 5) DC(중심) 제외하고 밴딩 후보 찾기
    banding_candidates = []
    for p in peaks:
        dist_from_center = abs(p - center_y)
        if dist_from_center > 0:
            candidate_val = freq_profile[p]
            banding_candidates.append((p, dist_from_center, candidate_val))

    if len(banding_candidates) == 0:
        # 밴딩 후보가 없다면 역변환 후 반환
        dft_ishift = np.fft.ifftshift(dft_shifted, axes=[0, 1])
        recovered = cv2.idft(
            dft_ishift, flags=cv2.DFT_REAL_OUTPUT | cv2.DFT_SCALE)
        recovered_norm = cv2.normalize(
            recovered, None, 0, 255, cv2.NORM_MINMAX)
        return np.uint8(recovered_norm)

    # 6) 노치 필터 생성
    notch_mask = np.ones((h, w, 2), np.float32)
    for (peak_idx, dist_k, val) in banding_candidates:
        # 위(+)와 아래(-) 대칭 위치
        y_up = peak_idx
        y_down = 2 * center_y - peak_idx
        cv2.circle(notch_mask, (center_x, y_up), radius, (0, 0), -1)
        cv2.circle(notch_mask, (center_x, y_down), radius, (0, 0), -1)

    # 7) 필터 적용
    dft_shifted *= notch_mask

    # 8) 역변환
    dft_ishift = np.fft.ifftshift(dft_shifted, axes=[0, 1])
    recovered = cv2.idft(dft_ishift, flags=cv2.DFT_REAL_OUTPUT | cv2.DFT_SCALE)

    # 9) 정규화 및 8비트 변환
    recovered_norm = cv2.normalize(recovered, None, 0, 255, cv2.NORM_MINMAX)
    filtered_channel = np.uint8(recovered_norm)

    return filtered_channel


def remove_horizontal_banding_rgb(img_path,
                                  peak_distance=5,
                                  peak_prominence=0.05,
                                  radius=5):
    """
    RGB 이미지에 대해, 각 채널(R/G/B)을 독립적으로 '수평 밴딩' 제거.
    결과를 최종적으로 다시 RGB로 merge.
    """
    # 1) 컬러 이미지 로드 (BGR 형태로 로드됨)
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print("Image load failed:", img_path)
        return None

    # 2) BGR -> RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 채널 분리 (R, G, B)
    r_channel = img_rgb[:, :, 0]
    g_channel = img_rgb[:, :, 1]
    b_channel = img_rgb[:, :, 2]

    # 3) 채널별로 'remove_horizontal_banding_single_channel' 수행
    r_filtered = remove_horizontal_banding_single_channel(
        r_channel,
        peak_distance=peak_distance,
        peak_prominence=peak_prominence,
        radius=radius
    )
    g_filtered = remove_horizontal_banding_single_channel(
        g_channel,
        peak_distance=peak_distance,
        peak_prominence=peak_prominence,
        radius=radius
    )
    b_filtered = remove_horizontal_banding_single_channel(
        b_channel,
        peak_distance=peak_distance,
        peak_prominence=peak_prominence,
        radius=radius
    )

    # 4) 필터링된 채널 합치기 (RGB)
    filtered_rgb = cv2.merge([r_filtered, g_filtered, b_filtered])

    # 5) 다시 BGR로 변환 (원하는 경우)
    filtered_bgr = cv2.cvtColor(filtered_rgb, cv2.COLOR_RGB2BGR)

    return filtered_bgr


if __name__ == "__main__":
    # 실행 예시
    filtered_bgr = remove_horizontal_banding_rgb(
        "/Users/lch/development/image_transformer/image/test_30_f1.jpg",
        peak_distance=2,
        peak_prominence=0.02,
        radius=2
    )

    if filtered_bgr is not None:
        # 결과 시각화
        original_bgr = cv2.imread(
            "/Users/lch/development/image_transformer/image/test_30_f1.jpg")

        plt.figure(figsize=(10, 5))

        # 원본
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB))
        plt.title("Original (RGB)")

        # 필터 결과
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(filtered_bgr, cv2.COLOR_BGR2RGB))
        plt.title("Banding Removed (RGB channels separately)")
        plt.show()

        # 필요 시 파일로 저장
        cv2.imwrite("result/test_30_f1_color_filtered.jpg", filtered_bgr)
