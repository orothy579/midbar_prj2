import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


def remove_horizontal_banding_single_channel(gray_img,
                                             peak_distance=5,
                                             peak_prominence=0.05,
                                             radius=5):
    """
    단일 채널(그레이)에 대해 수평 밴딩 제거.
    DFT -> 노치 필터 -> iDFT 과정을 수행.
    """
    float_img = np.float32(gray_img)
    dft_img = cv2.dft(float_img, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shifted = np.fft.fftshift(dft_img, axes=[0, 1])

    h, w = gray_img.shape
    center_y, center_x = h // 2, w // 2

    # 스펙트럼 크기 + 로그
    planes = cv2.split(dft_shifted)
    magnitude = cv2.magnitude(planes[0], planes[1])
    magnitude += 1.0
    mag_log = np.log(magnitude)

    # 중앙 열(세로 주파수) 프로파일에서 피크 탐색
    freq_profile = mag_log[:, center_x]
    peaks, properties = find_peaks(freq_profile,
                                   distance=peak_distance,
                                   prominence=peak_prominence)

    banding_candidates = []
    for p in peaks:
        dist = abs(p - center_y)
        if dist > 0:  # DC 제외
            candidate_val = freq_profile[p]
            banding_candidates.append((p, dist, candidate_val))

    if len(banding_candidates) == 0:
        # 밴딩 없음 → 역변환 후 리턴
        dft_ishift = np.fft.ifftshift(dft_shifted, axes=[0, 1])
        recovered = cv2.idft(
            dft_ishift, flags=cv2.DFT_REAL_OUTPUT | cv2.DFT_SCALE)
        recovered_norm = cv2.normalize(
            recovered, None, 0, 255, cv2.NORM_MINMAX)
        return np.uint8(recovered_norm)

    # 노치 필터 생성
    notch_mask = np.ones((h, w, 2), np.float32)
    for (peak_idx, dist_k, val) in banding_candidates:
        y_up = peak_idx
        y_down = 2 * center_y - peak_idx

        cv2.circle(notch_mask, (center_x, y_up), radius, (0, 0), -1)
        cv2.circle(notch_mask, (center_x, y_down), radius, (0, 0), -1)

    # 필터 적용
    dft_shifted *= notch_mask

    # 역변환
    dft_ishift = np.fft.ifftshift(dft_shifted, axes=[0, 1])
    recovered = cv2.idft(dft_ishift, flags=cv2.DFT_REAL_OUTPUT | cv2.DFT_SCALE)

    recovered_norm = cv2.normalize(recovered, None, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(recovered_norm)


def remove_horizontal_banding_hsv(img_path,
                                  peak_distance=1,
                                  peak_prominence=1,
                                  radius=1):
    """
    BGR -> HSV 변환 후, V 채널만 수평 밴딩 제거.
    H, S 채널은 변경 없이 유지.
    """
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print("Image load failed:", img_path)
        return None

    # 1) BGR -> HSV
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)

    # 2) V 채널만 밴딩 제거
    v_filtered = remove_horizontal_banding_single_channel(
        v,
        peak_distance=peak_distance,
        peak_prominence=peak_prominence,
        radius=radius
    )

    # 3) 합치고 다시 BGR로 변환
    filtered_hsv = cv2.merge([h, s, v_filtered])
    filtered_bgr = cv2.cvtColor(filtered_hsv, cv2.COLOR_HSV2BGR)

    return filtered_bgr


if __name__ == "__main__":
    # 테스트 예시
    result_bgr = remove_horizontal_banding_hsv(
        "/Users/lch/development/image_transformer/image/test_30_f1.jpg",
        peak_distance=1,
        peak_prominence=2.06,
        radius=1
    )

    if result_bgr is not None:
        # 비교 시각화
        original = cv2.imread(
            "/Users/lch/development/image_transformer/image/test_30_f1.jpg")
        plt.figure(figsize=(10, 5))
        # 원본
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        plt.title("Original")

        # 필터 결과
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB))
        plt.title("Banding Removed (HSV - V channel)")
        plt.show()

        # 필요하면 저장
        cv2.imwrite("result_banding_removed_hsv.jpg", result_bgr)
