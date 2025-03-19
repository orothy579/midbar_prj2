import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


def remove_horizontal_banding_single_channel(gray_img,
                                             peak_distance,
                                             peak_prominence,
                                             radius):
    """
    그레이스케일 채널에 대해 '수평 밴딩'을 제거한 뒤 필터링된 채널을 반환.
    1) DFT -> shift -> (중앙 열 프로파일) -> find_peaks -> Notch -> iDFT
    """
    # 1) float 변환
    float_img = np.float32(gray_img)

    # 2) DFT + shift
    dft_img = cv2.dft(float_img, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shifted = np.fft.fftshift(dft_img, axes=[0, 1])

    h, w = gray_img.shape
    center_y, center_x = h // 2, w // 2

    # 3) Magnitude + log
    planes = cv2.split(dft_shifted)  # [real, imag]
    magnitude = cv2.magnitude(planes[0], planes[1])
    magnitude += 1.0
    mag_log = np.log(magnitude)

    # 4) 중앙 열 프로파일 (가로줄 → 세로 주파수)
    freq_profile = mag_log[:, center_x]

    # 5) 피크 탐색 (수평 밴딩 → 세로축의 피크)
    peaks, properties = find_peaks(freq_profile,
                                   distance=peak_distance,
                                   prominence=peak_prominence)

    # 6) DC(중심) 제외 (dist=0)
    banding_candidates = []
    for p in peaks:
        dist_from_center = abs(p - center_y)
        if dist_from_center > 0:
            # 피크 진폭
            candidate_val = freq_profile[p]
            banding_candidates.append((p, dist_from_center, candidate_val))

    if len(banding_candidates) == 0:
        # 밴딩 후보 없음: 역변환 후 바로 리턴
        dft_ishift = np.fft.ifftshift(dft_shifted, axes=[0, 1])
        recovered = cv2.idft(
            dft_ishift, flags=cv2.DFT_REAL_OUTPUT | cv2.DFT_SCALE)
        recovered_norm = cv2.normalize(
            recovered, None, 0, 255, cv2.NORM_MINMAX)
        return np.uint8(recovered_norm)

    # 7) 노치 필터 생성
    notch_mask = np.ones((h, w, 2), np.float32)

    for (peak_index, dist_k, peak_val) in banding_candidates:
        # y_up, y_down
        y_up = peak_index
        y_down = 2 * center_y - peak_index  # 대칭 좌표
        # 원형 노치
        cv2.circle(notch_mask, (center_x, y_up), radius, (0, 0), -1)
        cv2.circle(notch_mask, (center_x, y_down), radius, (0, 0), -1)

    # 8) 필터 적용
    dft_shifted *= notch_mask

    # 9) 역변환
    dft_ishift = np.fft.ifftshift(dft_shifted, axes=[0, 1])
    recovered = cv2.idft(dft_ishift, flags=cv2.DFT_REAL_OUTPUT | cv2.DFT_SCALE)

    recovered_norm = cv2.normalize(recovered, None, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(recovered_norm)


def remove_horizontal_banding_luma(img_path,
                                   peak_distance=5,
                                   peak_prominence=0.05,
                                   radius=5):
    """
    컬러 이미지를 로드한 뒤, YCrCb로 변환하여 'Y 채널'만 수평 밴딩 제거.
    이후 CrCb와 합쳐서 BGR로 복원.
    """
    # 1) 컬러(BGR) 이미지 로드
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print("Image load failed:", img_path)
        return None

    # 2) BGR -> YCrCb 변환
    img_ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(img_ycrcb)

    # 3) Y 채널에 대해 수평 밴딩 제거
    y_filtered = remove_horizontal_banding_single_channel(
        y,
        peak_distance=peak_distance,
        peak_prominence=peak_prominence,
        radius=radius
    )

    # 4) 필터링 결과를 Cr, Cb와 합쳐서 YCrCb 이미지 복원
    filtered_ycrcb = cv2.merge([y_filtered, cr, cb])

    # 5) YCrCb -> BGR 변환
    filtered_bgr = cv2.cvtColor(filtered_ycrcb, cv2.COLOR_YCrCb2BGR)

    return filtered_bgr


if __name__ == "__main__":
    # 실제 테스트
    result_bgr = remove_horizontal_banding_luma(
        img_path="/Users/lch/development/image_transformer/image/test_30_f1.jpg",   # 실제 파일 경로
        peak_distance=2,
        peak_prominence=0.01,
        radius=5
    )

    if result_bgr is not None:
        # 결과 시각화
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(cv2.imread(
            "/Users/lch/development/image_transformer/image/test_30_f1.jpg"), cv2.COLOR_BGR2RGB))
        plt.title("Original (Color)")

        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB))
        plt.title("Banding Removed (Y channel only)")
        plt.show()

        # 원한다면 저장
        # cv2.imwrite(
        #     "/Users/lch/development/image_transformer/image/test_30_f1.jpg", result_bgr)
