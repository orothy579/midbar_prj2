import cv2
import numpy as np
from scipy.signal import find_peaks


def remove_banding_single_channel(gray_img,
                                  peak_distance,
                                  peak_prominence,
                                  radius):
    """
    단일 채널에서 수평 밴딩 제거.
    노치 필터 적용 전의 dft_shifted(시각화 용)와,
    최종 복원 이미지를 반환.
    """
    float_img = np.float32(gray_img)
    dft_img = cv2.dft(float_img, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shifted = np.fft.fftshift(dft_img, axes=[0, 1])

    h, w = gray_img.shape
    center_y, center_x = h // 2, w // 2

    # 주파수 세기(log 스케일)
    planes = cv2.split(dft_shifted)
    magnitude = cv2.magnitude(planes[0], planes[1])
    magnitude += 1.0
    mag_log = np.log(magnitude)

    # 중앙열 프로파일
    freq_profile = mag_log[:, center_x]

    # 피크 탐색 (기존 피크 탐색 코드는 유지하거나 아래의 계산된 주파수 사용)
    # peaks, _ = find_peaks(freq_profile,
    #                       distance=peak_distance,
    #                       prominence=peak_prominence)

    banding_candidates = []
    # 120 Hz LED, 30 fps 웹캠
    flicker_freq = 120  # Hz
    frame_rate = 30    # fps

    # 시간 영역에서의 앨리어싱 주파수 계산
    aliased_freqs = []
    for n in range(1, 5):  # 최대 4배까지 고려 (120 / 30 = 4)
        freq = abs(flicker_freq - n * frame_rate)
        if freq > 0 and freq not in aliased_freqs:
            aliased_freqs.append(freq)
    aliased_freqs.sort()
    print(f"앨리어싱된 시간 주파수: {aliased_freqs} Hz")

    # 시간 주파수를 공간 주파수로 변환 (사이클/이미지 높이)
    spatial_freqs = [freq / frame_rate for freq in aliased_freqs]
    print(f"대략적인 공간 주파수 (사이클/이미지 높이): {spatial_freqs}")

    # 공간 주파수를 DFT에서의 인덱스로 변환
    # 이미지 높이 h에 대해 s번의 사이클이 있다면, DFT에서는 중심으로부터 +/- s 떨어진 위치에 해당
    for s_freq in spatial_freqs:
        peak_offset = int(round(s_freq * h))  # 이미지 높이를 곱합니다.
        y_up = center_y + peak_offset
        y_down = center_y - peak_offset

        # 유효한 인덱스이고 DC 성분 보호
        if 0 < y_up < h and abs(y_up - center_y) > radius and y_up not in banding_candidates:
            banding_candidates.append(y_up)
        if 0 < y_down < h and abs(y_down - center_y) > radius and y_down != y_up and y_down not in banding_candidates:
            banding_candidates.append(y_down)

    # 기존에 탐지된 피크를 활용하고 싶다면 아래 코드를 추가하여 병합할 수 있습니다.
    # if peaks is not None:
    #     for p in peaks:
    #         dist = abs(p - center_y)
    #         if dist > radius and p not in banding_candidates:
    #             banding_candidates.append(p)
    #             y_down = 2 * center_y - p
    #             if y_down != p and y_down not in banding_candidates and 0 < y_down < h and abs(y_down - center_y) > radius:
    #                 banding_candidates.append(y_down)

    # 중복 제거 및 정렬
    banding_candidates = sorted(list(set(banding_candidates)))
    print(f"제거할 밴딩 후보 주파수 (DFT 인덱스): {banding_candidates}")

    # 노치 필터 마스크
    notch_mask = np.ones((h, w, 2), np.float32)
    for p in banding_candidates:
        y_up = p
        y_down = 2 * center_y - p

        # DC 성분 (center_y) 보호 조건
        if abs(y_up - center_y) > radius and 0 <= y_up < h:
            cv2.circle(notch_mask, (center_x, y_up), radius, (0, 0), -1)
        if abs(y_down - center_y) > radius and 0 <= y_down < h and y_down != y_up:
            cv2.circle(notch_mask, (center_x, y_down), radius, (0, 0), -1)

    # 노치 필터 적용
    dft_shifted *= notch_mask

    # 역변환
    dft_ishift = np.fft.ifftshift(dft_shifted, axes=[0, 1])
    recovered = cv2.idft(dft_ishift, flags=cv2.DFT_REAL_OUTPUT | cv2.DFT_SCALE)
    recovered_norm = cv2.normalize(recovered, None, 0, 255, cv2.NORM_MINMAX)

    return dft_img, dft_shifted, np.uint8(recovered_norm)
