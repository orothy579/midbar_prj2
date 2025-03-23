import matplotlib.pyplot as plt
import sys
import cv2
import numpy as np
from scipy.signal import find_peaks
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QSlider, QFileDialog, QPushButton, QComboBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage

import matplotlib
matplotlib.use("Qt5Agg")  # PyQt5와 호환

plt.ion()  # 인터랙티브 모드 활성화


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
    offsets = [2, 3, 5, 6]
    center_y, center_x = h // 2, w // 2

    # 주파수 세기(log 스케일)
    planes = cv2.split(dft_shifted)
    magnitude = cv2.magnitude(planes[0], planes[1])
    magnitude += 1.0
    mag_log = np.log(magnitude)

    # 중앙열 프로파일
    freq_profile = mag_log[:, center_x]

    distance_in_samples = int(round(peak_distance))
    distance_in_samples = max(distance_in_samples, 1)

    # 피크 탐색
    peaks, _ = find_peaks(freq_profile,
                          distance=distance_in_samples,
                          prominence=peak_prominence)

    banding_candidates = []
    for p in peaks:
        dist = abs(p - center_y)
        if dist > 0:
            banding_candidates.append(p)

    # 노치 필터 마스크 초기화 (모든 주파수 통과)
    notch_mask = np.ones((h, w, 2), np.float32)

    # 반영할 "사각형 너비/높이"를 정의합니다.
    # - radius를 그대로 쓰되, 사각형 모양으로 확장하고 싶다면
    #   가로/세로 반을 정해서 사용합니다.

    h, w = gray_img.shape
    offsets = [1, 2, 3, 5, 6]
    center_y, center_x = h // 2, w // 2

    rect_half_width = 1  # x 방향 반폭 (필요에 따라 조정)
    rect_half_height = radius  # y 방향 반높이 (필요에 따라 조정)

    for off in offsets:
        y_up = center_y + off
        y_down = center_y - off

        if abs(y_up - center_y) > radius:
            top_left = (center_x - rect_half_width, y_up - rect_half_height)
            bottom_right = (center_x + rect_half_width,
                            y_up + rect_half_height)
            cv2.rectangle(notch_mask, top_left,
                          bottom_right, (1/7, 1/7), -1)
        if abs(y_down - center_y) > radius:
            top_left = (center_x - rect_half_width, y_down - rect_half_height)
            bottom_right = (center_x + rect_half_width,
                            y_down + rect_half_height)
            cv2.rectangle(notch_mask, top_left,
                          bottom_right, (1/7, 1/7), -1)

        # cv2.circle(notch_mask, (center_x, y_up), radius, (0, 0), -1)
        # cv2.circle(notch_mask, (center_x, y_down), radius, (0, 0), -1)

    for p in banding_candidates:
        y_up = p
        y_down = 2 * center_y - p

    # 노치 필터 적용
    dft_shifted *= notch_mask

    # 역변환
    dft_ishift = np.fft.ifftshift(dft_shifted, axes=[0, 1])
    recovered = cv2.idft(dft_ishift, flags=cv2.DFT_REAL_OUTPUT | cv2.DFT_SCALE)
    recovered_norm = cv2.normalize(recovered, None, 0, 255, cv2.NORM_MINMAX)

    recovered_u8 = np.uint8(recovered_norm)

    # CLAHE 적용 (잔물결, 대비 문제 완화)
    # clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(15, 15))
    # dft_img = clahe.apply(recovered_u8)

    return dft_img, dft_shifted, np.uint8(recovered_norm)


def remove_horizontal_banding_bgr(bgr_img, mode,
                                  peak_distance,
                                  peak_prominence,
                                  radius):
    """
    BGR 이미지를 입력받고, mode(Gray, RGB, HSV, YCrCb)에 따라
    특정 채널 또는 모든 채널에 대해 remove_banding_single_channel을 수행한 뒤
    BGR로 되돌려 최종 이미지를 반환.
    + 노치 필터 적용 전 FFT를 '대표 1채널'만 반환
      (Gray면 그대로, RGB/HSV/YCrCb면 그 중 하나를 대표로...)
    """
    # 시각화용으로 쓸 "노치 필터 적용 전" dft_shifted / dft_img
    # -> Gray 모드면 바로 그레이 채널, RGB/HSV/YCbCr면 그냥 첫 채널? 원하는 방식으로...
    # 여기서는 "mode==Gray"인 경우에만 시각화한다고 가정(가장 단순).
    # 실제로 RGB 등에서 모든 채널 FFT를 보고 싶으면, 여러 개 관리해야 함.

    if mode == "Gray":
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        dft_img, dft_shifted, out_gray = remove_banding_single_channel(
            gray, peak_distance, peak_prominence, radius
        )
        filtered_bgr = cv2.cvtColor(out_gray, cv2.COLOR_GRAY2BGR)
        return dft_img, dft_shifted, filtered_bgr

    elif mode == "RGB":
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        channels = cv2.split(rgb)
        out_channels = []
        for ch in channels:
            _, _, out_ch = remove_banding_single_channel(
                ch, peak_distance, peak_prominence, radius
            )
            out_channels.append(out_ch)
        filtered_rgb = cv2.merge(out_channels)
        return None, None, cv2.cvtColor(filtered_rgb, cv2.COLOR_RGB2BGR)

    elif mode == "HSV":
        hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(hsv)
        dft_img_v, dft_shifted_v, V_filtered = remove_banding_single_channel(
            V, peak_distance, peak_prominence, radius
        )
        hsv_out = cv2.merge([H, S, V_filtered])
        filtered_bgr = cv2.cvtColor(hsv_out, cv2.COLOR_HSV2BGR)
        return dft_img_v, dft_shifted_v, filtered_bgr

    elif mode == "YCrCb":
        ycrcb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YCrCb)
        Y, Cr, Cb = cv2.split(ycrcb)
        dft_img_y, dft_shifted_y, Y_filtered = remove_banding_single_channel(
            Y, peak_distance, peak_prominence, radius
        )
        ycrcb_out = cv2.merge([Y_filtered, Cr, Cb])
        filtered_bgr = cv2.cvtColor(ycrcb_out, cv2.COLOR_YCrCb2BGR)
        return dft_img_y, dft_shifted_y, filtered_bgr

    # 모드가 정의 밖이면
    return None, None, bgr_img

# 360, 357, 363, 354, 366


def cvimg_to_qpixmap(cv_img):
    """OpenCV 이미지(numpy) -> QPixmap 변환 (Gray or BGR 3ch)."""
    if len(cv_img.shape) == 2:
        h, w = cv_img.shape
        bytes_per_line = w
        qimg = QImage(cv_img.data, w, h, bytes_per_line,
                      QImage.Format_Indexed8)
    else:
        h, w, ch = cv_img.shape
        cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        bytes_per_line = ch * w
        qimg = QImage(cv_img_rgb.data, w, h,
                      bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


class BandingRemovalApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Horizontal Banding Removal")

        # 기본 파라미터
        self.peak_distance = 1
        self.peak_prominence = 1
        self.radius = 1

        self.original_bgr = None  # 원본 이미지(BGR)
        self.filtered_bgr = None  # 필터링 결과(BGR)

        # 플롯 관련 멤버
        self.fig = None
        self.ax1 = None
        self.ax2 = None
        self.img2D = None
        self.line1 = None

        self.init_ui()
        self.init_plot()

    def init_ui(self):
        """PyQt 메인 GUI 세팅"""
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        vlayout = QVBoxLayout()
        main_widget.setLayout(vlayout)

        # 이미지 표시용 QLabel
        self.image_label = QLabel("No Image Loaded")
        self.image_label.setAlignment(Qt.AlignCenter)
        vlayout.addWidget(self.image_label)

        # 슬라이더 레이아웃
        slider_layout = QHBoxLayout()

        # Slider 1: peak_distance
        dist_layout = QHBoxLayout()
        self.slider_dist = QSlider(Qt.Horizontal)
        self.slider_dist.setMinimum(1)
        self.slider_dist.setMaximum(300)
        self.slider_dist.setValue(int(self.peak_distance))
        self.slider_dist.valueChanged.connect(self.on_slider_update)
        self.label_dist = QLabel(f"{self.peak_distance:}")
        dist_layout.addWidget(QLabel("distance:"))
        dist_layout.addWidget(self.slider_dist)
        dist_layout.addWidget(self.label_dist)
        slider_layout.addLayout(dist_layout)

        # Slider 2: peak_prominence
        prom_layout = QHBoxLayout()
        self.slider_prom = QSlider(Qt.Horizontal)
        self.slider_prom.setMinimum(1)
        self.slider_prom.setMaximum(1000)
        self.slider_prom.setValue(int(self.peak_prominence * 10000))
        self.slider_prom.valueChanged.connect(self.on_slider_update)
        self.label_prom = QLabel(f"{self.peak_prominence:}")
        prom_layout.addWidget(QLabel("prominence:"))
        prom_layout.addWidget(self.slider_prom)
        prom_layout.addWidget(self.label_prom)
        slider_layout.addLayout(prom_layout)

        # Slider 3: radius
        rad_layout = QHBoxLayout()
        self.slider_rad = QSlider(Qt.Horizontal)
        self.slider_rad.setMinimum(1)
        self.slider_rad.setMaximum(30)
        self.slider_rad.setValue(self.radius)
        self.slider_rad.valueChanged.connect(self.on_slider_update)
        self.label_rad = QLabel(f"{self.radius}")
        rad_layout.addWidget(QLabel("radius:"))
        rad_layout.addWidget(self.slider_rad)
        rad_layout.addWidget(self.label_rad)
        slider_layout.addLayout(rad_layout)

        vlayout.addLayout(slider_layout)

        # 색상 모드 선택 콤보박스
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Color Mode:"))
        self.color_mode_combo = QComboBox()
        self.color_mode_combo.addItems(["Gray", "RGB", "HSV", "YCrCb"])
        self.color_mode_combo.currentIndexChanged.connect(self.apply_filter)
        mode_layout.addWidget(self.color_mode_combo)
        vlayout.addLayout(mode_layout)

        # 버튼 레이아웃
        btn_layout = QHBoxLayout()
        vlayout.addLayout(btn_layout)

        # Load Image Button
        btn_load = QPushButton("Load Image")
        btn_load.clicked.connect(self.load_image)
        btn_layout.addWidget(btn_load)

    def init_plot(self):
        """Matplotlib Figure/Axes 한 번만 생성하고, 2D/1D 표시를 위한 객체를 잡아둔다."""
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(8, 4))

        # 2D 스펙트럼 (log scale)
        dummy_2d = np.zeros((10, 10), dtype=np.float32)
        self.img2D = self.ax1.imshow(dummy_2d)
        self.ax1.set_title("2D Spectrum (Log)")

        # 중심열 1D 프로파일
        dummy_1d = np.zeros(10, dtype=np.float32)
        (self.line1,) = self.ax2.plot(dummy_1d)
        self.ax2.set_title("Center Column Profile (Log scale)")

        self.fig.tight_layout()
        self.fig.show()

    def update_fft_plot(self, dft_shifted):
        """
        노치 필터 적용 전의 dft_shifted를 받아
        2D(log scale)와 1D(center column) 프로파일을 업데이트한다.
        """
        if dft_shifted is None:
            return

        # 분리 후 log 스케일
        planes = cv2.split(dft_shifted)
        magnitude = cv2.magnitude(planes[0], planes[1])
        magnitude += 1.0
        mag_log = np.log(magnitude)

        h, w = mag_log.shape
        cx = w // 2

        center_col = mag_log[:, cx]

        # 2D 스펙트럼 갱신
        self.img2D.set_data(mag_log)
        # 색상 범위 업데이트(자동)
        self.img2D.set_clim(vmin=mag_log.min(), vmax=mag_log.max())

        # 축 범위 재정의: -w/2 ~ w/2, -h/2 ~ h/2
        extent = [-w//2, w//2, -h//2, h//2]
        # 만약 y축을 아래->위로 그리려면 origin='lower' 설정
        self.img2D.set_extent(extent)

        # 1D 스펙트럼 갱신
        x = np.arange(len(center_col))
        self.line1.set_data(x, center_col)
        self.ax2.relim()
        self.ax2.autoscale_view()

        # 실시간 반영
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def load_image(self):
        """이미지 파일 열기 & BGR로 보관."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_path:
            img_bgr = cv2.imread(file_path, cv2.IMREAD_COLOR)
            if img_bgr is None:
                return
            # 원본 BGR 보관
            self.original_bgr = img_bgr
            # 필터 적용
            self.apply_filter()

    def apply_filter(self):
        """슬라이더 값 반영 -> 밴딩 제거 -> 표시."""
        if self.original_bgr is None:
            return

        # 슬라이더 값 읽기
        self.peak_distance = self.slider_dist.value() / 10.0
        self.peak_prominence = self.slider_prom.value() / 10000.0
        self.radius = self.slider_rad.value()

        # 라벨 업데이트
        self.label_dist.setText(f"{self.peak_distance:.1f}")
        self.label_prom.setText(f"{self.peak_prominence:.4f}")
        self.label_rad.setText(f"{self.radius}")

        # 현재 모드
        mode = self.color_mode_combo.currentText()

        # 필터 적용
        dft_img, dft_shifted, self.filtered_bgr = remove_horizontal_banding_bgr(
            self.original_bgr,
            mode,
            peak_distance=self.peak_distance,
            peak_prominence=self.peak_prominence,
            radius=self.radius
        )

        # self.filtered_bgr = cv2.GaussianBlur(self.filtered_bgr, (5, 5), 60)

        #  "노치 필터 적용 전" 스펙트럼 보기 원하는 경우
        #    dft_shifted가 "필터 적용 후" 상태가 되지 않도록 주의해야 함.
        #    지금 코드에서는 remove_banding_single_channel 내에서
        #    이미 notch_mask를 곱해버리므로 "전/후" 구분이 모호.
        #
        #    간단히 "전"이 보고 싶으면, remove_banding_single_channel 진입 직후에
        #    dft_shifted를 복사해서 리턴시키거나 별도의 함수를 두면 됨.
        #
        # 여기서는 "Gray 모드"일 때만 dft_shifted를 리턴하도록 했으므로,
        # Gray일 경우 그래프 업데이트, 아니면 패스
        if dft_shifted is not None:
            self.update_fft_plot(dft_shifted)

        # 화면 표시(QPixmap)
        pixmap = cvimg_to_qpixmap(self.filtered_bgr)
        self.image_label.setPixmap(pixmap.scaled(
            self.image_label.width(),
            self.image_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        ))

    def on_slider_update(self, value):
        """슬라이더 값 바뀔 때마다 실시간 반영."""
        self.apply_filter()


def main():
    app = QApplication(sys.argv)
    win = BandingRemovalApp()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
