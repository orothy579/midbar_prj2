import sys
import cv2
import numpy as np
from scipy.signal import find_peaks
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QSlider, QFileDialog, QPushButton
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage


def remove_horizontal_banding_single_channel(gray_img,
                                             peak_distance=5,
                                             peak_prominence=0.05,
                                             radius=5):
    """단일 채널에서 수평 밴딩 제거."""
    float_img = np.float32(gray_img)
    dft_img = cv2.dft(float_img, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shifted = np.fft.fftshift(dft_img, axes=[0, 1])

    h, w = gray_img.shape
    center_y, center_x = h // 2, w // 2

    # Magnitude + log
    planes = cv2.split(dft_shifted)
    magnitude = cv2.magnitude(planes[0], planes[1])
    magnitude += 1.0
    mag_log = np.log(magnitude)

    # 중앙 열 프로파일
    freq_profile = mag_log[:, center_x]
    # 피크 탐색
    peaks, _ = find_peaks(
        freq_profile, distance=peak_distance, prominence=peak_prominence)

    banding_candidates = []
    for p in peaks:
        dist = abs(p - center_y)
        if dist > 0:
            banding_candidates.append(p)

    # 노치 필터
    notch_mask = np.ones((h, w, 2), np.float32)
    for p in banding_candidates:
        y_up = p
        y_down = 2 * center_y - p
        cv2.circle(notch_mask, (center_x, y_up), radius, (0, 0), -1)
        cv2.circle(notch_mask, (center_x, y_down), radius, (0, 0), -1)

    # 적용
    dft_shifted *= notch_mask

    # 역변환
    dft_ishift = np.fft.ifftshift(dft_shifted, axes=[0, 1])
    recovered = cv2.idft(dft_ishift, flags=cv2.DFT_REAL_OUTPUT | cv2.DFT_SCALE)
    recovered_norm = cv2.normalize(recovered, None, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(recovered_norm)


def remove_horizontal_banding_gray(img, peak_distance, peak_prominence, radius):
    """그레이스케일 이미지에서 수평 밴딩 제거."""
    return remove_horizontal_banding_single_channel(img,
                                                    peak_distance=peak_distance,
                                                    peak_prominence=peak_prominence,
                                                    radius=radius)


def cvimg_to_qpixmap(cv_img):
    """OpenCV 이미지(numpy) -> QPixmap 변환 (Gray or BGR 3ch)."""
    if len(cv_img.shape) == 2:
        # 그레이스케일
        h, w = cv_img.shape
        bytes_per_line = w
        qimg = QImage(cv_img.data, w, h, bytes_per_line,
                      QImage.Format_Indexed8)
    else:
        # 컬러
        h, w, ch = cv_img.shape
        # BGR -> RGB
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
        self.peak_distance = 5
        self.peak_prominence = 5
        self.radius = 5

        self.original_gray = None  # 원본 이미지(그레이)
        self.filtered_gray = None  # 필터링 결과

        # 메인 위젯
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        # 레이아웃
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
        self.slider_dist.setMaximum(500)
        self.slider_dist.setValue(self.peak_distance)
        self.slider_dist.valueChanged.connect(self.on_slider_update)
        self.label_dist = QLabel(f"{self.peak_distance}")
        dist_layout.addWidget(QLabel("distance:"))
        dist_layout.addWidget(self.slider_dist)
        dist_layout.addWidget(self.label_dist)
        slider_layout.addLayout(dist_layout)

        # Slider 2: peak_prominence
        prom_layout = QHBoxLayout()
        self.slider_prom = QSlider(Qt.Horizontal)
        self.slider_prom.setMinimum(1)
        self.slider_prom.setMaximum(500)
        self.slider_prom.setValue(self.peak_prominence)
        self.slider_prom.valueChanged.connect(self.on_slider_update)

        self.label_prom = QLabel(f"{self.peak_prominence: .2f}")
        prom_layout.addWidget(QLabel("prominence:"))
        prom_layout.addWidget(self.slider_prom)
        prom_layout.addWidget(self.label_prom)
        slider_layout.addLayout(prom_layout)

        # Slider 3: radius
        rad_layout = QHBoxLayout()
        self.slider_rad = QSlider(Qt.Horizontal)
        self.slider_rad.setMinimum(1)
        self.slider_rad.setMaximum(50)
        self.slider_rad.setValue(self.radius)
        self.slider_rad.valueChanged.connect(self.on_slider_update)

        self.label_rad = QLabel(f"{self.radius}")  # 슬라이더 오른쪽에 배치
        rad_layout.addWidget(QLabel("radius:"))
        rad_layout.addWidget(self.slider_rad)
        rad_layout.addWidget(self.label_rad)
        slider_layout.addLayout(rad_layout)

        vlayout.addLayout(slider_layout)

        # 버튼 레이아웃
        btn_layout = QHBoxLayout()
        vlayout.addLayout(btn_layout)

        # Load Image Button
        btn_load = QPushButton("Load Image")
        btn_load.clicked.connect(self.load_image)
        btn_layout.addWidget(btn_load)

        # Refresh Button
        btn_refresh = QPushButton("Refresh Filter")
        btn_refresh.clicked.connect(self.apply_filter)
        btn_layout.addWidget(btn_refresh)

    def load_image(self):
        """이미지 파일 열기 & 그레이스케일로 보관."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            img_bgr = cv2.imread(file_path, cv2.IMREAD_COLOR)
            if img_bgr is None:
                return
            # 그레이 변환
            self.original_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            # 필터 적용
            self.apply_filter()

    def apply_filter(self):
        """슬라이더 값 반영 -> 밴딩 제거 -> 표시."""
        if self.original_gray is None:
            return

        self.peak_distance = self.slider_dist.value()
        self.peak_prominence = self.slider_prom.value() / 100.0
        self.radius = self.slider_rad.value()

        self.label_dist.setText(f"{self.peak_distance}")
        self.label_prom.setText(f"{self.peak_prominence:.2f}")
        self.label_rad.setText(f"{self.radius}")

        # 필터 적용
        self.filtered_gray = remove_horizontal_banding_gray(
            self.original_gray,
            peak_distance=self.peak_distance,
            peak_prominence=self.peak_prominence,
            radius=self.radius
        )

        # 화면 표시
        pixmap = cvimg_to_qpixmap(self.filtered_gray)
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
