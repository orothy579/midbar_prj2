import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이미지 로드
img = cv2.imread(
    "/Users/lch/development/image_transformer/image/test_30_f1.jpg", cv2.IMREAD_GRAYSCALE)

# 1. DFT 변환
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

# 2. 저역통과 필터 적용 (중앙을 남기고 외곽 제거)
rows, cols = img.shape
crow, ccol = rows // 2, cols // 2
mask = np.zeros((rows, cols, 2), np.uint8)
r = 30  # 반지름 설정 (저주파 영역 유지)
cv2.circle(mask, (ccol, crow), r, (1, 1), -1)

# 3. 필터 적용
dft_shift = dft_shift * mask

# 4. 역변환
dft_ishift = np.fft.ifftshift(dft_shift)
img_filtered = cv2.idft(dft_ishift)
img_filtered = cv2.magnitude(img_filtered[:, :, 0], img_filtered[:, :, 1])

# 5. 결과 시각화
plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title('Original Image')
plt.subplot(122), plt.imshow(
    img_filtered, cmap='gray'), plt.title('Low Pass Filtered')
plt.show()
