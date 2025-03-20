import ffmpeg
import cv2
import numpy as np

# 1. 원본 이미지 경로 & 결과 저장 경로
image_path = '/Users/lch/development/image_transformer/image/test_30_f1.jpg'
output_path = '../result/gradfun.png'  # 처리된 이미지 저장 경로

try:
    # 2. ffmpeg-python으로 gradfun 필터 적용
    (
        ffmpeg
        .input(image_path)
        .filter('gradfun', '61', '32')
        #  gradfun=strength:radius[:threshold[:nb_planes]]
        #  여기서는 strength=3.5, radius=8, threshold=0, nb_planes=기본값
        .output(output_path)
        .run(overwrite_output=True)  # 기존에 output_path가 있으면 덮어쓰기
    )
    print(f"Debanded image saved to {output_path}")

    # 3. (선택) OpenCV로 결과 이미지 확인
    debanned_img = cv2.imread(output_path)
    if debanned_img is not None:
        cv2.imshow('Debanded Image (FFmpeg gradfun)', debanned_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Failed to load debanded image with OpenCV.")

except ffmpeg.Error as e:
    print(f"Error applying gradfun filter: {e.stderr.decode('utf8')}")
