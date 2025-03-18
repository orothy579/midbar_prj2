#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main()
{
  // 이미지 로드
  Mat inputImage = imread("../image/test_30_f1.jpg");

  // 이미지가 제대로 로드되었는지 확인
  if (inputImage.empty())
  {
    cout << "이미지를 로드할 수 없습니다." << endl;
    return -1;
  }

  // 이미지를 그레이스케일로 변환 (밴딩 현상은 주로 밝기 변화에서 나타나므로)
  Mat grayImage;
  cvtColor(inputImage, grayImage, COLOR_BGR2GRAY);

  // CLAHE 객체 생성
  Ptr<CLAHE> clahe = createCLAHE();
  clahe->setClipLimit(2);              // 대비 제한 값 설정 (조절 가능)
  clahe->setTilesGridSize(Size(8, 8)); // 타일 격자 크기 설정 (조절 가능)

  // CLAHE 적용
  Mat outputImage;
  clahe->apply(grayImage, outputImage);

  // 결과 이미지 표시
  imshow("Original Image", grayImage);
  imshow("CLAHE Processed Image", outputImage);

  waitKey(0);
  return 0;
}