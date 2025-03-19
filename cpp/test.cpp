#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// 밴딩 패턴을 추정하여 제거하는 함수 (3채널 대응)
Mat removeBandingColor(const Mat &src)
{
  // 입력 이미지를 float형으로 변환 (0~1 범위)
  Mat floatImg;
  src.convertTo(floatImg, CV_32FC3, 1.0 / 255.0);

  // 저주파 성분을 추출 (가우시안 블러)
  Mat lowFreqImg;
  GaussianBlur(floatImg, lowFreqImg, Size(0, 0), 30);

  // 고주파 성분 계산
  Mat highFreqImg = floatImg - lowFreqImg;

  // 고주파 성분 가중치를 조절하여 밴딩 현상 완화
  Mat resultImg = lowFreqImg + highFreqImg * 0.5;

  // 결과 이미지 클리핑 및 변환
  Mat result;
  resultImg = cv::min(cv::max(resultImg, 0), 1); // 범위 초과 방지
  resultImg.convertTo(resultImg, CV_8UC3, 255.0);

  return resultImg;
}

int main()
{
  // 컬러 이미지로 로드
  Mat img = imread("../image/test_30_f1.jpg", IMREAD_COLOR);
  if (img.empty())
  {
    cerr << "이미지를 로드할 수 없습니다." << endl;
    return -1;
  }

  // 밴딩 제거 함수 적용
  Mat resultImg = removeBandingColor(img);

  // 결과 이미지 보기
  imshow("Original Color Image", img);
  imshow("Banding Removed Color Image", resultImg);
  waitKey(0);

  return 0;
}
