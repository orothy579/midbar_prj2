#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;

// shiftDFT 함수
void shiftDFT(Mat &src)
{
  int cx = src.cols / 2;
  int cy = src.rows / 2;

  Mat q0(src, Rect(0, 0, cx, cy));
  Mat q1(src, Rect(cx, 0, cx, cy));
  Mat q2(src, Rect(0, cy, cx, cy));
  Mat q3(src, Rect(cx, cy, cx, cy));

  Mat tmp;
  q0.copyTo(tmp);
  q3.copyTo(q0);
  tmp.copyTo(q3);

  q1.copyTo(tmp);
  q2.copyTo(q1);
  tmp.copyTo(q2);
}

// 노치 필터 적용 밴딩 제거 함수
Mat removeBandingDFT(const Mat &src, int radius = 10)
{
  Mat floatImg, dftImg;
  src.convertTo(floatImg, CV_32F);

  // DFT
  dft(floatImg, dftImg, DFT_COMPLEX_OUTPUT);
  shiftDFT(dftImg);

  // 노치 필터 생성 (여기서 주파수 위치를 직접 지정)
  Mat notchFilter = Mat::ones(dftImg.size(), CV_32FC2);

  // 예시: 수직 방향 밴딩 (중앙에서 좌우로 떨어진 곳을 필터링)
  vector<Point> notchCenters = {
      Point(dftImg.cols / 2, dftImg.rows / 2 + 20),
      Point(dftImg.cols / 2, dftImg.rows / 2 - 20)};

  // 노치 필터 적용
  for (Point p : notchCenters)
  {
    circle(notchFilter, p, radius, Scalar::all(0), -1);
  }

  mulSpectrums(dftImg, notchFilter, dftImg, 0);

  // 역변환
  shiftDFT(dftImg);
  Mat idftImg;
  idft(dftImg, idftImg, DFT_REAL_OUTPUT | DFT_SCALE);

  // 정규화 후 반환
  normalize(idftImg, idftImg, 0, 255, NORM_MINMAX);
  idftImg.convertTo(idftImg, CV_8U);

  return idftImg;
}

int main()
{
  Mat img = imread("../image/test_30_f1.jpg", IMREAD_COLOR);
  if (img.empty())
  {
    cerr << "이미지 로드 실패!" << endl;
    return -1;
  }

  vector<Mat> channels;
  split(img, channels);

  // 각 채널 밴딩 제거 처리
  for (int i = 0; i < channels.size(); i++)
  {
    channels[i] = removeBandingDFT(channels[i], 17); // radius는 실험적으로 조정 가능
  }

  Mat resultImg;
  merge(channels, resultImg);

  // 결과 표시 및 저장
  imshow("Original", img);
  imshow("Banding Removed", resultImg);

  waitKey(0);
  return 0;
}
