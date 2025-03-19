#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

// 250318
// DFT 결과의 사분면을 상호 교환(Shift)하여 0주파수를 중앙으로 이동시키는 함수
// (src는 CV_32FC2 형태여야 함: 복소수 2채널)
void shiftDFT(Mat &src)
{
  // 크기가 짝수인 경우를 위해 (x, y) 모서리 길이 구하기
  int cx = src.cols / 2;
  int cy = src.rows / 2;

  // 왼쪽 상단, 오른쪽 상단, 왼쪽 하단, 오른쪽 하단 사분면을 정의
  Mat q0(src, Rect(0, 0, cx, cy));   // Top-Left
  Mat q1(src, Rect(cx, 0, cx, cy));  // Top-Right
  Mat q2(src, Rect(0, cy, cx, cy));  // Bottom-Left
  Mat q3(src, Rect(cx, cy, cx, cy)); // Bottom-Right

  // 사분면 끼리 교환
  Mat tmp;
  // Top-Left <-> Bottom-Right
  q0.copyTo(tmp);
  q3.copyTo(q0);
  tmp.copyTo(q3);

  // Top-Right <-> Bottom-Left
  q1.copyTo(tmp);
  q2.copyTo(q1);
  tmp.copyTo(q2);
}

int main()
{
  // 1) 그레이스케일 영상 로드
  Mat img = imread("../image/test_30_f1.jpg", IMREAD_GRAYSCALE);
  if (img.empty())
  {
    cerr << "이미지를 로드할 수 없습니다." << endl;
    return -1;
  }

  // 2) float 형으로 변환 (푸리에 변환을 위해)
  Mat floatImg;
  img.convertTo(floatImg, CV_32F);

  // 3) DFT 수행 -> 복소수 2채널 형태
  Mat dftImg;
  dft(floatImg, dftImg, DFT_COMPLEX_OUTPUT);

  // 4) 0주파수를 중앙으로 이동(shift)
  shiftDFT(dftImg);

  // 5) 스펙트럼 계산(시각화용) -----------------------------------
  //    (복소수 -> 진폭 -> 로그 -> 정규화)
  {
    Mat planes[2];
    split(dftImg, planes); // real: planes[0], imag: planes[1]

    Mat magImg;
    magnitude(planes[0], planes[1], magImg);
    magImg += Scalar::all(1); // log(0) 방지
    log(magImg, magImg);      // 로그 스케일 변환

    normalize(magImg, magImg, 0, 1, NORM_MINMAX);

    imshow("Gray Image", img);
    imshow("Magnitude Spectrum (Shifted)", magImg);
  }

  // 6) 저주파 통과 필터(로패스) 생성 -----------------------------
  //    중앙( cols/2, rows/2 )을 기준으로 반지름 30 짜리 원 생성
  Mat filter = Mat::zeros(dftImg.size(), CV_32FC2);
  circle(filter,
         Point(filter.cols / 2, filter.rows / 2),
         30,             // 원하는 차단 반경
         Scalar::all(1), // 1 -> 통과
         -1);            // 내부 채우기

  // 7) 필터 적용
  Mat filteredDftImg;
  mulSpectrums(dftImg, filter, filteredDftImg, 0);

  // 8) 다시 shiftDFT()로 원위치 복원 (중앙 -> 모서리)
  shiftDFT(filteredDftImg);

  // 9) 역 DFT 수행 (실수 채널만 얻음)
  Mat idftImg;
  idft(filteredDftImg, idftImg, DFT_REAL_OUTPUT | DFT_SCALE);

  // 10) 결과 시각화
  normalize(idftImg, idftImg, 0, 1, NORM_MINMAX);

  imshow("Filtered Image (Low-pass)", idftImg);
  waitKey(0);
  return 0;
}
