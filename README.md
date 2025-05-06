# 프로젝트: 이미지의 주파수 분석 및 밴딩 제거

## 개요

이 프로젝트는 **디지털 이미지에서 주파수 도메인을 분석하고, 주기적인 밴딩(Banding) 현상을 자동으로 감지 및 제거하는 시스템**을 구축하는 것을 목표로 합니다. 이를 위해 \*\*푸리에 변환(DFT, Discrete Fourier Transform)\*\*을 활용하여 이미지의 주파수 성분을 분석하고, \*\*노치 필터(Notch Filter)\*\*를 적용하여 특정 주파수를 제거하는 방식을 사용합니다.

## 결과
![Before](https://github.com/orothy579/midbar_prj2/blob/main/image1.jpg)
![After](https://github.com/orothy579/midbar_prj2/blob/main/image2.png)


## 주요 기능

1. **이미지 주파수 분석**

   - OpenCV를 사용하여 2D 푸리에 변환(DFT) 수행
   - 주파수 영역을 시각화하여 특정 패턴을 분석
   - 중앙 열(Center Column) 프로파일을 추출하여 특정 주파수 피크 감지

2. **밴딩(Banding) 현상 감지**

   - `scipy.signal.find_peaks`를 이용하여 특정 주파수에서 강한 에너지를 갖는 부분(피크) 자동 탐색
   - DC 성분(중앙)에서 일정 거리(±k)에 있는 주요 피크를 분석하여 밴딩 여부 판단

3. **노치 필터(Notch Filter) 적용**

   - 감지된 밴딩 주파수를 제거하는 노치 필터 생성 및 적용
   - 특정 주파수 영역을 차단하여 밴딩을 줄이고 이미지 품질 개선

4. **결과 시각화**

   - 주파수 도메인에서의 주요 성분을 플롯
   - 변환 전/후 이미지 비교
   - 적용된 필터 및 주파수 스펙트럼 분석 결과 시각화

## 사용 기술

- **프로그래밍 언어**: Python, C++
- **라이브러리**: OpenCV, NumPy, SciPy, Matplotlib
- **알고리즘**: 2D 푸리에 변환(DFT), 노치 필터링, 피크 탐지(find\_peaks)

## 실행 방법

### 1. 환경 설정

Python 환경에서 다음 라이브러리를 설치합니다.

```sh
pip install opencv-python numpy scipy matplotlib
```

### 2. 코드 실행

```sh
python main.py --input image.jpg
```

### 3. 결과 확인

- 원본 이미지와 변환된 이미지 비교
- 주파수 분석 그래프 확인

## 향후 개선 사항

- **다양한 노이즈 제거 기법 적용** (가우시안 블러, 웨이블릿 변환 등)
- **실시간 스트림(비디오)에서의 밴딩 감지 및 제거**
- **딥러닝 기반 주파수 특징 분석 모델 적용**
