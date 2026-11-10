# 0001. 환경: uv + Python 3.12 + Keras 3(torch 백엔드)

- 상태: 채택
- 맥락: 사용자가 uv 기반 Python 환경을 요청, Python 3.14도 허용. 기획안은 TensorFlow/Keras.
- 관찰 (직접 확인):
  - `tensorflow-cpu 2.21.0`은 cp312/cp313 wheel만 있음 → 3.14에서 설치 불가 (uv 오류 메시지).
  - 3.12에 설치해도 `import tensorflow`가 `Illegal instruction`으로 종료. `/proc/cpuinfo` flags에 `avx`, `avx2` 없음(sse4_2까지). TF 공식 wheel은 AVX 필요.
  - mediapipe는 TF 없이 임포트 정상.
  - mediapipe가 `opencv-contrib-python`(libGL 필요)을 끌어와 cv2 임포트 실패 → `tool.uv.override-dependencies`로 제거하고 headless만 사용.
- 선택지: (a) TF 소스 빌드 — 2코어/3GB에서 비현실적. (b) PyTorch 직접 — Keras API 포기. (c) Keras 3 + torch 백엔드.
- 결정: (c). 기획안의 Keras 모델 정의를 유지하면서 AVX 없는 CPU에서 동작. torch CPU wheel은 pytorch-cpu 인덱스 사용.
- 결과: `keras 3.15.1 / backend=torch / torch 2.15.0+cpu / mediapipe 1.2.2 / opencv 5.0.0` 임포트 확인.
- 주의: `KERAS_BACKEND=torch`를 항상 설정해야 함.
