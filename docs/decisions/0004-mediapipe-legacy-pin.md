# 0004. MediaPipe 0.10.14(legacy solutions API)로 고정

- 상태: 채택
- 관찰 (직접 실행):
  1. `mediapipe 1.2.2`의 tasks API는 `libEGL.so.1`이 없어 로드 실패 → micromamba로 libegl 설치해 해결 가능했지만,
  2. 이어서 `FATAL ERROR: This binary was compiled with avx enabled, but this feature is not available on this processor` 로 **SIGILL**. 이 서버 CPU(QEMU vCPU, AVX 없음)에서는 1.2.2 네이티브 라이브러리가 실행 불가.
  3. `mediapipe 0.10.14` + `mp.solutions.hands`는 같은 CPU에서 정상 동작.
- 결정: `mediapipe==0.10.14` 고정. API는 `mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=…)`.
- 영향: protobuf 4.25.9로 다운그레이드(uv가 해결). 별도 모델 파일 불필요.
- 시험(깨끗한 이미지 200장 균등 샘플, conf=0.3): 검출 156/200 = 78%, 약 0.063 s/장 → 4,198장 약 4.4분.
- 결과 해석 주의: 검출률 78%는 이 데이터셋에 손 삽화·잘린 손이 섞여 있기 때문일 가능성(미확인, 검출 실패 샘플 시각 점검 예정).
