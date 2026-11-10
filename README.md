# asl-robustness

ASL 알파벳 인식: **CNN(이미지 픽셀)** vs **MediaPipe 손 관절 21점 + Random Forest** —
촬영 환경(조명/배경/손 위치) 변화에 어느 쪽이 더 강한가?

머신러닝 기말 프로젝트. 기획안: `docs/proposal.md`.

## 환경
- uv + Python 3.12 (`uv sync`)
- Keras 3 (torch backend, `KERAS_BACKEND=torch`) — 이 서버 CPU에 AVX가 없어 TensorFlow wheel이 실행되지 않음
- MediaPipe, scikit-learn, OpenCV(headless)

```bash
uv sync
KERAS_BACKEND=torch uv run python -c "import keras; print(keras.backend.backend())"
```

## 결과 요약
최종 요약은 [`docs/REPORT.md`](docs/REPORT.md), 가설 대장은 [`docs/hypotheses.md`](docs/hypotheses.md), 실험 기록은 `docs/experiments/`.

| 조건 (사진 정확도) | RF geom | CNN-plain | CNN-aug |
|---|---|---|---|
| clean | 0.942 | 0.577 | 0.477 |
| dark_0.5 | 0.699 | 0.110 | 0.318 |
| clutter | 0.826 | 0.199 | 0.374 |
| shift | 0.859 | 0.429 | 0.423 |

합성 환경 변화 기준이며 직접 촬영 검증은 아직 없다. 한계는 REPORT.md 참조.
