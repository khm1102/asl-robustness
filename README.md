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
