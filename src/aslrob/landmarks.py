"""MediaPipe Hands(legacy solutions API)로 손 관절 21점 추출.

검출 실패는 NaN + det=False 로 남긴다 (기획안: 실패도 인식 실패로 집계).
python -m aslrob.landmarks   # 깨끗한 x256 전체 캐시
"""
import time

import mediapipe as mp
import numpy as np

from . import config as C


def make_hands(conf: float = 0.3):
    return mp.solutions.hands.Hands(
        static_image_mode=True, max_num_hands=1, min_detection_confidence=conf
    )


def extract(images: np.ndarray, conf: float = 0.3, progress: int = 0):
    """images: (n,H,W,3) uint8 RGB → lm (n,21,3) float32 [x,y,z 이미지 비율좌표], det (n,) bool,
    hand (n,) int8 (-1 실패, 0 Left, 1 Right; MediaPipe 기준)."""
    n = len(images)
    lm = np.full((n, 21, 3), np.nan, np.float32)
    det = np.zeros(n, bool)
    hand = np.full(n, -1, np.int8)
    hands = make_hands(conf)
    t = time.time()
    for i in range(n):
        r = hands.process(np.ascontiguousarray(images[i]))
        if r.multi_hand_landmarks:
            pts = r.multi_hand_landmarks[0].landmark
            lm[i] = [(p.x, p.y, p.z) for p in pts]
            det[i] = True
            hand[i] = 1 if r.multi_handedness[0].classification[0].label == "Right" else 0
        if progress and (i + 1) % progress == 0:
            print(f"{i + 1}/{n} {time.time() - t:.0f}s det={det[: i + 1].mean():.3f}", flush=True)
    hands.close()
    return lm, det, hand


if __name__ == "__main__":
    x = np.load(C.PROC / "x256.npy", mmap_mode="r")
    lm, det, hand = extract(x, conf=0.3, progress=500)
    np.savez_compressed(C.PROC / "lm_clean.npz", lm=lm, det=det, hand=hand)
    y = np.load(C.PROC / "y.npy")
    print("detect rate", det.mean())
    for k, l in enumerate(C.LETTERS):
        print(l, round(float(det[y == k].mean()), 3))
