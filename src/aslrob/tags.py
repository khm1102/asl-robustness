"""이미지 메타 태그: 삽화(흑백 스케치) 여부. 데이터에 출처 정보가 없어 색 통계로 추정."""
import cv2
import numpy as np


def illustration_tag(x256: np.ndarray, sat_thr: float = 12.0, white_thr: float = 0.5) -> np.ndarray:
    """평균 채도(HSV S, 0~255)가 낮고 거의 흰 픽셀이 많으면 삽화로 본다."""
    out = np.zeros(len(x256), bool)
    for i, im in enumerate(x256):
        hsv = cv2.cvtColor(np.ascontiguousarray(im), cv2.COLOR_RGB2HSV)
        white = (np.asarray(im).min(axis=2) > 200).mean()
        out[i] = hsv[..., 1].mean() < sat_thr and white > white_thr
    return out
