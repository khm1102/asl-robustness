"""손 관절 → 특징. 입력 lm: (n,21,3), 검출 실패는 NaN 행.

- raw   : (x,y,z) 63차원. 이동/스케일 불변성 없음 (H4 대조군)
- norm  : 손목(0) 기준 이동 + 손바닥 크기(0→9 거리)로 스케일 정규화, 63차원
- geom  : norm + 21점 쌍거리(210, 같은 스케일) + 손가락 마디 굴곡각(각 손가락 3개 ×5 = 15)
"""
import itertools

import numpy as np

# MediaPipe 손가락 체인(손목 포함)
CHAINS = [(0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12), (0, 13, 14, 15, 16), (0, 17, 18, 19, 20)]
_PAIRS = np.array(list(itertools.combinations(range(21), 2)))


def normalize(lm: np.ndarray) -> np.ndarray:
    c = lm - lm[:, :1, :]
    scale = np.linalg.norm(c[:, 9, :2], axis=1)  # 손목→중지 MCP 이미지 평면 거리
    scale = np.where(scale < 1e-6, np.nan, scale)
    return c / scale[:, None, None]


def pair_dists(nl: np.ndarray) -> np.ndarray:
    d = nl[:, _PAIRS[:, 0]] - nl[:, _PAIRS[:, 1]]
    return np.linalg.norm(d, axis=2)


def joint_angles(nl: np.ndarray) -> np.ndarray:
    out = []
    for ch in CHAINS:
        for a, b, c in zip(ch[:-2], ch[1:-1], ch[2:]):
            v1 = nl[:, a] - nl[:, b]
            v2 = nl[:, c] - nl[:, b]
            cos = (v1 * v2).sum(1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-9)
            out.append(np.arccos(np.clip(cos, -1, 1)))
    return np.stack(out, 1)


def build(lm: np.ndarray, kind: str) -> np.ndarray:
    n = len(lm)
    if kind == "raw":
        return lm.reshape(n, -1)
    nl = normalize(lm)
    if kind == "norm":
        return nl.reshape(n, -1)
    if kind == "geom":
        return np.concatenate([nl.reshape(n, -1), pair_dists(nl), joint_angles(nl)], 1)
    raise ValueError(kind)


KINDS = ("raw", "norm", "geom")
