"""합성 환경 변화(테스트용). 입력/출력: (H,W,3) uint8 RGB, 256×256 기준.

CNN용 96×96은 변형 후 리사이즈한다 → 두 모델이 '같은 변형 이미지'를 본다.
한계(decisions/0003): 실제 촬영 환경 변화가 아닌 시뮬레이션이며 손 마스크가 없다.
"""
import cv2
import numpy as np


def brightness(img, factor):
    """선형 감광 (센서 노출 감소 근사). factor<1 어둡게."""
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def gamma(img, g):
    """g>1 어둡고 대비 낮은 그림자 느낌."""
    return (255.0 * (img.astype(np.float32) / 255.0) ** g).astype(np.uint8)


def low_light(img, factor, rng, noise_sigma=6.0):
    """어두운 조명: 감광 + 센서 노이즈 (어두울수록 노이즈가 눈에 띔)."""
    out = brightness(img, factor).astype(np.float32)
    out += rng.normal(0, noise_sigma, out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def clutter(img, rng, n=10, border=0.22):
    """복잡한 배경 근사: 가장자리(border 비율)에만 색/무늬 도형을 여러 개 배치.
    중앙은 보존 → 손을 덮지 않지만, 손이 가장자리에 닿은 이미지는 일부 가려질 수 있음."""
    out = img.copy()
    h, w = out.shape[:2]
    for _ in range(n):
        color = tuple(int(c) for c in rng.integers(0, 256, 3))
        side = rng.integers(0, 4)
        t = int(border * min(h, w))
        if side == 0:
            cx, cy = rng.integers(0, w), rng.integers(0, t)
        elif side == 1:
            cx, cy = rng.integers(0, w), rng.integers(h - t, h)
        elif side == 2:
            cx, cy = rng.integers(0, t), rng.integers(0, h)
        else:
            cx, cy = rng.integers(w - t, w), rng.integers(0, h)
        r = int(rng.integers(8, 34))
        if rng.random() < 0.5:
            cv2.circle(out, (int(cx), int(cy)), r, color, -1)
        else:
            cv2.rectangle(out, (int(cx - r), int(cy - r)), (int(cx + r), int(cy + r)), color, -1)
    c0, c1 = int(h * border), int(h * (1 - border))
    out[c0:c1, c0:c1] = img[c0:c1, c0:c1]  # 도형이 중앙으로 넘어온 부분 복원
    # 약한 텍스처 잡음
    tex = rng.normal(0, 12, out.shape).astype(np.float32)
    mask = np.ones((h, w, 1), np.float32)
    mask[c0:c1, c0:c1] = 0
    return np.clip(out.astype(np.float32) + tex * mask, 0, 255).astype(np.uint8)


def affine(img, scale=1.0, dx=0.0, dy=0.0, rot=0.0):
    """scale: 0.6=손이 작아짐, 1.3=확대. dx,dy: 이미지 크기 비율 이동. rot: 도. 경계는 reflect."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rot, scale)
    M[0, 2] += dx * w
    M[1, 2] += dy * h
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


# 이름 → (함수, 파라미터). 각 변형은 시드로 재현.
CONDITIONS = {
    "clean": lambda im, rng: im,
    "dark_0.5": lambda im, rng: low_light(im, 0.5, rng),
    "dark_0.25": lambda im, rng: low_light(im, 0.25, rng),
    "dark_0.12": lambda im, rng: low_light(im, 0.12, rng),
    "clutter": lambda im, rng: clutter(im, rng),
    "shift": lambda im, rng: affine(
        im, 1.0, rng.choice([-0.2, 0.2]), rng.choice([-0.2, 0.2])
    ),
    "scale_small": lambda im, rng: affine(im, 0.6),
    "scale_large": lambda im, rng: affine(im, 1.35),
    "rot15": lambda im, rng: affine(im, 1.0, rot=float(rng.choice([-15, 15]))),
}


def apply(name, images, seed=0):
    rng = np.random.default_rng(seed)
    f = CONDITIONS[name]
    return np.stack([f(im, rng) for im in images])


def to_cnn(images256, size=96):
    return np.stack([cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA) for im in images256])
