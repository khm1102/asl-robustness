"""테스트 분할(그룹 분할)에 합성 변형을 적용하고 랜드마크를 추출해 캐시.
python -m aslrob.testsets [--split test|val]
"""
import argparse

import numpy as np

from . import config as C
from . import landmarks, perturb


def main(split: str, conf: float):
    sp = np.load(C.PROC / "split_group.npy")
    idx = np.where(sp == {"val": 1, "test": 2}[split])[0]
    x = np.load(C.PROC / "x256.npy", mmap_mode="r")[idx]
    out = C.PROC / f"pert_{split}"
    out.mkdir(exist_ok=True)
    for k, name in enumerate(perturb.CONDITIONS):
        f = out / f"{name}.npz"
        if f.exists():
            continue
        xp = perturb.apply(name, x, seed=1000 + k)
        lm, det, hand = landmarks.extract(xp, conf=conf)
        np.savez_compressed(
            f, idx=idx, x96=perturb.to_cnn(xp, C.SIZE_CNN), lm=lm, det=det, hand=hand
        )
        print(name, "det", round(float(det.mean()), 3), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--conf", type=float, default=0.3)
    a = ap.parse_args()
    main(a.split, a.conf)
