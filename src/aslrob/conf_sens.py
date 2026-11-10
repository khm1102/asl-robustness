"""MediaPipe min_detection_confidence 민감도 (H5 보강).

python -m aslrob.conf_sens
같은 변형 이미지(testsets와 동일 시드)에 conf=0.1/0.3/0.5 로 랜드마크를 다시 뽑아
RF(geom, 학습은 clean conf=0.3 랜드마크) e2e / det-only / 검출률을 사진 부분집합에서 계산.
출력: results/conf_sens.json
"""
import json

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from . import config as C
from . import features as F
from . import landmarks, perturb

CONDS = ["clean", "dark_0.5", "dark_0.25", "dark_0.12", "clutter"]
CONFS = [0.1, 0.3, 0.5]


def main():
    y = np.load(C.PROC / "y.npy")
    illus = np.load(C.PROC / "illus.npy")
    d = np.load(C.PROC / "lm_clean.npz")
    sp = np.load(C.PROC / "split_group.npy")
    tr = np.where((sp == 0) & d["det"])[0]
    rf = RandomForestClassifier(n_estimators=300, n_jobs=1, random_state=0).fit(
        F.build(d["lm"][tr], "geom"), y[tr]
    )
    te = np.where(sp == 2)[0]
    x = np.load(C.PROC / "x256.npy", mmap_mode="r")[te]
    ph = ~illus[te]
    names = list(perturb.CONDITIONS)
    out = {}
    for cond in CONDS:
        xp = perturb.apply(cond, x, seed=1000 + names.index(cond))
        for conf in CONFS:
            lm, det, _ = landmarks.extract(xp, conf=conf)
            pred = np.full(len(te), -1)
            if det.any():
                pred[det] = rf.predict(F.build(lm[det], "geom"))
            ok = pred == y[te]
            m = ph & det
            out[f"{cond}|{conf}"] = {
                "det_rate": float(det[ph].mean()),
                "e2e": float((ok & det)[ph].mean()),
                "det_only": float(ok[m].mean()) if m.any() else None,
            }
            print(cond, conf, out[f"{cond}|{conf}"], flush=True)
    (C.RESULTS / "conf_sens.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
