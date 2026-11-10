"""H1: 무작위 분할 vs 그룹 분할 clean 정확도(사진 부분집합) 비교.

KERAS_BACKEND=torch python -m aslrob.h1_split --seeds 3 --epochs 40
각 분할의 자체 test(원본 clean 이미지; 변형 없음)로 평가. RF(geom, 검출 성공만 학습, e2e) + CNN-plain.
출력: results/h1_split.json
"""
import os

os.environ.setdefault("KERAS_BACKEND", "torch")
import argparse
import json

import keras
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from . import config as C
from . import features as F
from .cnn_exp import Seq, build_model, rs


def main(seeds, epochs, size=64, only=None):
    y = np.load(C.PROC / "y.npy")
    illus = np.load(C.PROC / "illus.npy")
    d = np.load(C.PROC / "lm_clean.npz")
    lm, det = d["lm"], d["det"]
    x = rs(np.load(C.PROC / "x96.npy"), size)
    f = C.RESULTS / "h1_split.json"
    out = json.loads(f.read_text()) if f.exists() else {}  # 이어서 실행
    for sk in ("group", "random"):
        if only and sk != only:
            continue
        sp = np.load(C.PROC / f"split_{sk}.npy")
        tr, va, te = (np.where(sp == k)[0] for k in range(3))
        ph = ~illus[te]
        r = {"n_test": int(len(te)), "n_photo": int(ph.sum()), "rf_e2e": [], "rf_det_only": [], "cnn": []}
        trd = tr[det[tr]]
        Xtr = F.build(lm[trd], "geom")
        for s in range(seeds):
            m = RandomForestClassifier(n_estimators=300, n_jobs=1, random_state=s).fit(Xtr, y[trd])
            pred = np.full(len(te), -1)
            dt = det[te]
            pred[dt] = m.predict(F.build(lm[te][dt], "geom"))
            r["rf_e2e"].append(float(((pred == y[te]) & dt)[ph].mean()))
            r["rf_det_only"].append(float((pred == y[te])[ph & dt].mean()))
        print(sk, "rf", r["rf_e2e"], flush=True)
        for s in range(seeds):
            keras.utils.set_random_seed(s)
            model = build_model(len(C.LETTERS), size)
            model.fit(Seq(x[tr], y[tr], 64, False, s),
                      validation_data=(x[va].astype(np.float32), y[va]), epochs=epochs, verbose=0,
                      callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)])
            p = model.predict(x[te].astype(np.float32), verbose=0).argmax(1)
            r["cnn"].append(float((p == y[te])[ph].mean()))
            print(sk, "cnn seed", s, r["cnn"][-1], flush=True)
        out[sk] = r
        f.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--only", default=None, choices=["group", "random"])
    a = ap.parse_args()
    main(a.seeds, a.epochs, only=a.only)
