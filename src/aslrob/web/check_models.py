"""웹 데모용 저장 모델이 실험 결과와 같은 수준인지 확인 (시드 0 재학습 모델).

KERAS_BACKEND=torch python -m aslrob.web.check_models   → results/web_model_check.json
"""
import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import json

import joblib
import keras
import numpy as np

from .. import config as C
from .. import features as F
from .. import perturb
from ..cnn_exp import rs

MODELS = C.ROOT / "models"


def main():
    y = np.load(C.PROC / "y.npy")
    illus = np.load(C.PROC / "illus.npy")
    rf = joblib.load(MODELS / "rf_geom.joblib")
    cnn = {v: keras.saving.load_model(MODELS / f"cnn_{v}.keras", compile=False) for v in ("plain", "aug")}
    ref = {v: json.loads((C.RESULTS / f"cnn_{v}_group.json").read_text()) for v in ("plain", "aug")}
    out = {}
    for cond in perturb.CONDITIONS:
        z = np.load(C.PROC / "pert_test" / f"{cond}.npz")
        idx, det, lm = z["idx"], z["det"], z["lm"]
        ph = ~illus[idx]
        yy = y[idx]
        pred = np.full(len(idx), -1)
        pred[det] = rf.predict(F.build(lm[det], "geom"))
        row = {"rf_geom_e2e": float(((pred == yy) & det)[ph].mean())}
        for v, m in cnn.items():
            p = m.predict(rs(z["x96"], 64).astype(np.float32), verbose=0).argmax(1)
            row[f"cnn_{v}"] = float((p == yy)[ph].mean())
            s = ref[v]["summary"][cond]
            row[f"cnn_{v}_exp_mean"] = s["photo"]
            row[f"cnn_{v}_exp_sd"] = s["photo_sd"]
        out[cond] = row
        print(cond, {k: round(v, 3) for k, v in row.items()}, flush=True)
    (C.RESULTS / "web_model_check.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
