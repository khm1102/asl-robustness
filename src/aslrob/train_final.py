"""웹 데모용 최종 모델 학습·저장 (실험과 동일 설정, 그룹 분할 train, 시드 0).

KERAS_BACKEND=torch python -m aslrob.train_final [--epochs 40]
출력: models/rf_geom.joblib, models/cnn_plain.keras, models/cnn_aug.keras, models/meta.json
(models/ 는 git 제외 → 재생성 가능)
"""
import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import argparse
import json

import joblib
import keras
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from . import config as C
from . import features as F
from .cnn_exp import Seq, build_model, rs

MODELS = C.ROOT / "models"


def main(epochs: int, size: int = 64, only: str | None = None):
    MODELS.mkdir(exist_ok=True)
    sp = np.load(C.PROC / "split_group.npy")
    y = np.load(C.PROC / "y.npy")
    tr, va = np.where(sp == 0)[0], np.where(sp == 1)[0]

    if only in (None, "rf"):
        d = np.load(C.PROC / "lm_clean.npz")
        trd = tr[d["det"][tr]]
        rf = RandomForestClassifier(n_estimators=300, n_jobs=1, random_state=0)
        rf.fit(F.build(d["lm"][trd], "geom"), y[trd])
        joblib.dump(rf, MODELS / "rf_geom.joblib", compress=3)
        print("rf saved", flush=True)

    x = rs(np.load(C.PROC / "x96.npy"), size)
    for variant in ("plain", "aug"):
        if only not in (None, variant):
            continue
        keras.utils.set_random_seed(0)
        m = build_model(len(C.LETTERS), size)
        m.fit(
            Seq(x[tr], y[tr], 64, variant == "aug", 0),
            validation_data=(x[va].astype(np.float32), y[va]),
            epochs=epochs,
            verbose=2,
            callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
        )
        m.save(MODELS / f"cnn_{variant}.keras")
        print(variant, "saved", flush=True)
    (MODELS / "meta.json").write_text(
        json.dumps({"letters": C.LETTERS, "cnn_size": size, "seed": 0, "split": "group"}, indent=1)
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--only", choices=["rf", "plain", "aug"])
    a = ap.parse_args()
    main(a.epochs, only=a.only)
