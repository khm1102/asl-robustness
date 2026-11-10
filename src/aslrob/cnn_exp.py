"""Keras 3 (torch backend) CNN-plain / CNN-aug 학습과 변형 조건별 평가.

KERAS_BACKEND=torch python -m aslrob.cnn_exp --variant plain|aug --seeds 3 [--epochs 25]
출력: results/cnn_<variant>_<split>.json, results/cnn_preds_<variant>_<split>.npz
"""
import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import argparse
import json
import time

import cv2
import keras
import numpy as np
from keras import layers

from . import config as C
from . import perturb


def build_model(n_classes: int, size: int = C.SIZE_CNN, widths=(16, 32, 64, 96)):
    inp = keras.Input((size, size, 3))
    x = layers.Rescaling(1 / 255.0)(inp)
    for f in widths:
        x = layers.Conv2D(f, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    m = keras.Model(inp, out)
    m.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return m


def augment_batch(xb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """학습용 증강(96×96). 회전은 의도적으로 제외 → rot15는 '미학습 변형'."""
    out = np.empty_like(xb)
    for i, im in enumerate(xb):
        r = rng.random()
        if r < 0.25:  # 어두움 + 노이즈
            f = rng.uniform(0.1, 1.0)
            im = perturb.low_light(im, f, rng, noise_sigma=rng.uniform(2, 8))
        if rng.random() < 0.3:  # 잡동사니
            im = perturb.clutter(im, rng, n=int(rng.integers(5, 12)))
        if rng.random() < 0.5:  # 이동/크기
            im = perturb.affine(
                im, rng.uniform(0.6, 1.35), rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2)
            )
        out[i] = im
    return out


class Seq(keras.utils.PyDataset):
    def __init__(self, x, y, bs, aug, seed):
        super().__init__()
        self.x, self.y, self.bs, self.aug = x, y, bs, aug
        self.rng = np.random.default_rng(seed)
        self.order = np.arange(len(x))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.x) / self.bs))

    def __getitem__(self, i):
        j = self.order[i * self.bs:(i + 1) * self.bs]
        xb = self.x[j]
        if self.aug:
            xb = augment_batch(xb, self.rng)
        return xb.astype(np.float32), self.y[j]

    def on_epoch_end(self):
        self.rng.shuffle(self.order)


def rs(a: np.ndarray, size: int) -> np.ndarray:
    if a.shape[1] == size:
        return a
    return np.stack([cv2.resize(i, (size, size), interpolation=cv2.INTER_AREA) for i in a])


def main(variant, seeds, epochs, split_kind, conds, size):
    sp = np.load(C.PROC / f"split_{split_kind}.npy")
    y = np.load(C.PROC / "y.npy")
    x = rs(np.load(C.PROC / "x96.npy"), size)
    illus = np.load(C.PROC / "illus.npy")
    tr, va = np.where(sp == 0)[0], np.where(sp == 1)[0]
    res, preds = {}, {}
    for seed in range(seeds):
        keras.utils.set_random_seed(seed)
        model = build_model(len(C.LETTERS), size)
        t = time.time()
        hist = model.fit(
            Seq(x[tr], y[tr], 64, variant == "aug", seed),
            validation_data=(x[va].astype(np.float32), y[va]),
            epochs=epochs,
            verbose=2,
            callbacks=[keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True)],
        )
        train_s = time.time() - t
        for cond in conds:
            z = np.load(C.PROC / "pert_test" / f"{cond}.npz")
            idx = z["idx"]
            p = model.predict(rs(z["x96"], size).astype(np.float32), verbose=0).argmax(1)
            ph = ~illus[idx]
            yy = y[idx]
            r = {
                "photo": float((p == yy)[ph].mean()),
                "all": float((p == yy).mean()),
                "epochs": len(hist.history["loss"]),
                "train_s": train_s,
            }
            res.setdefault(cond, []).append(r)
            preds[f"{cond}|s{seed}"] = p
        print("seed", seed, {c: round(res[c][-1]["photo"], 3) for c in conds}, flush=True)
    summary = {
        c: {k: float(np.mean([r[k] for r in v])) for k in ("photo", "all", "epochs", "train_s")}
        | {"photo_sd": float(np.std([r["photo"] for r in v]))}
        for c, v in res.items()
    }
    meta = {"variant": variant, "seeds": seeds, "epochs_max": epochs, "split_kind": split_kind,
            "params": int(model.count_params()), "size": size}
    (C.RESULTS / f"cnn_{variant}_{split_kind}.json").write_text(
        json.dumps({"meta": meta, "summary": summary, "runs": res}, indent=1, ensure_ascii=False)
    )
    np.savez_compressed(C.RESULTS / f"cnn_preds_{variant}_{split_kind}.npz", **preds)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="plain", choices=["plain", "aug"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--split_kind", default="group")
    ap.add_argument("--conds", default=",".join(perturb.CONDITIONS))
    a = ap.parse_args()
    s = main(a.variant, a.seeds, a.epochs, a.split_kind, a.conds.split(","), a.size)
    for c, v in s.items():
        print(c, "photo %.3f±%.3f all %.3f" % (v["photo"], v["photo_sd"], v["all"]))
