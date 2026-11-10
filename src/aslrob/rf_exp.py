"""랜드마크 + RF/DT 학습 및 변형 조건별 평가.

python -m aslrob.rf_exp [--seeds 3] [--split_kind group|random]
출력: results/rf_results.json, results/rf_preds.npz
"""
import argparse
import json

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from . import config as C
from . import features as F
from . import perturb


def load_train(split_kind: str):
    sp = np.load(C.PROC / f"split_{split_kind}.npy")
    y = np.load(C.PROC / "y.npy")
    d = np.load(C.PROC / "lm_clean.npz")
    tr = (sp == 0) & d["det"]
    return d["lm"][tr], y[tr], tr.sum(), int((sp == 0).sum())


def make_model(name: str, seed: int):
    if name == "rf":
        return RandomForestClassifier(n_estimators=300, n_jobs=1, random_state=seed)
    if name == "dt":
        return DecisionTreeClassifier(random_state=seed)
    raise ValueError(name)


def evaluate(pred, det, y, mask):
    """end-to-end(검출 실패=오답)와 detected-only 정확도."""
    m = mask
    e2e = float(((pred == y) & det)[m].mean())
    dm = m & det
    return {
        "e2e": e2e,
        "det_only": float((pred == y)[dm].mean()) if dm.any() else None,
        "det_rate": float(det[m].mean()),
        "n": int(m.sum()),
    }


def main(seeds: int, split_kind: str, conds):
    y_all = np.load(C.PROC / "y.npy")
    illus = np.load(C.PROC / "illus.npy")
    lm_tr, y_tr, n_tr_det, n_tr = load_train(split_kind)
    res, preds = {}, {}
    for kind in F.KINDS:
        Xtr = F.build(lm_tr, kind)
        for mname in ("rf", "dt"):
            for seed in range(seeds):
                model = make_model(mname, seed).fit(Xtr, y_tr)
                for cond in conds:
                    z = np.load(C.PROC / "pert_test" / f"{cond}.npz")
                    idx, det, lm = z["idx"], z["det"], z["lm"]
                    y = y_all[idx]
                    pred = np.full(len(idx), -1)
                    if det.any():
                        pred[det] = model.predict(F.build(lm[det], kind))
                    ph = ~illus[idx]
                    key = f"{kind}|{mname}|{cond}"
                    r = {"photo": evaluate(pred, det, y, ph), "all": evaluate(pred, det, y, np.ones_like(ph))}
                    res.setdefault(key, []).append(r)
                    preds[f"{key}|s{seed}"] = pred
    summary = {}
    for key, runs in res.items():
        summary[key] = {
            sub: {
                k: (float(np.mean([r[sub][k] for r in runs if r[sub][k] is not None]))
                     if any(r[sub][k] is not None for r in runs) else None)
                for k in ("e2e", "det_only", "det_rate", "n")
            }
            for sub in ("photo", "all")
        }
    meta = {"train_n": n_tr, "train_det_n": int(n_tr_det), "split_kind": split_kind, "seeds": seeds}
    (C.RESULTS / f"rf_results_{split_kind}.json").write_text(
        json.dumps({"meta": meta, "summary": summary}, indent=1, ensure_ascii=False)
    )
    np.savez_compressed(C.RESULTS / f"rf_preds_{split_kind}.npz", **preds)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--split_kind", default="group")
    ap.add_argument("--conds", default=",".join(perturb.CONDITIONS))
    a = ap.parse_args()
    s = main(a.seeds, a.split_kind, a.conds.split(","))
    for k, v in s.items():
        print(k, "photo e2e %.3f det_only %.3f det %.3f" % (
            v["photo"]["e2e"], v["photo"]["det_only"] or -1, v["photo"]["det_rate"]))
