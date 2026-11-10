"""RF vs CNN 쌍체 비교: paired bootstrap(95% CI), McNemar, 혼동행렬.

python -m aslrob.stats
모든 비교는 같은 테스트 샘플(사진 부분집합)에서 시드 평균 정답여부(0~1)로 수행.
RF 예측 -1 = 검출 실패(e2e에서 오답). 출력: results/stats.json, results/figs/confusion_*.png
"""
import json

import numpy as np
from scipy.stats import binomtest

from . import config as C
from . import perturb

B = 2000
rng = np.random.default_rng(0)
MODELS = {
    "rf_geom": ("rf_preds_group.npz", "geom|rf|{c}|s{s}"),
    "rf_norm": ("rf_preds_group.npz", "norm|rf|{c}|s{s}"),
    "rf_raw": ("rf_preds_group.npz", "raw|rf|{c}|s{s}"),
    "cnn_plain": ("cnn_preds_plain_group.npz", "{c}|s{s}"),
    "cnn_aug": ("cnn_preds_aug_group.npz", "{c}|s{s}"),
}


def load():
    y = np.load(C.PROC / "y.npy")
    illus = np.load(C.PROC / "illus.npy")
    idx = np.load(C.PROC / "pert_test" / "clean.npz")["idx"]
    ph = ~illus[idx]
    yy = y[idx][ph]
    corr, vote, preds = {}, {}, {}
    for m, (f, pat) in MODELS.items():
        z = np.load(C.RESULTS / f)
        for c in perturb.CONDITIONS:
            P = np.stack([z[pat.format(c=c, s=s)][ph] for s in range(3)])
            ok = (P == yy[None])
            corr[(m, c)] = ok.mean(0)  # 시드 평균 정답률(샘플별)
            vote[(m, c)] = ok.sum(0) >= 2  # 다수결 정답
            preds[(m, c)] = P
    return yy, corr, vote, preds


def boot_ci(d):
    n = len(d)
    ix = rng.integers(0, n, (B, n))
    bs = d[ix].mean(1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def mcnemar(a, b):
    n01 = int((a & ~b).sum())
    n10 = int((~a & b).sum())
    p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return {"a_only": n01, "b_only": n10, "p": float(p)}


def main():
    yy, corr, vote, preds = load()
    out = {"n_photo": int(len(yy)), "pairs": {}, "drop_pairs": {}}
    pairs = [("cnn_plain", "rf_geom"), ("cnn_aug", "rf_geom"), ("cnn_aug", "cnn_plain"),
             ("rf_geom", "rf_norm"), ("rf_norm", "rf_raw")]
    for a, b in pairs:
        for c in perturb.CONDITIONS:
            m, lo, hi = boot_ci(corr[(a, c)] - corr[(b, c)])
            out["pairs"][f"{a}-{b}|{c}"] = {"diff": m, "lo": lo, "hi": hi,
                                           "mcnemar": mcnemar(vote[(a, c)], vote[(b, c)])}
        # Accuracy Drop 차이: (clean - cond) 의 모델 간 차 = (dropA - dropB)
        for c in perturb.CONDITIONS:
            if c == "clean":
                continue
            d = (corr[(a, "clean")] - corr[(a, c)]) - (corr[(b, "clean")] - corr[(b, c)])
            m, lo, hi = boot_ci(d)
            out["drop_pairs"][f"{a}-{b}|{c}"] = {"drop_diff": m, "lo": lo, "hi": hi}
    (C.RESULTS / "stats.json").write_text(json.dumps(out, indent=1))
    for k, v in out["pairs"].items():
        print(k, "diff %+.3f [%+.3f,%+.3f] mcnemar p=%.3g" % (v["diff"], v["lo"], v["hi"], v["mcnemar"]["p"]))
    for k, v in out["drop_pairs"].items():
        print("DROP", k, "%+.3f [%+.3f,%+.3f]" % (v["drop_diff"], v["lo"], v["hi"]))
    figs(yy, preds)


def figs(yy, preds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    (C.RESULTS / "figs").mkdir(exist_ok=True)
    L = C.LETTERS
    for m in ("rf_geom", "cnn_plain", "cnn_aug"):
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        for a, c in zip(ax, ("clean", "dark_0.5", "shift")):
            P = preds[(m, c)]
            cm = np.zeros((len(L), len(L) + 1))  # 마지막 열 = 검출 실패(-1)
            for p in P:  # 3 시드 합산
                for t, q in zip(yy, p):
                    cm[t, q if q >= 0 else len(L)] += 1
            cm = cm / cm.sum(1, keepdims=True)
            a.imshow(cm, vmin=0, vmax=1, cmap="Blues")
            a.set_xticks(range(len(L) + 1), L + ["x"])
            a.set_yticks(range(len(L)), L)
            a.set_title(f"{m} {c}")
            a.set_xlabel("pred (x=undetected)")
        fig.tight_layout()
        fig.savefig(C.RESULTS / "figs" / f"confusion_{m}.png", dpi=80)
        plt.close(fig)


if __name__ == "__main__":
    main()
