"""parquet → 선택 알파벳 필터 → 96/256 uint8 캐시 + 중복 군집 + 분할.

python -m aslrob.prepare [--thr 20]
"""
import argparse
import io
import json

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from . import config as C
from .dedup import cluster, dhash


def main(thr: int, seed: int = 0):
    C.PROC.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(C.RAW_PARQUET)
    keep = [C.ALL_LETTERS.index(l) for l in C.LETTERS]
    df = df[df.label.isin(keep)].reset_index(drop=True)
    y = df.label.map({k: i for i, k in enumerate(keep)}).to_numpy().astype(np.int64)
    n = len(df)
    x96 = np.zeros((n, C.SIZE_CNN, C.SIZE_CNN, 3), np.uint8)
    x256 = np.zeros((n, C.SIZE_MP, C.SIZE_MP, 3), np.uint8)
    H = np.zeros((n, 256), bool)
    sizes = []
    for i, rec in enumerate(df.image):
        im = Image.open(io.BytesIO(rec["bytes"])).convert("RGB")
        sizes.append(im.size)
        x96[i] = np.asarray(im.resize((C.SIZE_CNN,) * 2, Image.BILINEAR))
        x256[i] = np.asarray(im.resize((C.SIZE_MP,) * 2, Image.BILINEAR))
        H[i] = dhash(im)
    g = cluster(H, thr)

    # 분할: 그룹 5-fold → fold0=test(≈20%), fold1=val(≈20%), 나머지=train
    sgk = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
    fold = np.zeros(n, int)
    for k, (_, te) in enumerate(sgk.split(np.zeros(n), y, g)):
        fold[te] = k
    split_group = np.where(fold == 0, 2, np.where(fold == 1, 1, 0))  # 0 train 1 val 2 test

    # 비교용 무작위 분할(누수 포함, H1)
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    rf = np.zeros(n, int)
    for k, (_, te) in enumerate(skf.split(np.zeros(n), y)):
        rf[te] = k
    split_random = np.where(rf == 0, 2, np.where(rf == 1, 1, 0))

    np.save(C.PROC / "x96.npy", x96)
    np.save(C.PROC / "x256.npy", x256)
    np.save(C.PROC / "y.npy", y)
    np.save(C.PROC / "group.npy", g)
    np.save(C.PROC / "split_group.npy", split_group)
    np.save(C.PROC / "split_random.npy", split_random)

    sz = np.array(sizes)
    sizes_count = np.bincount(g)
    # 무작위 분할에서 test 이미지 중 train에 같은 군집이 있는 비율 = 누수 지표
    tr_groups = set(g[split_random == 0])
    leak = float(np.mean([gi in tr_groups for gi in g[split_random == 2]]))
    stats = {
        "n_images": int(n), "letters": C.LETTERS, "dhash_thr": thr,
        "n_clusters": int(g.max() + 1), "n_clusters_size_gt1": int((sizes_count > 1).sum()),
        "images_in_dup_clusters": int(sizes_count[g][sizes_count[g] > 1].size),
        "largest_cluster": int(sizes_count.max()),
        "random_split_test_leak_frac": leak,
        "group_split_sizes": [int((split_group == k).sum()) for k in range(3)],
        "random_split_sizes": [int((split_random == k).sum()) for k in range(3)],
        "img_w_median": int(np.median(sz[:, 0])), "img_h_median": int(np.median(sz[:, 1])),
    }
    (C.RESULTS / "prepare_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--thr", type=int, default=20, help="dHash(256bit) hamming 임계값")
    main(ap.parse_args().thr)
