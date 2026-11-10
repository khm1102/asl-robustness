"""근접 중복 군집화 (H1: 누수 방지). 외부 의존성 없이 dHash + union-find."""
import numpy as np
from PIL import Image


def dhash(img: Image.Image, n: int = 16) -> np.ndarray:
    """n×n 차분 해시 (n*n bit). 16 → 256bit 로 8x8보다 충돌이 적다."""
    g = img.convert("L").resize((n + 1, n), Image.BILINEAR)
    a = np.asarray(g, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).ravel()


def hamming_matrix(H: np.ndarray) -> np.ndarray:
    Hf = H.astype(np.float32)
    # d = |a| + |b| - 2 a·b
    s = Hf.sum(1)
    return s[:, None] + s[None, :] - 2.0 * (Hf @ Hf.T)


def cluster(H: np.ndarray, thr: int) -> np.ndarray:
    """Hamming 거리 <= thr 인 쌍을 union-find 로 병합. 군집 id 반환."""
    n = len(H)
    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    D = hamming_matrix(H)
    ii, jj = np.where(np.triu(D <= thr, k=1))
    for i, j in zip(ii, jj):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri
    roots = np.array([find(i) for i in range(n)])
    _, gid = np.unique(roots, return_inverse=True)
    return gid
