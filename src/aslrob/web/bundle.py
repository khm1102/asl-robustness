"""웹 데모가 쓰는 최소 데이터 번들 생성 (Docker 이미지에 넣기 위해 용량을 줄임).

python -m aslrob.web.bundle   → web_data/
  test256.npy   (840,256,256,3) 그룹 분할 test 원본 이미지
  meta.npz      y(정답), illus(손 그림 여부)
  cond_<name>.npz  det, lm  (변형별 MediaPipe 결과, 실험에서 저장한 값)
"""
import numpy as np

from .. import config as C

OUT = C.ROOT / "web_data"


def main():
    OUT.mkdir(exist_ok=True)
    y = np.load(C.PROC / "y.npy")
    illus = np.load(C.PROC / "illus.npy")
    idx = np.load(C.PROC / "pert_test" / "clean.npz")["idx"]
    np.save(OUT / "test256.npy", np.load(C.PROC / "x256.npy", mmap_mode="r")[idx])
    np.savez(OUT / "meta.npz", y=y[idx], illus=illus[idx])
    for f in sorted((C.PROC / "pert_test").glob("*.npz")):
        z = np.load(f)
        np.savez_compressed(OUT / f"cond_{f.stem}.npz", det=z["det"], lm=z["lm"])
    print("bundle:", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
