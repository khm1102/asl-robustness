"""웹 데모 추론 엔진: MediaPipe → RF, CNN(plain/aug). 실험과 같은 전처리/모델을 사용.

입력은 256×256 RGB uint8(정사각 크롭 후). 검출 실패는 RF 예측 None(= e2e 오답).
"""
import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import threading
import time

import cv2
import joblib
import numpy as np

from .. import config as C
from .. import features as F
from .. import landmarks, perturb

MODELS = C.ROOT / "models"


class Engine:
    def __init__(self, conf: float = 0.3):
        self.lock = threading.Lock()
        self.hands = landmarks.make_hands(conf)
        self.conf = conf
        self.rf = None
        self.cnn = {}
        self.error = {}
        p = MODELS / "rf_geom.joblib"
        if p.exists():
            self.rf = joblib.load(p)
        else:
            self.error["rf"] = "models/rf_geom.joblib 없음 (python -m aslrob.train_final)"
        import keras

        for v in ("plain", "aug"):
            p = MODELS / f"cnn_{v}.keras"
            if p.exists():
                self.cnn[v] = keras.saving.load_model(p, compile=False)
            else:
                self.error[f"cnn_{v}"] = f"models/cnn_{v}.keras 없음"
        self._keras = keras

    def status(self):
        return {
            "rf": self.rf is not None,
            "cnn_plain": "plain" in self.cnn,
            "cnn_aug": "aug" in self.cnn,
            "errors": self.error,
            "conf": self.conf,
        }

    def detect(self, rgb256: np.ndarray):
        r = self.hands.process(np.ascontiguousarray(rgb256))
        if not r.multi_hand_landmarks:
            return None
        pts = r.multi_hand_landmarks[0].landmark
        return np.array([(p.x, p.y, p.z) for p in pts], np.float32)

    @staticmethod
    def _top(proba, k=3):
        o = np.argsort(proba)[::-1][:k]
        return [{"letter": C.LETTERS[i], "p": float(proba[i])} for i in o]

    def predict(self, rgb256: np.ndarray, cond: str = "clean", rng=None):
        rng = rng or np.random.default_rng(0)
        out = {"cond": cond}
        with self.lock:
            t0 = time.perf_counter()
            if cond != "clean":
                rgb256 = perturb.CONDITIONS[cond](rgb256, rng)
            out["t_pert"] = (time.perf_counter() - t0) * 1000
            t1 = time.perf_counter()
            lm = self.detect(rgb256)
            out["t_mp"] = (time.perf_counter() - t1) * 1000
            out["det"] = lm is not None
            out["lm"] = lm[:, :2].round(4).tolist() if lm is not None else None
            t2 = time.perf_counter()
            if self.rf is not None:
                if lm is None:
                    out["rf"] = None
                else:
                    pr = self.rf.predict_proba(F.build(lm[None], "geom"))[0]
                    out["rf"] = {"top": self._top(pr)}
            out["t_rf"] = (time.perf_counter() - t2) * 1000
            t3 = time.perf_counter()
            if self.cnn:
                x = cv2.resize(rgb256, (64, 64), interpolation=cv2.INTER_AREA)[None].astype(np.float32)
                for v, m in self.cnn.items():
                    pr = self._keras.ops.convert_to_numpy(m(x, training=False))[0]
                    out[f"cnn_{v}"] = {"top": self._top(pr)}
            out["t_cnn"] = (time.perf_counter() - t3) * 1000
        return out, rgb256
