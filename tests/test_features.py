import numpy as np
from aslrob import features as F


def _lm(seed=0):
    return np.random.default_rng(seed).random((4, 21, 3)).astype(np.float32)


def test_norm_translation_scale_invariant():
    lm = _lm()
    t = lm * 2.5 + np.array([0.3, -0.1, 0.0], np.float32)
    a, b = F.build(lm, "norm"), F.build(t, "norm")
    # z는 스케일은 맞지만 이동 평행: 손목 기준이라 동일해야 함 (z 이동 0)
    assert np.allclose(a, b, atol=1e-4)


def test_raw_not_invariant():
    lm = _lm()
    assert not np.allclose(F.build(lm, "raw"), F.build(lm + 0.2, "raw"))


def test_geom_dims():
    assert F.build(_lm(), "geom").shape == (4, 63 + 210 + 15)


def test_nan_propagates():
    lm = _lm()
    lm[1] = np.nan
    X = F.build(lm, "geom")
    assert np.isnan(X[1]).all() and not np.isnan(X[0]).any()
