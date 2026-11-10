import numpy as np
from aslrob import perturb as P


def _img():
    return np.random.default_rng(0).integers(0, 256, (256, 256, 3), dtype=np.uint8)


def test_all_conditions_shape_dtype():
    x = np.stack([_img(), _img()])
    for name in P.CONDITIONS:
        y = P.apply(name, x, seed=1)
        assert y.shape == x.shape and y.dtype == np.uint8


def test_dark_is_darker():
    x = np.stack([_img()])
    assert P.apply("dark_0.25", x).mean() < x.mean() * 0.4


def test_deterministic():
    x = np.stack([_img()])
    assert (P.apply("clutter", x, seed=3) == P.apply("clutter", x, seed=3)).all()


def test_clutter_keeps_center():
    x = np.stack([_img()])
    y = P.apply("clutter", x, seed=2)
    # 중앙(30%~70%)은 변형 안 됨
    assert (x[:, 80:170, 80:170] == y[:, 80:170, 80:170]).all()


def test_to_cnn():
    assert P.to_cnn(np.stack([_img()])).shape == (1, 96, 96, 3)
