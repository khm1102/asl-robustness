import pytest

from aslrob import config as C

pytestmark = pytest.mark.skipif(
    not (C.ROOT / "web_data" / "meta.npz").exists() or not (C.RESULTS / "cnn_plain_group.json").exists(),
    reason="web_data/results 필요 (python -m aslrob.web.bundle)",
)


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from aslrob.web.app import app

    return TestClient(app)


def test_results_consistent_with_json(client):
    d = client.get("/api/results").json()
    ids = {m["id"] for m in d["models"]}
    assert {"rf_geom", "cnn_plain", "cnn_aug"} <= ids
    plain = next(m for m in d["models"] if m["id"] == "cnn_plain")
    import json

    j = json.loads((C.RESULTS / "cnn_plain_group.json").read_text())
    assert plain["e2e"]["clean"] == pytest.approx(j["summary"]["clean"]["photo"])


def test_testset_filter_and_image(client):
    d = client.get("/api/testset", params={"cond": "dark_0.5", "show": "rf_wrong", "limit": 5}).json()
    assert d["total"] > 0 and all(it["rf"] != it["true"] for it in d["items"])
    r = client.get("/api/testset/img", params={"cond": "dark_0.5", "i": d["items"][0]["i"], "lm": 1})
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"


def test_doc_path_validation(client):
    assert client.get("/api/doc/REPORT.md").status_code == 200
    assert client.get("/api/doc/..%2Fpyproject.toml").status_code in (400, 404)
