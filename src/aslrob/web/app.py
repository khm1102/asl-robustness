"""FastAPI 웹 데모: 웹캠 실시간 인식 + 실험 결과 + 테스트셋 육안 확인.

uvicorn aslrob.web.app:app --host 0.0.0.0 --port 8080 [--ssl-keyfile ... --ssl-certfile ...]
(브라우저 웹캠은 https 또는 localhost 에서만 허용된다)
"""
import asyncio
import base64
import json
import re
import time
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

import cv2
import markdown
import numpy as np
from fastapi import FastAPI, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .. import config as C
from .. import perturb

STATIC = Path(__file__).parent / "static"
CAPTURES = C.DATA / "captures"
DOCS = C.ROOT / "docs"
ENVS = ["bright", "dark", "clutter", "moved"]  # 직접 촬영 환경 태그

app = FastAPI(title="ASL robustness demo")
_engine = None


def engine():
    global _engine
    if _engine is None:
        from .infer import Engine

        _engine = Engine()
    return _engine


# ---------------------------------------------------------------- 정적 페이지
@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC / "index.html").read_text(encoding="utf-8")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
if (C.RESULTS / "figs").exists():
    app.mount("/figs", StaticFiles(directory=C.RESULTS / "figs"), name="figs")


@app.get("/api/status")
def status():
    e = engine()
    return {
        **e.status(),
        "letters": C.LETTERS,
        "conditions": list(perturb.CONDITIONS),
        "envs": ENVS,
    }


# ---------------------------------------------------------------- 웹캠 WebSocket
@app.websocket("/ws/live")
async def ws_live(ws: WebSocket):
    """클라이언트: text(JSON {"cond": ..}) 또는 binary(JPEG 256×256). 서버: JSON 결과.
    변형이 켜지면 모델이 실제로 본 이미지를 base64 JPEG 로 함께 돌려준다."""
    await ws.accept()
    e = engine()
    cond = "clean"
    rng = np.random.default_rng(int(time.time()))
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if msg.get("text"):
                cond = json.loads(msg["text"]).get("cond", "clean")
                if cond not in perturb.CONDITIONS:
                    cond = "clean"
                continue
            data = msg.get("bytes")
            if not data:
                continue
            bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            rgb = cv2.cvtColor(cv2.resize(bgr, (256, 256)), cv2.COLOR_BGR2RGB)
            t0 = time.perf_counter()
            res, proc = await asyncio.to_thread(e.predict, rgb, cond, rng)
            res["t_total"] = (time.perf_counter() - t0) * 1000
            if cond != "clean":
                ok, buf = cv2.imencode(".jpg", cv2.cvtColor(proc, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
                res["img"] = base64.b64encode(buf.tobytes()).decode()
            await ws.send_text(json.dumps(res))
    except WebSocketDisconnect:
        pass


# ---------------------------------------------------------------- 실험 결과
def _load(name):
    p = C.RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


@app.get("/api/results")
def results():
    rf = _load("rf_results_group.json")
    plain = _load("cnn_plain_group.json")
    aug = _load("cnn_aug_group.json")
    conds = list(perturb.CONDITIONS)
    models = []
    if rf:
        for kind, name in (("geom", "RF · 거리·각도(geom)"), ("norm", "RF · 정규화 좌표(norm)"), ("raw", "RF · 원시 좌표(raw)")):
            s = rf["summary"]
            models.append({
                "id": f"rf_{kind}", "name": name, "family": "rf",
                "e2e": {c: s[f"{kind}|rf|{c}"]["photo"]["e2e"] for c in conds},
                "det_only": {c: s[f"{kind}|rf|{c}"]["photo"]["det_only"] for c in conds},
                "det_rate": {c: s[f"{kind}|rf|{c}"]["photo"]["det_rate"] for c in conds},
            })
    for key, name, d in (("cnn_plain", "CNN · plain", plain), ("cnn_aug", "CNN · aug", aug)):
        if d:
            s = d["summary"]
            models.append({
                "id": key, "name": name, "family": "cnn",
                "e2e": {c: s[c]["photo"] for c in conds},
                "sd": {c: s[c]["photo_sd"] for c in conds},
            })
    return {
        "conditions": conds,
        "models": models,
        "n_photo": (_load("stats.json") or {}).get("n_photo"),
        "conf_sens": _load("conf_sens.json"),
        "h1": _load("h1_split.json"),
        "prepare": _load("prepare_stats.json"),
        "figs": sorted(p.name for p in (C.RESULTS / "figs").glob("*.png")) if (C.RESULTS / "figs").exists() else [],
    }


def _md(name):
    p = DOCS / name
    if not p.exists():
        raise HTTPException(404)
    return markdown.markdown(p.read_text(encoding="utf-8"), extensions=["tables", "fenced_code"])


@app.get("/api/doc/{name}", response_class=HTMLResponse)
def doc(name: str):
    if not re.fullmatch(r"[A-Za-z0-9_\-/]+\.md", name):
        raise HTTPException(400)
    return _md(name)


@app.get("/api/docs")
def docs_list():
    out = [{"name": "REPORT.md", "title": "최종 요약"}, {"name": "hypotheses.md", "title": "가설 대장"}]
    out += [{"name": f"experiments/{p.name}", "title": p.stem} for p in sorted((DOCS / "experiments").glob("*.md"))]
    out += [{"name": f"decisions/{p.name}", "title": p.stem} for p in sorted((DOCS / "decisions").glob("*.md"))]
    return out


# ---------------------------------------------------------------- 테스트셋 둘러보기
WEB = C.ROOT / "web_data"  # python -m aslrob.web.bundle 로 생성


@lru_cache(maxsize=1)
def _ts_meta():
    m = np.load(WEB / "meta.npz")
    y, illus = m["y"], m["illus"]
    idx = np.arange(len(y))
    rf = np.load(C.RESULTS / "rf_preds_group.npz")
    pl = np.load(C.RESULTS / "cnn_preds_plain_group.npz")
    ag = np.load(C.RESULTS / "cnn_preds_aug_group.npz")
    return y, illus, idx, rf, pl, ag


@lru_cache(maxsize=9)
def _ts_cond(cond):
    y, illus, idx, rf, pl, ag = _ts_meta()
    z = np.load(WEB / f"cond_{cond}.npz")
    return {
        "det": z["det"],
        "lm": z["lm"],
        "rf": rf[f"geom|rf|{cond}|s0"],
        "plain": pl[f"{cond}|s0"],
        "aug": ag[f"{cond}|s0"],
    }


_img_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()


def _ts_images(cond):
    if cond in _img_cache:
        _img_cache.move_to_end(cond)
        return _img_cache[cond]
    x = np.load(WEB / "test256.npy", mmap_mode="r")
    k = list(perturb.CONDITIONS).index(cond)
    arr = perturb.apply(cond, x, seed=1000 + k)  # testsets.py 와 동일 → 실험에 쓴 바로 그 이미지
    _img_cache[cond] = arr
    while len(_img_cache) > 2:
        _img_cache.popitem(last=False)
    return arr


def _letter(v):
    return None if v < 0 else C.LETTERS[int(v)]


@app.get("/api/testset")
def testset(cond: str = "clean", kind: str = "photo", show: str = "all", offset: int = 0, limit: int = 40, letter: str = ""):
    if cond not in perturb.CONDITIONS:
        raise HTTPException(400, "cond")
    y, illus, idx, *_ = _ts_meta()
    d = _ts_cond(cond)
    yt = y[idx]
    sel = np.ones(len(idx), bool)
    if kind == "photo":
        sel &= ~illus[idx]
    elif kind == "illus":
        sel &= illus[idx]
    if letter in C.LETTERS:
        sel &= yt == C.LETTERS.index(letter)
    rf_ok, pl_ok, ag_ok = d["rf"] == yt, d["plain"] == yt, d["aug"] == yt
    if show == "rf_wrong":
        sel &= ~rf_ok
    elif show == "plain_wrong":
        sel &= ~pl_ok
    elif show == "aug_wrong":
        sel &= ~ag_ok
    elif show == "rf_ok_cnn_wrong":
        sel &= rf_ok & ~pl_ok
    elif show == "cnn_ok_rf_wrong":
        sel &= ~rf_ok & pl_ok
    pos = np.where(sel)[0]
    page = pos[offset: offset + limit]
    items = [{
        "i": int(i), "true": C.LETTERS[int(yt[i])], "det": bool(d["det"][i]),
        "rf": _letter(d["rf"][i]), "plain": _letter(d["plain"][i]), "aug": _letter(d["aug"][i]),
        "illus": bool(illus[idx[i]]),
    } for i in page]
    return {
        "total": int(len(pos)), "items": items,
        "acc": {
            "rf": float(rf_ok[sel].mean()) if sel.any() else None,
            "plain": float(pl_ok[sel].mean()) if sel.any() else None,
            "aug": float(ag_ok[sel].mean()) if sel.any() else None,
            "det": float(d["det"][sel].mean()) if sel.any() else None,
        },
    }


@app.get("/api/testset/img")
def testset_img(cond: str, i: int, lm: int = 0, size: int = 192):
    if cond not in perturb.CONDITIONS:
        raise HTTPException(400, "cond")
    arr = _ts_images(cond)
    if not 0 <= i < len(arr):
        raise HTTPException(404)
    im = cv2.cvtColor(arr[i], cv2.COLOR_RGB2BGR)
    if lm:
        d = _ts_cond(cond)
        if d["det"][i]:
            pts = (d["lm"][i][:, :2] * 256).astype(int)
            for a, b in HAND_EDGES:
                cv2.line(im, tuple(pts[a]), tuple(pts[b]), (0, 255, 0), 1)
            for p in pts:
                cv2.circle(im, tuple(p), 2, (0, 0, 255), -1)
        else:
            cv2.putText(im, "NO HAND", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    im = cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA if size < 256 else cv2.INTER_LINEAR)
    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return Response(buf.tobytes(), media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})


HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
              (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


# ---------------------------------------------------------------- 직접 촬영 저장/평가
@app.post("/api/capture")
async def capture(file: UploadFile, label: str = Form(...), env: str = Form(...)):
    if label not in C.LETTERS or env not in ENVS:
        raise HTTPException(400, "label/env")
    d = CAPTURES / env / label
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{int(time.time() * 1000)}.jpg"
    p.write_bytes(await file.read())
    return {"saved": str(p.relative_to(C.DATA)), "count": _cap_count()}


def _cap_count():
    return {e: len(list((CAPTURES / e).glob("*/*.jpg"))) for e in ENVS if (CAPTURES / e).exists()}


@app.get("/api/captures")
def captures():
    return _cap_count()


@app.post("/api/captures/eval")
async def captures_eval():
    """저장한 직접 촬영 사진을 세 모델로 평가 (환경 태그별). 변형은 적용하지 않는다."""
    e = engine()
    rows = {}
    for env in ENVS:
        for lp in sorted((CAPTURES / env).glob("*/*.jpg")) if (CAPTURES / env).exists() else []:
            bgr = cv2.imread(str(lp))
            if bgr is None:
                continue
            rgb = cv2.cvtColor(cv2.resize(bgr, (256, 256)), cv2.COLOR_BGR2RGB)
            res, _ = await asyncio.to_thread(e.predict, rgb, "clean")
            true = lp.parent.name
            r = rows.setdefault(env, {"n": 0, "det": 0, "rf": 0, "plain": 0, "aug": 0})
            r["n"] += 1
            r["det"] += int(res["det"])
            if res.get("rf"):
                r["rf"] += int(res["rf"]["top"][0]["letter"] == true)
            for v in ("plain", "aug"):
                if res.get(f"cnn_{v}"):
                    r[v] += int(res[f"cnn_{v}"]["top"][0]["letter"] == true)
    return rows
