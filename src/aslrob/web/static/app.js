"use strict";
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const COLORS = {rf_geom: "#3ecf8e", rf_norm: "#2aa774", rf_raw: "#1a7a56", cnn_plain: "#ff8a5b", cnn_aug: "#f5b84a"};
const CNAME = {clean: "clean", "dark_0.5": "어두움 0.5", "dark_0.25": "어두움 0.25", "dark_0.12": "어두움 0.12",
  clutter: "잡동사니 배경", shift: "손 이동", scale_small: "손 작게", scale_large: "손 크게", rot15: "회전 15°"};
const ENVNAME = {bright: "밝은 환경", dark: "어두운 환경", clutter: "복잡한 배경", moved: "손 위치/크기 변화"};
const EDGES = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];
let ST = null;

// ---------------------------------------------------------------- 탭
$$("nav button").forEach(b => b.onclick = () => {
  $$("nav button").forEach(x => x.classList.toggle("active", x === b));
  $$(".tab").forEach(t => t.classList.toggle("active", t.id === "tab-" + b.dataset.tab));
  if (b.dataset.tab === "results") loadResults();
  if (b.dataset.tab === "testset") loadTs(true);
  if (b.dataset.tab === "mine") loadCapCount();
  if (b.dataset.tab === "docs") loadDoc();
});

async function jget(u) { const r = await fetch(u); if (!r.ok) throw new Error(u + " " + r.status); return r.json(); }
function opt(sel, items, fmt) { sel.innerHTML = items.map(v => `<option value="${v}">${fmt ? fmt(v) : v}</option>`).join(""); }

// ---------------------------------------------------------------- 초기화
(async function init() {
  ST = await jget("/api/status");
  const miss = Object.entries(ST.errors);
  $("#status").textContent = miss.length ? "모델 누락: " + miss.map(e => e[1]).join("; ")
    : `RF ✓ · CNN-plain ✓ · CNN-aug ✓ · MediaPipe conf=${ST.conf}`;
  if (miss.length) $("#status").style.color = "var(--warn)";
  opt($("#cond"), ST.conditions, c => CNAME[c] || c);
  opt($("#tsCond"), ST.conditions, c => CNAME[c] || c);
  opt($("#capLabel"), ST.letters);
  opt($("#capEnv"), ST.envs, e => ENVNAME[e] || e);
  $("#tsLetter").innerHTML += ST.letters.map(l => `<option>${l}</option>`).join("");
  $("#letterList").textContent = ST.letters.join(" ");
  opt($("#wordSel"), WORDS);
  drawPractice();
  ["tsCond", "tsKind", "tsLetter", "tsShow", "tsLm"].forEach(id => $("#" + id).onchange = () => loadTs(true));
  $("#cond").onchange = () => { if (ws && ws.readyState === 1) ws.send(JSON.stringify({cond: $("#cond").value})); };
})();

// ---------------------------------------------------------------- 따라하기(연습)
const WORDS = ["FLY", "BUY", "WAY", "DAY", "LAB", "LID", "BAD", "FAIL", "WILD", "DIAL", "AVID", "FLAW", "DUAL", "VIA"];
const HINT = {
  A: "주먹을 쥐고 엄지를 주먹 옆에 붙입니다.", B: "손가락을 모두 펴서 붙이고 엄지는 손바닥 앞으로 접습니다.",
  D: "검지만 세우고, 나머지 손가락 끝을 엄지와 맞대어 동그랗게 만듭니다.", F: "엄지와 검지 끝을 맞대어 원을 만들고 나머지 세 손가락은 폅니다.",
  I: "주먹을 쥐고 새끼손가락만 세웁니다.", L: "검지는 위로, 엄지는 옆으로 펴서 L 모양을 만듭니다.",
  U: "검지와 중지를 붙여서 세우고 나머지는 접습니다.", V: "검지와 중지를 벌려 V 모양을 만듭니다.",
  W: "검지·중지·약지 세 손가락을 벌려 세우고 엄지로 새끼손가락을 누릅니다.", Y: "엄지와 새끼손가락만 펴고 나머지는 접습니다."};
const HOLD = 5;  // 판정 모델이 연속 5프레임 맞히면 성공
const MODELS3 = [["rf", "RF"], ["cnn_plain", "CNN-plain"], ["cnn_aug", "CNN-aug"]];
let P = {mode: "free", word: [], pos: 0, streak: 0, stats: {}, finished: false};

function startPractice() {
  if (!ST) return;  // /api/status 로딩 전
  const mode = $("#mode").value; P.mode = mode; P.pos = 0; P.streak = 0; P.finished = false; P.stats = {};
  $("#wordSelWrap").style.display = mode === "word" ? "" : "none";
  $("#judgeWrap").style.display = mode === "free" ? "none" : "";
  $("#skipBtn").style.display = mode === "free" ? "none" : "";
  $("#refPanel").style.display = mode === "free" ? "none" : "";
  $(".grid3a").classList.toggle("practice", mode !== "free");
  $("#practiceResult").innerHTML = "";
  if (mode === "word") P.word = $("#wordSel").value.split("");
  else if (mode === "letter") P.word = [P.word.length === 1 && P.word[0] ? P.word[0] : ST.letters[Math.floor(Math.random() * ST.letters.length)]];
  else P.word = [];
  drawPractice();
}
function drawPractice() {
  if (!ST) return;
  const cur = P.word[P.pos];
  $("#wordRow").innerHTML = P.mode === "free" ? "" : (P.mode === "word" ? `<div class="w">${P.word.join("")}</div>` : "") +
    P.word.map((l, i) => `<div class="l ${i < P.pos || P.finished ? "done" : i === P.pos ? "cur" : ""}">${l}</div>`).join("");
  $("#refStrip").innerHTML = ST.letters.map(l => `<div class="rc ${l === cur ? "cur" : ""}" data-l="${l}"><img src="/static/ref/${l}.png"><span>${l}</span></div>`).join("");
  $$(".rc").forEach(el => el.onclick = () => { $("#mode").value = "letter"; P.mode = "letter"; P.word = [el.dataset.l]; P.pos = 0; P.finished = false; P.stats = {}; P.streak = 0; $("#practiceResult").innerHTML = ""; $("#judgeWrap").style.display = ""; $("#skipBtn").style.display = ""; $("#refPanel").style.display = ""; $(".grid3a").classList.add("practice"); $("#wordSelWrap").style.display = "none"; drawPractice(); });
  if (P.mode === "free" || P.finished || !cur) { $("#practiceMsg").textContent = P.finished ? "완료!" : "도감에서 글자를 눌러 연습하거나, 모드를 바꿔 보세요."; if (P.finished) $("#refPanel").style.display = "none"; return; }
  $("#refImg").src = `/static/ref/${cur}.png`; $("#refLetter").textContent = cur;
  $("#refHint").textContent = HINT[cur];
  $("#practiceMsg").textContent = `지금 만들 글자: ${cur}  (${P.pos + 1}/${P.word.length}) — 그림처럼 손 모양을 만들고 ${HOLD}프레임 유지하면 성공`;
  $("#holdFill").style.width = "0";
}
function practiceStep(r) {
  if (P.mode === "free" || P.finished) return;
  const cur = P.word[P.pos]; if (!cur) return;
  const st = P.stats[cur] || (P.stats[cur] = {n: 0, rf: 0, cnn_plain: 0, cnn_aug: 0, det: 0});
  st.n++; st.det += r.det ? 1 : 0;
  MODELS3.forEach(([k]) => { const p = r[k]; if (p && p.top[0].letter === cur) st[k]++; });
  const j = r[$("#judge").value];
  P.streak = j && j.top[0].letter === cur ? P.streak + 1 : 0;
  $("#holdFill").style.width = Math.min(100, P.streak / HOLD * 100) + "%";
  if (P.streak >= HOLD) {
    P.streak = 0; P.pos++;
    if (P.pos >= P.word.length) { P.finished = true; showPracticeResult(); }
    drawPractice();
  }
}
function showPracticeResult() {
  const pct = (a, n) => n ? (a / n * 100).toFixed(0) + "%" : "–";
  const rows = Object.entries(P.stats);
  $("#practiceResult").innerHTML = `<div class="panel"><h3 class="pr-ok">완료! 연습 중 모델별 정답 프레임 비율</h3><table><tr><th>글자</th><th>프레임</th><th>손 검출</th><th>RF</th><th>CNN-plain</th><th>CNN-aug</th></tr>` +
    rows.map(([l, s]) => `<tr><td>${l}</td><td>${s.n}</td><td>${pct(s.det, s.n)}</td><td>${pct(s.rf, s.n)}</td><td>${pct(s.cnn_plain, s.n)}</td><td>${pct(s.cnn_aug, s.n)}</td></tr>`).join("") +
    `</table><p class="muted small">성공 직전까지 걸린 프레임이 대상이라 모델이 맞힐 때까지 시도한 오답 프레임이 많을수록 비율이 낮습니다. 같은 사람·환경 몇 번의 시도일 뿐이라 통계가 아닙니다.</p></div>`;
}
$("#mode").onchange = startPractice; $("#wordSel").onchange = startPractice;
$("#skipBtn").onclick = () => { if (P.mode === "letter") { P.word = [ST.letters[Math.floor(Math.random() * ST.letters.length)]]; } else { P.pos = Math.min(P.pos + 1, P.word.length); if (P.pos >= P.word.length) { P.finished = true; showPracticeResult(); } } P.streak = 0; drawPractice(); };

// ---------------------------------------------------------------- 실시간 웹캠
let stream = null, ws = null, running = false, busy = false, lastSent = null, fpsT = [];
const video = document.createElement("video"); video.playsInline = true; video.muted = true;
const send = document.createElement("canvas"); send.width = send.height = 256;
const view = $("#view"), seen = $("#seen");

$("#camBtn").onclick = async () => {
  if (running) return stop();
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    alert("이 주소에서는 웹캠을 쓸 수 없습니다. 브라우저는 https 또는 localhost 에서만 카메라를 허용합니다."); return;
  }
  try { stream = await navigator.mediaDevices.getUserMedia({video: {width: 640, height: 480}, audio: false}); }
  catch (e) { alert("카메라를 열 수 없습니다: " + e.message); return; }
  video.srcObject = stream; await video.play();
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws/live`);
  ws.onopen = () => { ws.send(JSON.stringify({cond: $("#cond").value})); running = true; $("#camBtn").textContent = "카메라 중지"; viewLoop(); loop(); };
  ws.onmessage = ev => { busy = false; render(JSON.parse(ev.data)); loop(); };
  ws.onclose = () => stop();
};
function stop() {
  running = false; busy = false;
  if (stream) stream.getTracks().forEach(t => t.stop());
  if (ws) { try { ws.close(); } catch (e) {} }
  $("#camBtn").textContent = "카메라 시작";
}
function cropSquare(ctx, w, flip) {
  const vw = video.videoWidth, vh = video.videoHeight, s = Math.min(vw, vh);
  const sx = (vw - s) / 2, sy = (vh - s) / 2;
  ctx.save(); if (flip) { ctx.translate(w, 0); ctx.scale(-1, 1); }
  ctx.drawImage(video, sx, sy, s, s, 0, 0, w, w); ctx.restore();
}
function drawView() {
  const ctx = view.getContext("2d");
  cropSquare(ctx, view.width, $("#mirrorView").checked);
  ctx.strokeStyle = "#3ecf8e"; ctx.lineWidth = 3; ctx.strokeRect(2, 2, view.width - 4, view.height - 4);
}
function viewLoop() { if (!running) return; drawView(); requestAnimationFrame(viewLoop); }
function loop() {  // 응답을 받은 뒤에만 다음 프레임 전송 (큐가 쌓이지 않음)
  if (!running || busy || !ws || ws.readyState !== 1 || !video.videoWidth) return;
  cropSquare(send.getContext("2d"), 256, $("#flipSend").checked);
  busy = true;
  send.toBlob(b => { lastSent = b; ws.send(b); }, "image/jpeg", 0.85);
}

function render(r) {
  const now = performance.now(); fpsT.push(now); fpsT = fpsT.filter(t => now - t < 2000);
  $("#fps").textContent = (fpsT.length / 2).toFixed(1) + " FPS";
  // 모델 입력 + 관절
  const ctx = seen.getContext("2d");
  const done = () => {
    if (r.lm) {
      const W = seen.width, p = r.lm.map(([x, y]) => [x * W, y * W]);
      ctx.strokeStyle = "#3ecf8e"; ctx.lineWidth = 2;
      EDGES.forEach(([a, b]) => { ctx.beginPath(); ctx.moveTo(...p[a]); ctx.lineTo(...p[b]); ctx.stroke(); });
      ctx.fillStyle = "#ff4d4d"; p.forEach(([x, y]) => { ctx.beginPath(); ctx.arc(x, y, 3.5, 0, 7); ctx.fill(); });
    } else { ctx.fillStyle = "rgba(255,0,0,.85)"; ctx.font = "bold 22px sans-serif"; ctx.fillText("손 검출 실패", 12, 32); }
  };
  const src = r.img ? "data:image/jpeg;base64," + r.img : (lastSent ? URL.createObjectURL(lastSent) : null);
  if (src) { const im = new Image(); im.onload = () => { ctx.drawImage(im, 0, 0, seen.width, seen.height); done(); if (!r.img) URL.revokeObjectURL(src); }; im.src = src; }
  const b = $("#detBadge"); b.textContent = r.det ? "손 검출됨" : "검출 실패"; b.className = "badge " + (r.det ? "ok" : "bad");
  fillPred("#pred-rf", r.rf, r.det ? null : "MediaPipe가 손을 못 찾음 → 인식 실패(실험에서는 오답 처리)");
  fillPred("#pred-cnn_plain", r.cnn_plain); fillPred("#pred-cnn_aug", r.cnn_aug);
  practiceStep(r);
  $("#timing").textContent = `전체 ${r.t_total.toFixed(0)} · MediaPipe ${r.t_mp.toFixed(0)} · RF ${r.t_rf.toFixed(0)} · CNN×2 ${r.t_cnn.toFixed(0)} · 변형 ${r.t_pert.toFixed(0)}`;
}
function fillPred(sel, p, note) {
  const el = $(sel), big = $(".big", el), bars = $(".bars", el);
  $(".note", el).textContent = note || "";
  if (!p) { big.textContent = "인식 실패"; big.className = "big none"; bars.innerHTML = ""; return; }
  big.className = "big"; big.textContent = p.top[0].letter;
  bars.innerHTML = p.top.map(t => `<div class="row"><span>${t.letter}</span><div class="track"><div class="bar" style="width:${(t.p * 100).toFixed(0)}%"></div></div><span style="width:34px">${(t.p * 100).toFixed(0)}%</span></div>`).join("");
}

$("#capBtn").onclick = async () => {
  if (!running || !lastSent) { $("#capMsg").textContent = "카메라를 먼저 시작하세요"; return; }
  const fd = new FormData(); fd.append("file", lastSent, "f.jpg");
  fd.append("label", $("#capLabel").value); fd.append("env", $("#capEnv").value);
  const r = await fetch("/api/capture", {method: "POST", body: fd});
  $("#capMsg").textContent = r.ok ? "저장됨 " + JSON.stringify((await r.json()).count) : "저장 실패 " + r.status;
};

// ---------------------------------------------------------------- 테스트셋
let tsOff = 0;
async function loadTs(reset) {
  if (reset) { tsOff = 0; $("#tsGrid").innerHTML = ""; }
  const q = new URLSearchParams({cond: $("#tsCond").value, kind: $("#tsKind").value, show: $("#tsShow").value, letter: $("#tsLetter").value, offset: tsOff, limit: 40});
  const d = await jget("/api/testset?" + q);
  const pct = v => v == null ? "–" : (v * 100).toFixed(1) + "%";
  $("#tsSummary").innerHTML = `<div class="chip">해당 이미지 <b>${d.total}</b>장</div>` +
    `<div class="chip">RF 정확도 <b>${pct(d.acc.rf)}</b></div><div class="chip">CNN-plain <b>${pct(d.acc.plain)}</b></div>` +
    `<div class="chip">CNN-aug <b>${pct(d.acc.aug)}</b></div><div class="chip">손 검출률 <b>${pct(d.acc.det)}</b></div>`;
  const lm = $("#tsLm").checked ? 1 : 0, cond = $("#tsCond").value;
  const tag = (pred, t) => pred == null ? `<span class="tag na">실패</span>` : `<span class="tag ${pred === t ? "ok" : "bad"}">${pred}</span>`;
  $("#tsGrid").insertAdjacentHTML("beforeend", d.items.map(it => `<div class="card"><img loading="lazy" src="/api/testset/img?cond=${cond}&i=${it.i}&lm=${lm}&size=192">` +
    `<div class="meta"><span class="t">정답 ${it.true}</span>${it.illus ? " <span class='muted'>(그림)</span>" : ""}<br>RF ${tag(it.rf, it.true)} plain ${tag(it.plain, it.true)} aug ${tag(it.aug, it.true)}</div></div>`).join(""));
  tsOff += d.items.length; $("#tsMore").style.display = tsOff < d.total ? "" : "none";
}
$("#tsMore").onclick = () => loadTs(false);

// ---------------------------------------------------------------- 결과
let RES = null;
$("#metric").onchange = () => RES && drawChart();
async function loadResults() {
  if (!RES) RES = await jget("/api/results");
  drawChart(); drawExtra();
  $("#resMeta").textContent = `평가: 그룹 분할 test 사진 ${RES.n_photo ?? "?"}장, CNN 3시드 평균, RF 3시드 평균`;
  $("#figs").innerHTML = RES.figs.map(f => `<div><div class="muted small">${f}</div><img src="/figs/${f}"></div>`).join("");
}
function val(m, c, metric) {
  if (metric === "e2e") return m.e2e[c];
  if (metric === "det_only") return m.det_only ? m.det_only[c] : null;
  if (metric === "det_rate") return m.det_rate ? m.det_rate[c] : null;
  if (metric === "drop") return (m.e2e.clean - m.e2e[c]) * 100;
}
function drawChart() {
  const metric = $("#metric").value, pctMetric = metric !== "drop";
  const rows = RES.conditions.filter(c => metric !== "drop" || c !== "clean");
  const mx = metric === "drop" ? 100 : 1;
  $("#chart").innerHTML = `<div class="legend">${RES.models.map(m => `<span><i style="background:${COLORS[m.id]}"></i>${m.name}</span>`).join("")}</div>` +
    rows.map(c => `<div class="cg"><div class="lab">${CNAME[c] || c}</div><div class="bg">${RES.models.map(m => {
      const v = val(m, c, metric);
      if (v == null) return `<div class="br"><div class="v">–</div></div>`;
      const sd = m.sd && metric === "e2e" ? ` ±${(m.sd[c] * 100).toFixed(0)}` : "";
      return `<div class="br"><div class="fill" style="width:${Math.max(0, v / mx * 100) * 0.8}%;background:${COLORS[m.id]}"></div><div class="v">${pctMetric ? (v * 100).toFixed(1) + "%" : v.toFixed(1) + "p"}${sd}</div></div>`;
    }).join("")}</div></div>`).join("");
  const hd = `<tr><th>모델</th>${RES.conditions.map(c => `<th>${CNAME[c] || c}</th>`).join("")}</tr>`;
  $("#resTable").innerHTML = `<table>${hd}${RES.models.map(m => `<tr><td>${m.name}</td>${RES.conditions.map(c => { const v = val(m, c, metric); return `<td>${v == null ? "–" : pctMetric ? (v * 100).toFixed(1) : v.toFixed(1)}</td>`; }).join("")}</tr>`).join("")}</table>`;
}
function drawExtra() {
  let h = "";
  if (RES.h1) {
    const g = RES.h1.group, r = RES.h1.random, mean = a => a.reduce((x, y) => x + y, 0) / a.length;
    if (g && r) h += `<div class="panel"><b>분할 누수(H1)</b> — 그룹 분할 vs 무작위 분할 (clean 사진 정확도)<table><tr><th>분할</th><th>RF e2e</th><th>CNN-plain(3시드)</th></tr>` +
      `<tr><td>그룹(중복 제거)</td><td>${(mean(g.rf_e2e) * 100).toFixed(1)}%</td><td>${(mean(g.cnn) * 100).toFixed(1)}%</td></tr>` +
      `<tr><td>무작위</td><td>${(mean(r.rf_e2e) * 100).toFixed(1)}%</td><td>${(mean(r.cnn) * 100).toFixed(1)}%</td></tr></table></div>`;
  }
  if (RES.conf_sens) {
    const cs = RES.conf_sens, cs_c = ["clean", "dark_0.5", "dark_0.25", "dark_0.12", "clutter"];
    h += `<div class="panel"><b>MediaPipe 검출 임계값 민감도</b> — RF e2e / 검출률<table><tr><th>조건</th><th>conf 0.1</th><th>conf 0.3(기본)</th><th>conf 0.5</th></tr>` +
      cs_c.map(c => `<tr><td>${CNAME[c]}</td>${[0.1, 0.3, 0.5].map(k => { const x = cs[c + "|" + k]; return `<td>${(x.e2e * 100).toFixed(1)}% / ${(x.det_rate * 100).toFixed(0)}%</td>`; }).join("")}</tr>`).join("") + `</table></div>`;
  }
  if (RES.prepare) h += `<div class="panel"><b>데이터</b> 알파벳 ${RES.prepare.letters.join(" ")} · ${RES.prepare.n_images}장 · 근접 중복 군집 ${RES.prepare.n_clusters_size_gt1}개 (해시 임계 ${RES.prepare.dhash_thr})</div>`;
  $("#extra").innerHTML = h;
}

// ---------------------------------------------------------------- 내 촬영
async function loadCapCount() { $("#capCount").textContent = "저장된 사진: " + JSON.stringify(await jget("/api/captures")); }
$("#evalBtn").onclick = async () => {
  $("#evalOut").textContent = "채점 중…";
  const d = await (await fetch("/api/captures/eval", {method: "POST"})).json();
  const rows = Object.entries(d);
  if (!rows.length) { $("#evalOut").innerHTML = "<p class='muted'>저장된 사진이 없습니다. 실시간 탭에서 저장하세요.</p>"; return; }
  const p = (a, n) => n ? (a / n * 100).toFixed(1) + "% (" + a + "/" + n + ")" : "–";
  $("#evalOut").innerHTML = `<table><tr><th>환경</th><th>장수</th><th>손 검출</th><th>RF(e2e)</th><th>CNN-plain</th><th>CNN-aug</th></tr>` +
    rows.map(([e, r]) => `<tr><td>${ENVNAME[e] || e}</td><td>${r.n}</td><td>${p(r.det, r.n)}</td><td>${p(r.rf, r.n)}</td><td>${p(r.plain, r.n)}</td><td>${p(r.aug, r.n)}</td></tr>`).join("") + "</table>";
};

// ---------------------------------------------------------------- 문서
let docsLoaded = false;
async function loadDoc() {
  if (!docsLoaded) { const l = await jget("/api/docs"); $("#docSel").innerHTML = l.map(d => `<option value="${d.name}">${d.title}</option>`).join(""); $("#docSel").onchange = loadDoc; docsLoaded = true; }
  $("#docBody").innerHTML = await (await fetch("/api/doc/" + $("#docSel").value)).text();
}
