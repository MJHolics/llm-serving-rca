// AI 서버 관제탑 — 기록 재생·그리기. 탐지와 원인 판정 계산은 core.js(순수 함수, node 테스트 대상).
import { parseForest, parseFeatures, anomalyScore, standardize, debouncer, deviations, naiveCause, rulesV2 } from './core.js';

const SPEED = 6;              // 기록 1초를 1/6초에
const LEAD_AUTO = 14, LEAD_CLICK = 5;   // 장애를 넣기 전 평상시를 몇 초 보여 주나
const TAIL = 34;              // 넣은 뒤 몇 초까지 재생하나(가장 늦은 탐지가 27초)
const HOLD_MS = 3200;         // 판정이 나온 화면을 멈춰 두는 시간
const SPAN = 60;              // 그래프 가로 폭(초). 장애 사이 간격이 60초라 앞 장애가 화면에 안 걸린다
const ORDER = ['surge', 'longctx', 'longout', 'gpu_hog', 'cpu_hog'];
const $ = (id) => document.getElementById(id);

const FAULT = {
  surge: { name: '요청 급증', how: '동시 사용자 8명 → 64명', act: '서버를 늘립니다(스케일아웃).' },
  longctx: { name: '긴 입력 요청', how: '약 3,000토큰 입력 12개 추가', act: '입력 길이를 제한하거나 긴 문서용 서버를 따로 둡니다.' },
  longout: { name: '긴 출력 요청', how: '1,500토큰 출력 16개 추가', act: '최대 출력 길이(max_tokens) 정책을 둡니다.' },
  gpu_hog: { name: 'GPU 경합', how: '같은 GPU에서 다른 계산이 돎', act: '추론 GPU에 다른 작업이 못 들어오게 노드를 분리합니다.' },
  cpu_hog: { name: 'CPU 경합', how: '같은 서버에서 CPU 작업이 돎', act: '같은 노드의 CPU 작업을 옮깁니다.' },
  unknown: { name: '모름', how: '', act: '사람이 확인합니다.' },
};

// 그래프 8칸. key가 score면 이상 점수, 아니면 특징 이름.
const CHARTS = [
  { key: 'lat_p90', name: '응답 지연 p90', fmt: (v) => `${v.toFixed(2)}초` },
  { key: 'score', name: '이상 점수', fmt: (v) => v.toFixed(3) },
  { key: 'running', name: '처리 중인 요청', fmt: (v) => `${v.toFixed(0)}개` },
  { key: 'prompt_per_req', name: '요청당 입력 토큰', fmt: (v) => `${v.toFixed(0)}` },
  { key: 'kv_usage', name: 'KV 캐시 사용률', fmt: (v) => `${(v * 100).toFixed(0)}%` },
  { key: 'itl', name: '토큰 사이 간격', fmt: (v) => `${(v * 1000).toFixed(1)}ms` },
  { key: 'gpu_util', name: 'GPU 사용률', fmt: (v) => `${v.toFixed(0)}%` },
  { key: 'host_cpu', name: '서버 CPU', fmt: (v) => `${v.toFixed(0)}%` },
];

let meta, t, rows, scores, col;
let ep = null;                // 지금 재생 중인 구간
let timer = null, runId = 0;
const seen = [];              // 방금 본 장애(화면 아래 표)
const nextOf = {};            // 장애 종류별로 다음에 보여 줄 기록 번호

const ratio = (key, v) => v / meta.normal_mean[key];

/** 왜 그렇게 판정했나 — 이 구간에서 잰 값으로 문장을 만든다. */
function explain(inc, z, mean) {
  const x = (k) => ratio(k, mean[k]);
  const times = (k) => `${x(k).toFixed(x(k) >= 10 ? 0 : 1)}배`;
  switch (inc.fault) {
    case 'surge': return `처리 중인 요청이 평소의 ${times('running')}, 초당 완료 건수도 ${times('done_rps')}로 같이 늘었습니다. 일이 많아진 것이지 서버가 느려진 게 아닙니다.`;
    case 'longctx': return `요청 하나에 들어오는 입력이 평소의 ${times('prompt_per_req')}입니다. KV 캐시가 차서 기다리는 요청이 생깁니다.`;
    case 'longout': return `KV 캐시 사용률이 가장 크게 튀어서(평소의 ${times('kv_usage')}) 지표만 보면 "긴 입력"으로 읽습니다. 그런데 요청당 입력은 ${times('prompt_per_req')}로 그대로입니다. 길어진 것은 출력입니다.`;
    case 'gpu_hog': return `GPU 사용률은 평소의 ${x('gpu_util').toFixed(2)}배로 그대로입니다(원래 97%). 대신 토큰 사이 간격이 ${times('itl')}로 늘었습니다. 요청은 그대로인데 GPU를 나눠 쓰고 있습니다.`;
    case 'cpu_hog': return `서버 CPU가 평소의 ${times('host_cpu')}인데 추론 서버 자신의 CPU는 ${times('vllm_cpu')}로 줄었습니다. 다른 작업에 밀려 굶는 모양입니다.`;
    default: return '';
  }
}

// ───────────────────────── 재생 ─────────────────────────
function startEpisode(index, lead) {
  const inc = meta.incidents[index];
  let i = 0;
  while (t[i] < inc.start - lead) i++;
  ep = { inc, index, i0: i, i, deb: debouncer(meta.debounce), alarmAt: null, cause: null, done: false, logged: false };
  document.querySelectorAll('#inject button').forEach((b) => b.setAttribute('aria-pressed', b.dataset.fault === inc.fault ? 'true' : 'false'));
  for (const id of ['jRule', 'jNaive', 'jTruth']) { $(id).textContent = '—'; $(id).className = 'wait'; }
  $('jWhy').textContent = '';
}

/** 기록을 한 칸(약 1초) 진행한다. 구간이 끝나면 false. */
function step() {
  const { inc } = ep;
  const now = t[ep.i];
  const alarm = ep.deb(scores[ep.i] > meta.forest.threshold);
  if (alarm && ep.alarmAt === null && now >= inc.start) ep.alarmAt = now;
  if (ep.cause === null && ep.alarmAt !== null && now >= inc.start + meta.rca_win[1]) {
    const [a, b] = meta.rca_win;
    const win = rows.filter((_, k) => t[k] >= inc.start + a && t[k] < inc.start + b);
    const z = deviations(win, meta.feats, meta.rca.mu, meta.rca.sd);
    const mean = Object.fromEntries(meta.feats.map((k, j) => [k, win.reduce((s, r) => s + r[j], 0) / win.length]));
    ep.cause = { rule: rulesV2(z), naive: naiveCause(z), z, mean };
  }
  render();
  if (now >= inc.start + TAIL && ep.cause) { ep.done = true; return false; }
  ep.i++;
  return ep.i < rows.length;
}

function play(index, lead, then) {
  const my = ++runId;
  clearInterval(timer);
  startEpisode(index, lead);
  let last = performance.now();
  timer = setInterval(() => {
    if (my !== runId) return;
    // 가려진 탭은 타이머가 느려진다 → 지난 시간만큼 한꺼번에 따라잡는다(최대 2초치)
    const due = Math.min(2 * SPEED, Math.floor((performance.now() - last) * SPEED / 1000));
    last += due * 1000 / SPEED;
    for (let k = 0; k < due; k++) {
      if (!step()) { clearInterval(timer); setTimeout(() => { if (my === runId) then(); }, HOLD_MS); return; }
    }
  }, 1000 / SPEED);
}

function pick(fault) {
  const list = meta.incidents.map((r, i) => [r, i]).filter(([r]) => r.fault === fault).map(([, i]) => i);
  const n = nextOf[fault] ?? 0;
  nextOf[fault] = (n + 1) % list.length;
  return list[n];
}

function auto(k = 0) { play(pick(ORDER[k % ORDER.length]), LEAD_AUTO, () => auto(k + 1)); }

function inject(fault) {
  const k = ORDER.indexOf(fault);
  play(pick(fault), LEAD_CLICK, () => auto(k + 1));
}

// ───────────────────────── 그리기 ─────────────────────────
function render() {
  const { inc } = ep;
  const now = t[ep.i], since = now - inc.start;
  const mm = String(Math.floor(now / 60)).padStart(2, '0'), ss = String(Math.floor(now % 60)).padStart(2, '0');
  $('clock').textContent = `기록 재생 · ${mm}:${ss} / 41:00`;

  const st = $('status');
  if (ep.cause) {
    const f = FAULT[ep.cause.rule.cause];
    st.dataset.kind = 'cause';
    $('verdict').textContent = `원인: ${f.name}`;
    $('why').innerHTML = `장애를 넣고 <b>${(ep.alarmAt - inc.start).toFixed(1)}초</b> 만에 탐지했습니다. 권하는 조치: ${f.act}`;
  } else if (ep.alarmAt !== null) {
    st.dataset.kind = 'alarm';
    $('verdict').textContent = '이상 탐지';
    $('why').innerHTML = `장애를 넣고 <b>${(ep.alarmAt - inc.start).toFixed(1)}초</b> 만에 잡았습니다. 원인을 가리는 중입니다. <small>넣은 뒤 5~20초 구간을 봅니다.</small>`;
  } else if (since >= 0) {
    st.dataset.kind = 'ok';
    $('verdict').textContent = '정상 운영';
    $('why').innerHTML = `<small>방금 장애를 넣었습니다(${FAULT[inc.fault].how}). 아직 경보 기준을 넘지 않았습니다 — ${since.toFixed(0)}초 경과.</small>`;
  } else {
    st.dataset.kind = 'ok';
    $('verdict').textContent = '정상 운영';
    $('why').innerHTML = `<small>동시 사용자 8명의 평상시 부하입니다. ${Math.ceil(-since)}초 뒤 장애가 들어옵니다.</small>`;
  }

  const evidence = ep.cause ? new Set(ep.cause.rule.because) : new Set();
  for (const c of CHARTS) drawChart(c, now, inc, evidence.has(c.key) || (c.key === 'score' && ep.alarmAt !== null && !ep.cause));

  if (ep.cause && !ep.logged) {
    ep.logged = true;
    const rule = ep.cause.rule.cause, naive = ep.cause.naive.cause;
    $('jRule').textContent = FAULT[rule].name;
    $('jRule').className = rule === inc.fault ? 'hit' : 'miss';
    $('jNaive').textContent = `${FAULT[naive].name}${naive === inc.fault ? '' : ' (틀림)'}`;
    $('jNaive').className = naive === inc.fault ? 'hit' : 'miss';
    $('jTruth').textContent = FAULT[inc.fault].name;
    $('jTruth').className = '';
    $('jWhy').textContent = explain(inc, ep.cause.z, ep.cause.mean);
    seen.unshift({ fault: inc.fault, delay: ep.alarmAt - inc.start, rule, naive });
    $('log').replaceChildren(...seen.slice(0, 6).map((r) => {
      const tr = document.createElement('tr');
      const cells = [FAULT[r.fault].name, `${r.delay.toFixed(1)}초`, `${FAULT[r.rule].name}${r.rule === r.fault ? '' : ' ✕'}`, `${FAULT[r.naive].name}${r.naive === r.fault ? '' : ' ✕'}`];
      for (const c of cells) tr.append(Object.assign(document.createElement('td'), { textContent: c }));
      return tr;
    }));
  }
}

function drawChart(c, now, inc, on) {
  const box = c.el;
  const data = c.key === 'score' ? scores : col[c.key];
  const v = data[ep.i];
  box.dataset.on = on ? '1' : '0';
  box.querySelector('b').textContent = c.fmt(v);
  const r = c.key === 'score' ? null : ratio(c.key, v);
  box.querySelector('.r').textContent = c.key === 'score'
    ? `경보 기준 ${meta.forest.threshold.toFixed(3)}`
    : `평소의 ${r.toFixed(r >= 10 ? 0 : 2)}배`;

  const cv = box.querySelector('canvas');
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const w = cv.clientWidth, h = cv.clientHeight;
  if (cv.width !== Math.round(w * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
  const g = cv.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);

  const t0 = now - SPAN;
  let lo = 0;
  while (lo < ep.i && (t[lo] < t0 || t[lo] < inc.start - 40)) lo++;   // 앞 장애의 회복 구간은 그리지 않는다
  const base = c.key === 'score' ? meta.forest.threshold : meta.normal_mean[c.key];
  let max = base * 1.6, min = 0;
  for (let k = lo; k <= ep.i; k++) if (data[k] > max) max = data[k];
  if (c.key === 'score') min = 0.3;
  const X = (tt) => (tt - t0) / SPAN * w;
  const Y = (val) => h - 3 - (val - min) / (max * 1.08 - min) * (h - 6);

  if (now >= inc.start) {                       // 장애를 넣은 구간
    g.fillStyle = 'rgba(255, 59, 78, .10)';
    g.fillRect(X(inc.start), 0, w - X(inc.start), h);
    g.strokeStyle = 'rgba(255, 59, 78, .7)'; g.lineWidth = 1; g.setLineDash([3, 3]);
    g.beginPath(); g.moveTo(X(inc.start), 0); g.lineTo(X(inc.start), h); g.stroke(); g.setLineDash([]);
  }
  g.strokeStyle = 'rgba(139, 155, 176, .55)'; g.lineWidth = 1; g.setLineDash([2, 4]);   // 평소 값(이상 점수는 경보 기준)
  g.beginPath(); g.moveTo(0, Y(base)); g.lineTo(w, Y(base)); g.stroke(); g.setLineDash([]);
  if (c.key === 'score' && ep.alarmAt !== null) {
    g.strokeStyle = '#f08a3c'; g.lineWidth = 2;
    g.beginPath(); g.moveTo(X(ep.alarmAt), 0); g.lineTo(X(ep.alarmAt), h); g.stroke();
  }
  g.strokeStyle = on ? '#ff3b4e' : '#5bc0ff'; g.lineWidth = 1.8; g.lineJoin = 'round';
  g.beginPath();
  for (let k = lo; k <= ep.i; k++) { const x = X(t[k]), y = Y(data[k]); if (k === lo) g.moveTo(x, y); else g.lineTo(x, y); }
  g.stroke();
}

// ───────────────────────── 시작 ─────────────────────────
function build() {
  for (const c of CHARTS) {
    const el = document.createElement('div');
    el.className = 'chart';
    el.innerHTML = `<div class="h"><span>${c.name}</span><b></b></div><div class="r"></div><canvas></canvas>`;
    $('charts').append(el);
    c.el = el;
  }
  for (const f of ORDER) {
    const b = document.createElement('button');
    b.dataset.fault = f;
    b.innerHTML = `<b>${FAULT[f].name}</b><span>${FAULT[f].how}</span>`;
    b.onclick = () => inject(f);
    $('inject').append(b);
  }
  const d4 = meta.expected.d4, n = meta.incidents.length;
  $('nDetect').textContent = `${Math.round(d4.recall * n)}/${n}`;
  $('nDelay').textContent = `${d4.delay_median_s.toFixed(1)}초`;
  $('nCause').textContent = `${meta.incidents.filter((r) => r.rules_v2 === r.fault).length}/${n}`;
  const naive = meta.incidents.filter((r) => r.naive === r.fault).length;
  $('nNote').textContent = `가장 늦은 탐지는 ${d4.delay_max_s.toFixed(0)}초(긴 출력 요청)였습니다. 가장 크게 튄 지표만 보고 원인을 읽으면 ${naive}/${n}입니다 — 긴 출력 ${n / 5 | 0}건을 전부 긴 입력으로 읽습니다.`;
}

async function main() {
  const [m, fb, tb] = await Promise.all([
    fetch('data/replay.json').then((r) => r.json()),
    fetch('data/features.f64').then((r) => r.arrayBuffer()),
    fetch('data/iforest.bin').then((r) => r.arrayBuffer()),
  ]);
  meta = m;
  ({ t, rows } = parseFeatures(fb, meta.feats.length));
  const trees = parseForest(tb);
  const t0 = performance.now();
  scores = Float64Array.from(rows, (r) => anomalyScore(trees, standardize(Array.from(r), meta.scaler), meta.forest.max_samples));
  const scoreMs = performance.now() - t0;
  col = Object.fromEntries(meta.feats.map((k, j) => [k, Float64Array.from(rows, (r) => r[j])]));
  build();
  window.__tower = { meta, scores, scoreMs, inject, step, startEpisode, get ep() { return ep; }, stop() { runId++; clearInterval(timer); } };
  auto();
}

main().catch((e) => { $('verdict').textContent = '열 수 없습니다'; $('why').textContent = e.message; });
