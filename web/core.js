// 탐지와 원인 판정의 순수 계산. 정의의 원본은 rca/analyze.py이고 여기는 이식본이다.
// node 테스트(tests/core.test.mjs)가 파이썬이 낸 정답(data/replay.json)과 대조한다.

/** data/iforest.bin → 나무 배열. 배치: scripts/build_web.py export_forest. */
export function parseForest(buffer) {
  const view = new DataView(buffer);
  const trees = [];
  let o = 0;
  const pad = () => { o += (4 - o % 4) % 4; };
  while (o < buffer.byteLength) {
    const n = view.getInt32(o, true); o += 4;
    const feature = new Int16Array(buffer.slice(o, o + 2 * n)); o += 2 * n; pad();
    const threshold = new Float32Array(buffer.slice(o, o + 4 * n)); o += 4 * n;
    const left = new Int16Array(buffer.slice(o, o + 2 * n)); o += 2 * n;
    const right = new Int16Array(buffer.slice(o, o + 2 * n)); o += 2 * n;
    const samples = new Int16Array(buffer.slice(o, o + 2 * n)); o += 2 * n; pad();
    trees.push({ feature, threshold, left, right, samples });
  }
  return trees;
}

/** 표본 n개짜리 이진 탐색 나무에서 탐색이 실패할 때까지의 평균 경로 길이(Isolation Forest의 c(n)). */
export function avgPathLength(n) {
  if (n <= 1) return 0;
  if (n === 2) return 1;
  return 2 * (Math.log(n - 1) + 0.5772156649015329) - 2 * (n - 1) / n;
}

/** 표준화한 한 행 → 이상 점수(0~1, 클수록 이상). sklearn의 -score_samples와 같은 값. */
export function anomalyScore(trees, x, maxSamples) {
  const xf = Float32Array.from(x);          // sklearn 나무는 입력을 float32로 비교한다
  let sum = 0;
  for (const t of trees) {
    let node = 0, depth = 0;
    while (t.left[node] !== -1) {
      node = xf[t.feature[node]] <= t.threshold[node] ? t.left[node] : t.right[node];
      depth++;
    }
    sum += depth + avgPathLength(t.samples[node]);
  }
  return 2 ** (-(sum / trees.length) / avgPathLength(maxSamples));
}

export function standardize(row, scaler) {
  return row.map((v, i) => (v - scaler.mean[i]) / scaler.scale[i]);
}

/** 연속 n번 넘어야 경보. 한 칸씩 넣는 상태 기계(재생 화면이 초마다 부른다). */
export function debouncer(n) {
  let run = 0;
  return (raw) => { run = raw ? run + 1 : 0; return run >= n; };
}

/** 구간 평균이 dev 정상 평균에서 몇 표준편차 떨어졌나. rows = 특징 배열들. */
export function deviations(rows, feats, mu, sd) {
  const z = {};
  feats.forEach((k, i) => {
    let s = 0;
    for (const r of rows) s += r[i];
    z[k] = (s / rows.length - mu[i]) / sd[i];
  });
  return z;
}

// 가장 크게 튄 지표를 원인으로 읽는 매핑 — 대시보드를 보는 사람의 추론(analyze.METRIC_TO_CAUSE)
const METRIC_TO_CAUSE = {
  running: 'surge', done_rps: 'surge',
  prompt_tps: 'longctx', prompt_per_req: 'longctx', kv_usage: 'longctx', waiting: 'longctx', preempt: 'longctx', ttft_p90: 'longctx',
  gen_tps: 'longout', gen_per_running: 'longout',
  gpu_util: 'gpu_hog', gpu_power: 'gpu_hog', itl: 'gpu_hog',
  host_cpu: 'cpu_hog', vllm_cpu: 'cpu_hog',
};

/** "가장 붉은 지표"로 추정. 지연(lat_p90)은 증상이라 뺀다. */
export function naiveCause(z) {
  let top = null;
  for (const k of Object.keys(z)) {
    if (k === 'lat_p90') continue;
    if (top === null || Math.abs(z[k]) > Math.abs(z[top])) top = k;
  }
  return { cause: METRIC_TO_CAUSE[top], metric: top };
}

/** 규칙 v2(analyze.rca_rules_v2). 맞은 규칙의 근거도 같이 돌려준다. */
export function rulesV2(z) {
  if (z.host_cpu > 30 || z.vllm_cpu < -5) return { cause: 'cpu_hog', because: ['host_cpu', 'vllm_cpu'] };
  if (z.prompt_per_req > 5) return { cause: 'longctx', because: ['prompt_per_req'] };
  if (z.running > 5 && z.gen_per_running < 0 && z.done_rps > 3) return { cause: 'surge', because: ['running', 'done_rps', 'gen_per_running'] };
  if (z.running > 5) return { cause: 'longout', because: ['running', 'prompt_per_req'] };
  if (z.itl > 3) return { cause: 'gpu_hog', because: ['itl', 'gpu_util'] };
  return { cause: 'unknown', because: [] };
}

/** 고치기 전 규칙 v1(analyze.rca_rules) — GPU 경합을 CPU 경합으로 읽는다. 화면의 비교용. */
export function rulesV1(z) {
  if (z.host_cpu > 5) return 'cpu_hog';
  if (z.prompt_per_req > 5) return 'longctx';
  if (z.running > 5 && z.gen_per_running < 0 && z.done_rps > 3) return 'surge';
  if (z.running > 5) return 'longout';
  if (z.itl > 3) return 'gpu_hog';
  return 'unknown';
}

/** features.f64 → { t, rows }. 한 행은 [t, ...feats]. */
export function parseFeatures(buffer, nFeats) {
  const a = new Float64Array(buffer);
  const w = nFeats + 1, n = a.length / w;
  const t = new Float64Array(n), rows = [];
  for (let i = 0; i < n; i++) {
    t[i] = a[i * w];
    rows.push(a.subarray(i * w + 1, (i + 1) * w));
  }
  return { t, rows };
}
