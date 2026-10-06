// core.js(이식본)를 파이썬 원본이 낸 정답(data/replay.json, scripts/build_web.py)과 대조한다.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { parseForest, parseFeatures, anomalyScore, standardize, debouncer, deviations, naiveCause, rulesV2, rulesV1, avgPathLength } from '../core.js';

const read = (name) => readFileSync(new URL(`../data/${name}`, import.meta.url));
const ab = (b) => b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
const meta = JSON.parse(read('replay.json'));
const trees = parseForest(ab(read('iforest.bin')));
const { t, rows } = parseFeatures(ab(read('features.f64')), meta.feats.length);

const scores = rows.map((r) => anomalyScore(trees, standardize(Array.from(r), meta.scaler), meta.forest.max_samples));
const deb = debouncer(meta.debounce);
const alarm = scores.map((s) => deb(s > meta.forest.threshold));

test('자료 모양', () => {
  assert.equal(trees.length, meta.forest.trees);
  assert.equal(trees.reduce((s, tr) => s + tr.feature.length, 0), meta.forest.nodes);
  assert.equal(rows.length, meta.rows);
});

test('평균 경로 길이 c(n)', () => {
  assert.equal(avgPathLength(1), 0);
  assert.equal(avgPathLength(2), 1);
  assert.ok(Math.abs(avgPathLength(256) - 10.244770920119917) < 1e-12);   // sklearn _average_path_length
  assert.ok(Math.abs(avgPathLength(17) - 4.817255833106158) < 1e-12);
});

test('이상 점수가 sklearn score_samples와 같다', () => {
  let worst = 0;
  for (const [i, v] of Object.entries(meta.expected.score_samples)) worst = Math.max(worst, Math.abs(scores[+i] - v));
  assert.ok(worst < 1e-9, `최대 차 ${worst}`);
});

test('경보가 켜진 초와 켜지는 시각이 파이썬과 같다', () => {
  assert.equal(alarm.filter(Boolean).length, meta.expected.alarm_seconds);
  const onsets = [];
  alarm.forEach((a, i) => { if (a && !alarm[i - 1]) onsets.push(t[i]); });
  assert.deepEqual(onsets, meta.expected.alarm_onsets);
});

test('장애 20건: 탐지 지연이 파이썬과 같다', () => {
  for (const inc of meta.incidents) {
    const i = alarm.findIndex((a, k) => a && t[k] >= inc.start && t[k] < inc.end);
    const delay = i < 0 ? null : t[i] - inc.start;
    assert.equal(delay, inc.delay);
  }
});

test('장애 20건: 구간 편차와 세 가지 원인 판정이 파이썬과 같다', () => {
  const [a, b] = meta.rca_win;
  for (const inc of meta.incidents) {
    const win = rows.filter((_, k) => t[k] >= inc.start + a && t[k] < inc.start + b);
    const z = deviations(win, meta.feats, meta.rca.mu, meta.rca.sd);
    for (const k of meta.feats) assert.ok(Math.abs(z[k] - inc.z[k]) < 1e-6 * Math.max(1, Math.abs(inc.z[k])), `${inc.fault} ${k}`);
    assert.equal(rulesV2(z).cause, inc.rules_v2);
    assert.equal(rulesV1(z), inc.rules_v1);
    assert.deepEqual(naiveCause(z), { cause: inc.naive, metric: inc.naive_metric });
  }
});

test('보고한 수치: 규칙 v2 20/20, 가장 튄 지표 16/20, 규칙 v1 16/20', () => {
  const hit = (key) => meta.incidents.filter((r) => r[key] === r.fault).length;
  assert.deepEqual([hit('rules_v2'), hit('naive'), hit('rules_v1')], [20, 16, 16]);
});

test('연속 3번 넘어야 경보', () => {
  const d = debouncer(3);
  assert.deepEqual([1, 1, 0, 1, 1, 1, 1, 0].map((v) => d(!!v)), [false, false, false, false, false, true, true, false]);
});
