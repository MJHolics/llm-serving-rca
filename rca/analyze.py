"""dev 실행으로 보정·학습하고 test 실행에서 장애 탐지와 원인 분석(RCA)을 평가한다.

탐지: 클라이언트 지연 p90(SLO 신호) 기반 3종 + 다변량 IsolationForest
RCA : 가장 크게 튄 메트릭 / 도메인 규칙 / dev 장애로 학습한 분류기

사용: python rca/analyze.py --dev results/dev --test results/test --out results/report
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FAULTS = ["surge", "longctx", "longout", "gpu_hog", "cpu_hog"]
GRACE = 30        # 장애 종료 후 이 시간 안의 경보는 오경보로 세지 않는다(회복 중)
DEBOUNCE = 3      # 연속 N초 넘어야 경보
RCA_WIN = (5, 20)  # 장애 시작 +5~+20초 구간으로 원인 판정


# ------------------------------------------------------------------ 특징

def load(run: Path):
    d = pd.read_csv(run / "metrics.csv")
    ev = pd.read_csv(run / "events.csv")
    f = pd.DataFrame({"t": d["t"], "label": d["label"]})

    def rate(col):
        return d[col].diff().clip(lower=0)

    def hist_mean(name):
        return (d[f"{name}_sum"].diff() / d[f"{name}_count"].diff().replace(0, np.nan))

    f["lat_p90"] = d["cli_lat_p90"].ffill()
    f["ttft_p90"] = d["cli_ttft_p90"].ffill()
    f["done_rps"] = d["cli_done"]
    f["running"] = d["vllm:num_requests_running"]
    f["waiting"] = d["vllm:num_requests_waiting"]
    f["kv_usage"] = d["vllm:kv_cache_usage_perc"]
    f["gen_tps"] = rate("vllm:generation_tokens_total")
    f["prompt_tps"] = rate("vllm:prompt_tokens_total")
    f["preempt"] = rate("vllm:num_preemptions_total")
    f["itl"] = hist_mean("vllm:inter_token_latency_seconds").ffill()
    f["prompt_per_req"] = (rate("vllm:prompt_tokens_total")
                           / rate("vllm:request_success_total").replace(0, np.nan)).ffill()
    f["gen_per_running"] = f["gen_tps"] / f["running"].replace(0, np.nan)
    f["gpu_util"] = d["gpu_util"]
    f["gpu_power"] = d["gpu_power_w"]
    f["host_cpu"] = d["host_cpu_pct"]
    f["vllm_cpu"] = d["vllm_cpu_pct"]
    f = f.iloc[1:].reset_index(drop=True)
    f = f.fillna(f.median(numeric_only=True))
    return f, ev


FEATS = ["lat_p90", "ttft_p90", "done_rps", "running", "waiting", "kv_usage", "gen_tps", "prompt_tps",
         "preempt", "itl", "prompt_per_req", "gen_per_running", "gpu_util", "gpu_power", "host_cpu", "vllm_cpu"]


def normal_mask(f, ev):
    """장애 구간과 종료 후 GRACE초를 뺀 순수 정상 구간."""
    m = pd.Series(True, index=f.index)
    for _, e in ev.iterrows():
        m &= ~((f["t"] >= e.start) & (f["t"] < e.end + GRACE))
    return m & (f["t"] > min(60, ev.start.min()))  # 워밍업 제외


# ------------------------------------------------------------------ 탐지

def debounce(raw: np.ndarray, n=DEBOUNCE) -> np.ndarray:
    out = np.zeros_like(raw, dtype=bool)
    run = 0
    for i, v in enumerate(raw):
        run = run + 1 if v else 0
        out[i] = run >= n
    return out


def score_alarms(f, ev, alarm):
    """장애 단위 재현율·탐지 지연, 정상 구간 오경보(경보 '발생' 횟수/시간)."""
    t = f["t"].to_numpy()
    hits, delays = [], []
    for _, e in ev.iterrows():
        w = (t >= e.start) & (t < e.end)
        idx = np.where(alarm & w)[0]
        hits.append(len(idx) > 0)
        delays.append(t[idx[0]] - e.start if len(idx) else np.nan)
    nm = normal_mask(f, ev).to_numpy()
    onsets = alarm & ~np.r_[False, alarm[:-1]]
    fa = int((onsets & nm).sum())
    hours = nm.sum() / 3600
    per_fault = {k: float(np.mean([h for h, n in zip(hits, ev.fault) if n == k])) for k in FAULTS}
    return {"recall": float(np.mean(hits)), "fa_per_hour": fa / hours, "false_alarms": fa,
            "normal_hours": round(hours, 3), "delay_median_s": float(np.nanmedian(delays)),
            "delay_max_s": float(np.nanmax(delays)), "recall_by_fault": per_fault}


def detectors(dev, dev_ev, test, test_ev):
    res = {}
    nd = normal_mask(dev, dev_ev)

    # D1 선험적 SLO — "p90 1.5초"처럼 흔히 정하는 숫자. 보정 없음.
    res["D1_static_SLO_1.5s"] = score_alarms(test, test_ev, debounce(test["lat_p90"].to_numpy() > 1.5))

    # D2 dev 정상 구간의 99.5 분위수를 임계로
    thr = float(np.quantile(dev.loc[nd, "lat_p90"], 0.995))
    res[f"D2_quantile_{thr:.2f}s"] = score_alarms(test, test_ev, debounce(test["lat_p90"].to_numpy() > thr))

    # D3 rolling z-score: 직전 120초(자기 자신 제외) 대비. k는 dev에서 오경보 0이 되는 최소값
    def zs(f, k):
        x = f["lat_p90"]
        mu = x.shift(1).rolling(120, min_periods=30).median()
        sd = x.shift(1).rolling(120, min_periods=30).std().clip(lower=0.05)
        return debounce(((x - mu) / sd).to_numpy() > k)
    k_best = None
    # 격자 하한은 처음 2.0이었는데 dev에서 2.0부터 이미 오경보 0이라 더 낮은 k를 못 봤다 → 0.5부터
    for k in np.arange(0.5, 12.01, 0.25):
        if score_alarms(dev, dev_ev, zs(dev, k))["false_alarms"] == 0:
            k_best = float(k)
            break
    k_best = k_best or 12.0
    res[f"D3_rolling_z_k{k_best}"] = score_alarms(test, test_ev, zs(test, k_best))

    # D4 IsolationForest(전 특징) — dev 정상으로 학습, 임계는 dev 정상 점수의 0.5 분위
    sc = StandardScaler().fit(dev.loc[nd, FEATS])
    iso = IsolationForest(n_estimators=300, random_state=0).fit(sc.transform(dev.loc[nd, FEATS]))
    s_dev = -iso.score_samples(sc.transform(dev.loc[nd, FEATS]))
    ithr = float(np.quantile(s_dev, 0.995))
    s_test = -iso.score_samples(sc.transform(test[FEATS]))
    res["D4_isolation_forest"] = score_alarms(test, test_ev, debounce(s_test > ithr))
    return res, {"D2_thr": thr, "D3_k": k_best}


# ------------------------------------------------------------------ RCA

def incident_windows(f, ev, win=RCA_WIN):
    rows = []
    for i, e in ev.iterrows():
        w = f[(f["t"] >= e.start + win[0]) & (f["t"] < e.start + win[1])]
        rows.append((i, e.fault, w))
    return rows


def deviations(w, mu, sd):
    return ((w[FEATS].mean() - mu) / sd)


# "가장 크게 튄 메트릭"을 원인으로 읽는 매핑 — 사람이 대시보드를 보고 하는 추론을 그대로 옮겼다
METRIC_TO_CAUSE = {
    "running": "surge", "done_rps": "surge",
    "prompt_tps": "longctx", "prompt_per_req": "longctx", "kv_usage": "longctx", "waiting": "longctx",
    "preempt": "longctx", "ttft_p90": "longctx",
    "gen_tps": "longout", "gen_per_running": "longout",
    "gpu_util": "gpu_hog", "gpu_power": "gpu_hog", "itl": "gpu_hog",
    "host_cpu": "cpu_hog", "vllm_cpu": "cpu_hog",
    "lat_p90": None,  # 증상이지 원인이 아니다
}


def rca_max_dev(z):
    z = z.drop("lat_p90").abs().sort_values(ascending=False)
    return METRIC_TO_CAUSE[z.index[0]], z.index[0]


def rca_rules(z):
    """도메인 규칙. smoke 실행만 보고 test를 보기 전에 고정했다."""
    if z["host_cpu"] > 5:
        return "cpu_hog"
    if z["prompt_per_req"] > 5:
        return "longctx"
    if z["running"] > 5 and z["gen_per_running"] < 0 and z["done_rps"] > 3:
        return "surge"
    if z["running"] > 5:
        return "longout"
    if z["itl"] > 3:
        return "gpu_hog"
    return "unknown"


def rca_rules_v2(z):
    """dev에서 고친 규칙(test를 보기 전 고정).

    v1은 dev gpu_hog 3건을 전부 cpu_hog로 틀렸다. GPU를 점유하는 프로세스도 커널을 쉬지 않고
    제출하느라 CPU 코어 하나를 태워서 host_cpu z가 6까지 오른다(v1 문턱 5를 넘는다).
    진짜 CPU 포화는 z 78이고 vLLM 프로세스 자신의 CPU가 오히려 줄어든다(-9.9, 굶는다).
    """
    if z["host_cpu"] > 30 or z["vllm_cpu"] < -5:
        return "cpu_hog"
    if z["prompt_per_req"] > 5:
        return "longctx"
    if z["running"] > 5 and z["gen_per_running"] < 0 and z["done_rps"] > 3:
        return "surge"
    if z["running"] > 5:
        return "longout"
    if z["itl"] > 3:
        return "gpu_hog"
    return "unknown"


def rca(dev, dev_ev, test, test_ev):
    nd = normal_mask(dev, dev_ev)
    mu, sd = dev.loc[nd, FEATS].mean(), dev.loc[nd, FEATS].std()
    # 평상시 거의 0인 계수(대기·선점)는 표준편차도 0이라 z가 수십만으로 튄다 → 분모 하한
    floor = np.maximum(0.05 * mu.abs(), 1e-3)
    floor[["waiting", "preempt"]] = 1.0
    sd = np.maximum(sd, floor)

    def table(f, ev):
        X, y = [], []
        for _, fault, w in incident_windows(f, ev):
            X.append(deviations(w, mu, sd))
            y.append(fault)
        return pd.DataFrame(X), y

    Xd, yd = table(dev, dev_ev)
    Xt, yt = table(test, test_ev)
    out = {}

    pred_md, top_metric = zip(*[rca_max_dev(Xt.iloc[i]) for i in range(len(Xt))])
    out["R1_max_deviation"] = (list(pred_md), list(top_metric))
    out["R2_rules_v1_smoke"] = ([rca_rules(Xt.iloc[i]) for i in range(len(Xt))], None)
    out["R2b_rules_v2_dev"] = ([rca_rules_v2(Xt.iloc[i]) for i in range(len(Xt))], None)

    # 초 단위 창을 표본으로 늘려 학습(장애 20건 × 15초)
    def seconds_table(f, ev):
        X, y = [], []
        for _, fault, w in incident_windows(f, ev):
            X.append((w[FEATS] - mu) / sd)
            y += [fault] * len(w)
        return pd.concat(X), y
    Xs, ys = seconds_table(dev, dev_ev)
    rf = RandomForestClassifier(n_estimators=300, random_state=0).fit(Xs, ys)
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xs, ys)
    out["R3_random_forest"] = (list(rf.predict(Xt[FEATS])), None)
    out["R3b_logistic"] = (list(lr.predict(Xt[FEATS])), None)
    importances = dict(sorted(zip(FEATS, rf.feature_importances_), key=lambda kv: -kv[1]))

    summary = {}
    for name, (pred, extra) in out.items():
        acc = float(np.mean([p == t for p, t in zip(pred, yt)]))
        cm = confusion_matrix(yt, pred, labels=FAULTS + ["unknown"])
        summary[name] = {"top1_acc": acc, "n": len(yt),
                         "by_fault": {k: float(np.mean([p == t for p, t in zip(pred, yt) if t == k])) for k in FAULTS},
                         "confusion": cm[:len(FAULTS)].tolist(),
                         "errors": [{"true": t, "pred": p, **({"top_metric": extra[i]} if extra else {})}
                                    for i, (p, t) in enumerate(zip(pred, yt)) if p != t]}
    dev_z = Xd.groupby(pd.Series(yd, name="fault")).mean().round(1)
    return summary, importances, dev_z


def main():
    global GRACE
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--grace", type=int, default=GRACE)
    a = ap.parse_args()
    GRACE = a.grace
    dev, dev_ev = load(Path(a.dev))
    test, test_ev = load(Path(a.test))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    det, cal = detectors(dev, dev_ev, test, test_ev)
    rc, imp, dev_z = rca(dev, dev_ev, test, test_ev)

    # 장애별 평균 프로파일(정상 대비 배율) — 어떤 신호가 어떤 장애에 반응하는가
    nd = normal_mask(test, test_ev)
    prof = test.groupby("label")[FEATS].mean().T
    prof = prof.div(test.loc[nd, FEATS].mean(), axis=0).round(2)

    (out / "detection.json").write_text(json.dumps({"calibration": cal, **det}, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "rca.json").write_text(json.dumps({"summary": rc, "rf_importance": imp}, indent=1, ensure_ascii=False), encoding="utf-8")
    prof.to_csv(out / "profile_ratio_vs_normal.csv")
    dev_z.T.to_csv(out / "dev_incident_zscores.csv")

    print("== 탐지 (test)")
    for k, v in det.items():
        print(f"{k:28s} recall {v['recall']:.2f}  FA/h {v['fa_per_hour']:.2f}  delay med {v['delay_median_s']:.1f}s "
              f"max {v['delay_max_s']:.1f}s  by_fault {v['recall_by_fault']}")
    print("\n== RCA top-1 (test)")
    for k, v in rc.items():
        print(f"{k:20s} {v['top1_acc']:.2f}  {v['by_fault']}")
        for e in v["errors"][:8]:
            print("    ", e)
    print("\n== RF 중요도 상위", list(imp.items())[:6])
    print("\n== 정상 대비 배율(test)\n", prof.to_string())
    print("\n== dev 장애별 평균 z\n", dev_z.T.to_string())


if __name__ == "__main__":
    main()
