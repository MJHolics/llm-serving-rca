"""브라우저 재생 화면(web/)에 들어갈 자료를 만든다.

web/data/features.f64  test 실행의 초 단위 특징(t + FEATS, float64 행 우선)
web/data/iforest.bin   dev 정상 구간으로 학습한 IsolationForest의 나무(브라우저가 직접 점수를 낸다)
web/data/replay.json   구간·보정값·파이썬이 낸 정답(JS 대조용)

탐지·원인 판정은 rca/analyze.py와 같은 정의를 그대로 쓴다. 여기서 새로 정하는 값은 없다.

사용: python scripts/build_web.py
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rca"))
import analyze as A  # noqa: E402

OUT = ROOT / "web" / "data"


def export_forest(iso: IsolationForest, path: Path) -> dict:
    """나무마다 [노드 수][feature i16][threshold f32][left i16][right i16][n_samples i16]."""
    buf = bytearray()
    total = 0
    for est in iso.estimators_:
        t = est.tree_
        n = t.node_count
        total += n
        buf += struct.pack("<i", n)
        buf += t.feature.astype("<i2").tobytes()
        buf += b"\0" * ((4 - len(buf) % 4) % 4)
        buf += t.threshold.astype("<f4").tobytes()
        buf += t.children_left.astype("<i2").tobytes()
        buf += t.children_right.astype("<i2").tobytes()
        buf += t.n_node_samples.astype("<i2").tobytes()
        buf += b"\0" * ((4 - len(buf) % 4) % 4)
    path.write_bytes(bytes(buf))
    return {"trees": len(iso.estimators_), "nodes": total, "max_samples": int(iso.max_samples_)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dev, dev_ev = A.load(ROOT / "results" / "dev")
    test, test_ev = A.load(ROOT / "results" / "test")
    nd = A.normal_mask(dev, dev_ev)

    # 탐지 D4 — analyze.detectors와 같은 순서·같은 시드
    sc = StandardScaler().fit(dev.loc[nd, A.FEATS])
    iso = IsolationForest(n_estimators=300, random_state=0).fit(sc.transform(dev.loc[nd, A.FEATS]))
    ithr = float(np.quantile(-iso.score_samples(sc.transform(dev.loc[nd, A.FEATS])), 0.995))
    s_test = -iso.score_samples(sc.transform(test[A.FEATS]))
    alarm = A.debounce(s_test > ithr)
    d4 = A.score_alarms(test, test_ev, alarm)
    d2_thr = float(np.quantile(dev.loc[nd, "lat_p90"], 0.995))
    d1 = A.score_alarms(test, test_ev, A.debounce(test["lat_p90"].to_numpy() > 1.5))

    # 원인 판정 — analyze.rca와 같은 mu·sd(분모 하한 포함)
    mu, sd = dev.loc[nd, A.FEATS].mean(), dev.loc[nd, A.FEATS].std()
    floor = np.maximum(0.05 * mu.abs(), 1e-3)
    floor[["waiting", "preempt"]] = 1.0
    sd = np.maximum(sd, floor)

    t = test["t"].to_numpy()
    incidents = []
    for i, fault, w in A.incident_windows(test, test_ev):
        e = test_ev.iloc[i]
        z = A.deviations(w, mu, sd)
        idx = np.where(alarm & (t >= e.start) & (t < e.end))[0]
        naive, top = A.rca_max_dev(z)
        incidents.append({
            "fault": fault, "start": float(e.start), "end": float(e.end),
            "delay": float(t[idx[0]] - e.start) if len(idx) else None,
            "rules_v2": A.rca_rules_v2(z), "rules_v1": A.rca_rules(z),
            "naive": naive, "naive_metric": top,
            "z": {k: float(z[k]) for k in A.FEATS},
        })

    nm = A.normal_mask(test, test_ev).to_numpy()
    normal_mean = test.loc[nm, A.FEATS].mean()

    np.column_stack([t, test[A.FEATS].to_numpy()]).astype("<f8").tofile(OUT / "features.f64")
    forest = export_forest(iso, OUT / "iforest.bin")

    meta = {
        "feats": A.FEATS, "rows": int(len(test)), "debounce": A.DEBOUNCE, "rca_win": list(A.RCA_WIN),
        "scaler": {"mean": sc.mean_.tolist(), "scale": sc.scale_.tolist()},
        "forest": {**forest, "threshold": ithr},
        "rca": {"mu": mu.tolist(), "sd": sd.tolist()},
        "slo": {"static": 1.5, "dev_quantile": d2_thr},
        "normal_mean": {k: float(normal_mean[k]) for k in A.FEATS},
        "incidents": incidents,
        "expected": {
            "score_samples": {str(i): float(s_test[i]) for i in range(0, len(s_test), 97)},
            "alarm_onsets": [float(v) for v in t[alarm & ~np.r_[False, alarm[:-1]]]],
            "alarm_seconds": int(alarm.sum()),
            "d4": d4, "d1": d1,
        },
    }
    (OUT / "replay.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    acc = np.mean([r["rules_v2"] == r["fault"] for r in incidents])
    print(f"rows {len(test)}  forest {forest}  thr {ithr:.4f}")
    print(f"D4 recall {d4['recall']:.2f} FA {d4['false_alarms']} median {d4['delay_median_s']:.2f}s max {d4['delay_max_s']:.2f}s")
    print(f"rules v2 {acc:.2f}  naive {np.mean([r['naive'] == r['fault'] for r in incidents]):.2f}"
          f"  v1 {np.mean([r['rules_v1'] == r['fault'] for r in incidents]):.2f}")
    for p in sorted(OUT.iterdir()):
        print(f"  {p.name:16s} {p.stat().st_size / 1024:8.1f} KB")


if __name__ == "__main__":
    main()
