"""vLLM 서버에 평상시 부하를 걸고, 원인이 알려진 장애를 순서를 섞어 주입하며 1초 단위 메트릭을 기록한다.

출력 (results/<run>/):
  metrics.csv  1초마다 한 행 — vLLM /metrics, GPU(NVML), 호스트·vLLM 프로세스 CPU, 클라이언트 지연, 현재 라벨
  events.csv   장애 구간(정답 라벨) — fault, start, end
  run.json     실행 설정

행을 매초 flush한다 — 긴 측정이 중간에 죽어도 그때까지는 남는다.
"""
import argparse
import asyncio
import csv
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import aiohttp
import psutil

try:
    import pynvml
    pynvml.nvmlInit()
    _NVML = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:  # NVML이 없으면 GPU 열은 빈 값
    _NVML = None

URL = "http://localhost:8000"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
FAULTS = ["surge", "longctx", "longout", "gpu_hog", "cpu_hog"]
HERE = Path(__file__).resolve().parent

_WORDS = ("system metric latency cache block token queue request server model engine "
          "anomaly root cause gpu memory kernel batch decode prefill schedule monitor").split()


def make_prompt(n_words: int, rng: random.Random) -> str:
    # 영어 단어 1개 ≈ 1~1.3 토큰. 정확한 길이보다 "요청마다 같은 일량"이 중요하다.
    return " ".join(rng.choice(_WORDS) for _ in range(n_words))


class Client:
    """완료된 요청을 쌓아두고, 수집기가 매초 비워 간다."""

    def __init__(self, session: aiohttp.ClientSession):
        self.s = session
        self.done: list[tuple[float, float, bool]] = []  # (e2e, ttft, ok)

    async def one(self, prompt: str, max_tokens: int):
        t0 = time.perf_counter()
        ttft = float("nan")
        ok = False
        try:
            body = {"model": MODEL, "prompt": prompt, "max_tokens": max_tokens,
                    "ignore_eos": True, "stream": True, "temperature": 0.0}
            async with self.s.post(f"{URL}/v1/completions", json=body) as r:
                async for line in r.content:
                    if line.startswith(b"data:") and ttft != ttft:
                        ttft = time.perf_counter() - t0
                ok = r.status == 200
        except Exception:
            ok = False
        self.done.append((time.perf_counter() - t0, ttft, ok))


async def worker(client: Client, stop: asyncio.Event, n_words: int, max_tokens: int, seed: int):
    rng = random.Random(seed)
    while not stop.is_set():
        await client.one(make_prompt(n_words, rng), max_tokens)
        await asyncio.sleep(rng.expovariate(1 / 0.2))  # 사용자 생각 시간 평균 0.2초


def spawn_workers(client, n, n_words, max_tokens, seed_base):
    stop = asyncio.Event()
    tasks = [asyncio.create_task(worker(client, stop, n_words, max_tokens, seed_base + i)) for i in range(n)]
    return stop, tasks


# ---------------------------------------------------------------- 수집

def parse_prom(text: str) -> dict[str, float]:
    """vllm:* 계열만. 라벨은 합산하고, 히스토그램은 _sum/_count만 남긴다."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line.startswith("vllm:") or "_bucket" in line:
            continue
        try:
            head, val = line.rsplit(" ", 1)
            name = head.split("{", 1)[0]
            out[name] = out.get(name, 0.0) + float(val)
        except ValueError:
            continue
    return out


def find_vllm_procs() -> list[psutil.Process]:
    procs = []
    for p in psutil.process_iter(["cmdline"]):
        cmd = " ".join(p.info["cmdline"] or [])
        if "vllm serve" in cmd or "VLLM" in (p.name() or ""):
            procs.append(p)
            procs.extend(p.children(recursive=True))
    uniq = {p.pid: p for p in procs}
    for p in uniq.values():
        try:
            p.cpu_percent(None)  # 기준점
        except psutil.Error:
            pass
    return list(uniq.values())


def gpu_row() -> dict[str, float]:
    if _NVML is None:
        return {}
    u = pynvml.nvmlDeviceGetUtilizationRates(_NVML)
    m = pynvml.nvmlDeviceGetMemoryInfo(_NVML)
    return {
        "gpu_util": u.gpu,
        "gpu_mem_util": u.memory,
        "gpu_mem_used_mb": m.used / 2**20,
        "gpu_power_w": pynvml.nvmlDeviceGetPowerUsage(_NVML) / 1000,
        "gpu_temp_c": pynvml.nvmlDeviceGetTemperature(_NVML, 0),
        "gpu_sm_clock_mhz": pynvml.nvmlDeviceGetClockInfo(_NVML, 0),
    }


def pct(xs, q):
    xs = sorted(x for x in xs if x == x)
    if not xs:
        return float("nan")
    return xs[min(len(xs) - 1, int(q * len(xs)))]


async def collector(session, client, state, path: Path, stop: asyncio.Event):
    procs = find_vllm_procs()
    psutil.cpu_percent(None)
    writer = None
    f = open(path, "w", newline="", encoding="utf-8")
    t_start = time.time()
    next_t = time.perf_counter()
    while not stop.is_set():
        next_t += 1.0
        row: dict[str, object] = {"t": round(time.time() - t_start, 2), "label": state["label"]}
        try:
            async with session.get(f"{URL}/metrics", timeout=aiohttp.ClientTimeout(total=2)) as r:
                row.update(parse_prom(await r.text()))
        except Exception:
            row["scrape_error"] = 1
        row.update(gpu_row())
        row["host_cpu_pct"] = psutil.cpu_percent(None)
        vcpu = 0.0
        for p in procs:
            try:
                vcpu += p.cpu_percent(None)
            except psutil.Error:
                pass
        row["vllm_cpu_pct"] = vcpu
        done, client.done = client.done, []
        lat = [d[0] for d in done if d[2]]
        row["cli_done"] = len(lat)
        row["cli_err"] = sum(1 for d in done if not d[2])
        row["cli_lat_p50"] = pct(lat, 0.5)
        row["cli_lat_p90"] = pct(lat, 0.9)
        row["cli_ttft_p90"] = pct([d[1] for d in done if d[2]], 0.9)
        if writer is None:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()), extrasaction="ignore")
            writer.writeheader()
        writer.writerow(row)
        f.flush()
        await asyncio.sleep(max(0.0, next_t - time.perf_counter()))
    f.close()


# ---------------------------------------------------------------- 장애

class Fault:
    def __init__(self, name, client, seed):
        self.name, self.client, self.seed = name, client, seed
        self.stop = None
        self.tasks = []
        self.procs: list[subprocess.Popen] = []

    def start(self):
        c = self.client
        if self.name == "surge":            # 동시 사용자 8 → 64
            self.stop, self.tasks = spawn_workers(c, 56, 200, 128, self.seed)
        elif self.name == "longctx":        # 긴 입력(≈3,000토큰) 12개 동시 — KV 캐시 압박
            self.stop, self.tasks = spawn_workers(c, 12, 2300, 128, self.seed)
        elif self.name == "longout":        # 긴 출력 1,500토큰 16개 동시 — 디코드 점유
            self.stop, self.tasks = spawn_workers(c, 16, 200, 1500, self.seed)
        elif self.name == "gpu_hog":        # 같은 GPU를 쓰는 다른 프로세스(학습 잡이 끼어든 상황)
            self.procs = [subprocess.Popen([sys.executable, str(HERE / "gpu_hog.py")])]
        elif self.name == "cpu_hog":        # 호스트 CPU 전 코어 포화(배치 잡·로그 압축 등이 끼어든 상황)
            n = (os.cpu_count() or 4) * 2
            self.procs = [subprocess.Popen([sys.executable, "-c", "while True: pass"]) for _ in range(n)]
        else:
            raise ValueError(self.name)

    async def end(self):
        if self.stop is not None:
            self.stop.set()
            # 진행 중 요청(긴 출력)은 끝나게 두면 회복 구간이 오염되므로 취소한다
            for t in self.tasks:
                t.cancel()
            await asyncio.gather(*self.tasks, return_exceptions=True)
        for p in self.procs:
            p.kill()
            p.wait()


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base-users", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=90)
    ap.add_argument("--fault-sec", type=int, default=60)
    ap.add_argument("--gap-sec", type=int, default=90)
    ap.add_argument("--faults", default=",".join(FAULTS))
    args = ap.parse_args()

    out = HERE.parent / "results" / args.run
    out.mkdir(parents=True, exist_ok=True)
    faults = args.faults.split(",")
    rng = random.Random(args.seed)
    schedule = []
    for _ in range(args.reps):
        order = faults[:]
        rng.shuffle(order)
        schedule += order
    (out / "run.json").write_text(json.dumps({**vars(args), "schedule": schedule}, indent=1))

    conn = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=conn, timeout=aiohttp.ClientTimeout(total=180)) as s:
        client = Client(s)
        state = {"label": "normal"}
        cstop = asyncio.Event()
        ctask = asyncio.create_task(collector(s, client, state, out / "metrics.csv", cstop))
        t0 = time.time()
        bstop, btasks = spawn_workers(client, args.base_users, 200, 128, args.seed * 1000)
        ev = open(out / "events.csv", "w", newline="", encoding="utf-8")
        ew = csv.writer(ev)
        ew.writerow(["fault", "start", "end"])
        await asyncio.sleep(args.warmup)
        for i, name in enumerate(schedule):
            f = Fault(name, client, args.seed * 1000 + 100 * (i + 1))
            st = time.time() - t0
            state["label"] = name
            f.start()
            await asyncio.sleep(args.fault_sec)
            await f.end()
            state["label"] = "normal"
            en = time.time() - t0
            ew.writerow([name, round(st, 2), round(en, 2)])
            ev.flush()
            print(f"[{i + 1}/{len(schedule)}] {name} {st:.0f}-{en:.0f}s", flush=True)
            await asyncio.sleep(args.gap_sec)
        bstop.set()
        for t in btasks:
            t.cancel()
        await asyncio.gather(*btasks, return_exceptions=True)
        cstop.set()
        await ctask
        ev.close()


if __name__ == "__main__":
    asyncio.run(main())
