#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, time, math, os, sys, threading, signal
from collections import deque
from datetime import datetime
from typing import Deque, Tuple, Optional, List

# Deps: pip install nvidia-ml-py3 torch psutil
try:
    import pynvml as nvml
except Exception:
    print("Please install NVML bindings: pip install nvidia-ml-py3", file=sys.stderr); raise
try:
    import torch
except Exception:
    print("Please install PyTorch: pip install torch", file=sys.stderr); raise
try:
    import psutil
except Exception:
    psutil = None

GB = 1024**3

def now_hms(): return datetime.now().strftime("%H:%M:%S")
def b2g(x): return x / GB

# ---- 2h time-weighted sliding average ----
class UtilWin:
    def __init__(self, window_s=7200):
        self.W = window_s
        self.q: Deque[Tuple[float, float]] = deque()
    def add(self, ts, util):
        self.q.append((ts, util))
        cutoff = ts - self.W
        while len(self.q) > 1 and self.q[0][0] < cutoff and self.q[1][0] <= cutoff:
            self.q.popleft()
    def avg(self) -> float:
        if not self.q: return 0.0
        now = time.time(); cutoff = now - self.W
        total_t, weighted = 0.0, 0.0
        prev = now
        for t, u in reversed(self.q):
            seg_start = max(t, cutoff)
            dt = max(0.0, prev - seg_start)
            weighted += u * dt; total_t += dt; prev = t
            if t <= cutoff: break
        return (weighted/total_t) if total_t>0 else self.q[-1][1]

# ---- NVML wrapper ----
class NV:
    def __init__(self, idx:int):
        nvml.nvmlInit()
        self.h = nvml.nvmlDeviceGetHandleByIndex(idx)
        self._last_proc_ts = 0
    def util_total(self) -> int:
        u = nvml.nvmlDeviceGetUtilizationRates(self.h)
        return int(u.gpu)
    def mem(self):
        m = nvml.nvmlDeviceGetMemoryInfo(self.h)
        return m.total, m.used, m.free
    def procs(self):
        out = []
        for fn in (
            getattr(nvml, "nvmlDeviceGetComputeRunningProcesses_v3", None),
            getattr(nvml, "nvmlDeviceGetComputeRunningProcesses_v2", None),
            getattr(nvml, "nvmlDeviceGetComputeRunningProcesses", None),
        ):
            if not fn: continue
            try:
                arr = fn(self.h)
                for p in arr:
                    out.append({"pid": int(p.pid), "mem": int(getattr(p, "usedGpuMemory", 0))})
                break
            except nvml.NVMLError:
                continue
        if psutil:
            for p in out:
                try: p["name"] = psutil.Process(p["pid"]).name()
                except Exception: p["name"] = "unknown"
        else:
            for p in out: p["name"] = "unknown"
        return out
    def pick_main_pid(self, exclude_pid:int) -> Optional[int]:
        cand = [p for p in self.procs() if p["pid"] != exclude_pid]
        if not cand: return None
        return max(cand, key=lambda x: x["mem"])["pid"]
    def proc_util(self, pid:int) -> Optional[int]:
        try:
            nowms = int(time.time()*1000)
            arr = nvml.nvmlDeviceGetProcessUtilization(self.h, self._last_proc_ts)
            self._last_proc_ts = nowms
            for s in arr:
                if int(s.pid)==int(pid):
                    sm = getattr(s, "smUtil", None)
                    gpu = getattr(s, "gpuUtilization", None)
                    val = sm if sm is not None else gpu
                    if val is not None: return int(val)
        except Exception: pass
        try:
            st = nvml.nvmlDeviceGetAccountingStats(self.h, pid)
            return int(st.gpuUtilization)
        except Exception: pass
        return None

# ---- Burner: high-SM load with low VRAM ----
class Burner:
    def __init__(self, device, reserve_gb:float, max_prealloc_gb:float=2.5,
                 dtype=torch.bfloat16, streams:int=8):
        self.dev = device
        self.reserve = reserve_gb
        self.max_pre = max_prealloc_gb
        self.dtype = dtype
        self.streams = [torch.cuda.Stream(device=device) for _ in range(streams)]
        self.N = 0; self.A=self.B=self.C=None
        torch.backends.cuda.matmul.allow_tf32 = True
        try: torch.set_float32_matmul_precision('high')
        except Exception: pass
    def _choose_N(self, free_gb:float, bytes_per:int=2):
        budget = max(0.0, min(self.max_pre, free_gb - self.reserve))
        if budget <= 0.1: return 2048
        N = int(math.sqrt(max(1.0, budget*GB / (3.0*bytes_per))))  # 3*N^2*bytes ≈ budget
        return max(1024, (N//1024)*1024)
    def maybe_alloc(self, free_gb:float):
        bytes_per = 2  # bf16
        tgt = self._choose_N(free_gb, bytes_per)
        if tgt == self.N and self.A is not None: return
        torch.cuda.empty_cache()
        self.N = tgt
        with torch.cuda.device(self.dev):
            self.A = torch.randn((self.N, self.N), device=self.dev, dtype=self.dtype)
            self.B = torch.randn((self.N, self.N), device=self.dev, dtype=self.dtype)
            self.C = torch.empty((self.N, self.N), device=self.dev, dtype=self.dtype)
    @torch.inference_mode()
    def burn(self, seconds:float, intensity:int=1):
        end = time.time()+max(0.0, seconds)
        calls = max(1, int(intensity))
        while time.time()<end:
            for s in self.streams:
                with torch.cuda.stream(s):
                    n=self.N
                    for _ in range(calls):
                        torch.mm(self.A[:n,:n], self.B[:n,:n], out=self.C[:n,:n])
                        self.C[:1,:1].add_(1)
            torch.cuda.synchronize(self.dev)

# ---- Per-GPU controller thread ----
class Controller(threading.Thread):
    # 常量内置：尽量少参数
    CONTROL_PERIOD = 5.0
    SAMPLE_INTERVAL = 1.0
    KP = 0.06
    UPPER_SLACK = 2
    INTENSITY_MAX = 8

    def __init__(self, gpu_idx:int, threshold:int, reserve_mem_gb:float):
        super().__init__(daemon=True)
        self.gpu = gpu_idx
        self.threshold = threshold
        self.target_margin = 8  # 固定额外边际（pp）
        self.reserve = reserve_mem_gb
        self._stop = threading.Event()

    def stop(self): self._stop.set()

    def run(self):
        torch.cuda.init()
        dev = torch.device(f"cuda:{self.gpu}")
        torch.cuda.set_device(dev)

        nv = NV(self.gpu)
        win = UtilWin(window_s=7200)

        duty = 0.0
        burner = Burner(dev, reserve_gb=self.reserve,
                        max_prealloc_gb=2.5, dtype=torch.bfloat16, streams=8)

        self_pid = os.getpid()
        main_pid = None
        last_pick = 0.0
        last_ctrl = 0.0
        last_samp = 0.0

        while not self._stop.is_set():
            now = time.time()

            if now - last_samp >= self.SAMPLE_INTERVAL:
                last_samp = now

                total_util = nv.util_total()
                total, used, free = nv.mem()
                win.add(now, float(total_util))

                if (now - last_pick > 5.0) or (main_pid is None):
                    main_pid = nv.pick_main_pid(exclude_pid=self_pid)
                    last_pick = now
                main_util = nv.proc_util(main_pid) if main_pid is not None else None

                avg2h = win.avg()
                target = self.threshold + self.target_margin
                stop_on_avg = (avg2h >= (target + self.UPPER_SLACK))
                main_busy = (main_util is not None and main_util >= self.threshold)

                burn_now = False
                burn_sec = 0.0
                intensity = 1

                if now - last_ctrl >= self.CONTROL_PERIOD:
                    last_ctrl = now
                    if not (main_busy or stop_on_avg):
                        deficit = max(0.0, target - avg2h)
                        if avg2h < 0.6 * target:
                            duty = 0.9
                        else:
                            duty = max(0.0, min(1.0, duty + self.KP * deficit))
                        burn_sec = self.CONTROL_PERIOD * duty
                        intensity = 1 + int(min(self.INTENSITY_MAX-1, (deficit + 2)//3))
                        burn_now = burn_sec > 0.02
                    else:
                        duty = max(0.0, duty - 0.25)

                free_gb = b2g(free)
                if free_gb < self.reserve + 0.2:
                    burn_now = False; duty = 0.0

                burner.maybe_alloc(free_gb)
                if burn_now: burner.burn(burn_sec, intensity=intensity)

                used_gb = b2g(used); total_gb = b2g(total)
                main_str = ("{:3d}%".format(main_util) if main_util is not None else "N/A")
                print(
                    f"[{now_hms()}] gpu:{self.gpu} | main:{main_str} | total:{total_util:3d}% | "
                    f"mem:{used_gb:5.1f}/{total_gb:.1f} GB | avg2h:{avg2h:5.1f}% | "
                    f"duty:{duty:4.2f} | burning:{'Y' if burn_now else 'N'} | "
                    f"main_pid:{main_pid if main_pid else 'N/A'}"
                )
                sys.stdout.flush()

            time.sleep(0.05)

# ---- utils ----
def parse_gpus_arg(arg: str, max_count: int) -> List[int]:
    arg = arg.strip().lower()
    if arg == "all":
        return list(range(max_count))
    idxs = []
    for tok in arg.split(","):
        tok = tok.strip()
        if not tok: continue
        i = int(tok)
        if i < 0 or i >= max_count:
            raise ValueError(f"GPU index {i} out of range [0, {max_count-1}]")
        idxs.append(i)
    seen = set(); out=[]
    for i in idxs:
        if i not in seen:
            seen.add(i); out.append(i)
    return out

def parse_reserve_arg(arg: str, n: int) -> List[float]:
    arg = arg.strip()
    if "," not in arg:
        v = float(arg)
        return [v]*n
    vals = [float(x.strip()) for x in arg.split(",") if x.strip()]
    if len(vals) != n:
        raise ValueError(f"--reserve expects 1 value or {n} values (matched to selected GPUs). Got {len(vals)}.")
    return vals

# ---- main ----
def main():
    ap = argparse.ArgumentParser(
        description="Dynamic GPU util keeper (multi-device, minimal flags)"
    )
    ap.add_argument("--threshold", type=int, default=70, help="Util threshold (%)")
    ap.add_argument("--gpus", type=str, default="all", help="GPU list: 'all' or '0,2,3'")
    ap.add_argument("--reserve", type=str, default="5.0",
                    help="GB to keep free per GPU. Single value or comma list matching --gpus.")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available.", file=sys.stderr); sys.exit(1)

    dev_count = torch.cuda.device_count()
    gpu_list = parse_gpus_arg(args.gpus, dev_count)
    if not gpu_list:
        print("No GPUs selected.", file=sys.stderr); sys.exit(1)

    reserves = parse_reserve_arg(args.reserve, len(gpu_list))

    controllers: List[Controller] = []
    for g, r in zip(gpu_list, reserves):
        c = Controller(gpu_idx=g, threshold=args.threshold, reserve_mem_gb=r)
        c.start()
        controllers.append(c)

    stop_event = threading.Event()
    def handle_sig(sig, frame):
        print("\nStopping controllers...", file=sys.stderr)
        stop_event.set()
        for c in controllers: c.stop()
    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    try:
        while not stop_event.is_set():
            time.sleep(0.5)
    finally:
        for c in controllers: c.stop()
        for c in controllers: c.join(timeout=5.0)

if __name__ == "__main__":
    main()
