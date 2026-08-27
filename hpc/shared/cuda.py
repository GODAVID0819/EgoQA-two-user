#!/usr/bin/env python3
"""Dynamic low-utilization CUDA keeper used by long Torch jobs."""

from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime
import math
import os
import signal
import sys
import threading
import time
from typing import Deque, List, Optional, Tuple

try:
    import pynvml as nvml
except Exception:
    print("Please install NVML bindings: pip install nvidia-ml-py3", file=sys.stderr)
    raise

try:
    import torch
except Exception:
    print("Please install PyTorch", file=sys.stderr)
    raise

try:
    import psutil
except Exception:
    psutil = None


GB = 1024**3


def now_hms() -> str:
    return datetime.now().strftime("%H:%M:%S")


def b2g(value: int) -> float:
    return value / GB


class UtilWin:
    def __init__(self, window_s: int = 7200) -> None:
        self.window_s = window_s
        self.samples: Deque[Tuple[float, float]] = deque()

    def add(self, timestamp: float, utilization: float) -> None:
        self.samples.append((timestamp, utilization))
        cutoff = timestamp - self.window_s
        while (
            len(self.samples) > 1
            and self.samples[0][0] < cutoff
            and self.samples[1][0] <= cutoff
        ):
            self.samples.popleft()

    def avg(self) -> float:
        if not self.samples:
            return 0.0
        current = time.time()
        cutoff = current - self.window_s
        total_time = 0.0
        weighted = 0.0
        previous = current
        for timestamp, utilization in reversed(self.samples):
            segment_start = max(timestamp, cutoff)
            duration = max(0.0, previous - segment_start)
            weighted += utilization * duration
            total_time += duration
            previous = timestamp
            if timestamp <= cutoff:
                break
        return weighted / total_time if total_time > 0 else self.samples[-1][1]


class NV:
    def __init__(self, index: int) -> None:
        nvml.nvmlInit()
        self.h = nvml.nvmlDeviceGetHandleByIndex(index)
        self._last_proc_ts = 0

    def util_total(self) -> int:
        return int(nvml.nvmlDeviceGetUtilizationRates(self.h).gpu)

    def mem(self):
        memory = nvml.nvmlDeviceGetMemoryInfo(self.h)
        return memory.total, memory.used, memory.free

    def procs(self):
        processes = []
        for function in (
            getattr(nvml, "nvmlDeviceGetComputeRunningProcesses_v3", None),
            getattr(nvml, "nvmlDeviceGetComputeRunningProcesses_v2", None),
            getattr(nvml, "nvmlDeviceGetComputeRunningProcesses", None),
        ):
            if function is None:
                continue
            try:
                for process in function(self.h):
                    processes.append(
                        {"pid": int(process.pid), "mem": int(getattr(process, "usedGpuMemory", 0))}
                    )
                break
            except nvml.NVMLError:
                continue
        for process in processes:
            if psutil is None:
                process["name"] = "unknown"
                continue
            try:
                process["name"] = psutil.Process(process["pid"]).name()
            except Exception:
                process["name"] = "unknown"
        return processes

    def pick_main_pid(self, exclude_pid: int) -> Optional[int]:
        candidates = [process for process in self.procs() if process["pid"] != exclude_pid]
        if not candidates:
            return None
        return max(candidates, key=lambda process: process["mem"])["pid"]

    def proc_util(self, pid: int) -> Optional[int]:
        try:
            now_ms = int(time.time() * 1000)
            samples = nvml.nvmlDeviceGetProcessUtilization(self.h, self._last_proc_ts)
            self._last_proc_ts = now_ms
            for sample in samples:
                if int(sample.pid) == int(pid):
                    value = getattr(sample, "smUtil", None)
                    if value is None:
                        value = getattr(sample, "gpuUtilization", None)
                    if value is not None:
                        return int(value)
        except Exception:
            pass
        try:
            return int(nvml.nvmlDeviceGetAccountingStats(self.h, pid).gpuUtilization)
        except Exception:
            return None


class Burner:
    def __init__(
        self,
        device,
        reserve_gb: float,
        max_prealloc_gb: float,
        dtype=torch.bfloat16,
        streams: int = 8,
    ) -> None:
        self.device = device
        self.reserve_gb = reserve_gb
        self.max_prealloc_gb = max_prealloc_gb
        self.dtype = dtype
        self.streams = [torch.cuda.Stream(device=device) for _ in range(streams)]
        self.size = 0
        self.left = None
        self.right = None
        self.output = None
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    def _choose_size(self, free_gb: float, bytes_per_value: int = 2) -> int:
        budget = max(0.0, min(self.max_prealloc_gb, free_gb - self.reserve_gb))
        if budget <= 0.1:
            return 1024
        size = int(math.sqrt(max(1.0, budget * GB / (3.0 * bytes_per_value))))
        return max(1024, (size // 1024) * 1024)

    def maybe_alloc(self, free_gb: float) -> None:
        target = self._choose_size(free_gb)
        if target == self.size and self.left is not None:
            return
        torch.cuda.empty_cache()
        self.size = target
        with torch.cuda.device(self.device):
            self.left = torch.randn((target, target), device=self.device, dtype=self.dtype)
            self.right = torch.randn((target, target), device=self.device, dtype=self.dtype)
            self.output = torch.empty((target, target), device=self.device, dtype=self.dtype)

    def release(self) -> None:
        self.left = None
        self.right = None
        self.output = None
        self.size = 0
        torch.cuda.empty_cache()

    @torch.inference_mode()
    def burn(self, seconds: float, intensity: int = 1) -> None:
        deadline = time.time() + max(0.0, seconds)
        calls = max(1, int(intensity))
        while time.time() < deadline:
            for stream in self.streams:
                with torch.cuda.stream(stream):
                    for _ in range(calls):
                        torch.mm(self.left, self.right, out=self.output)
                        self.output[:1, :1].add_(1)
            torch.cuda.synchronize(self.device)


class Controller(threading.Thread):
    CONTROL_PERIOD = 5.0
    SAMPLE_INTERVAL = 1.0
    KP = 0.06
    UPPER_SLACK = 2
    INTENSITY_MAX = 8

    def __init__(
        self,
        gpu_idx: int,
        threshold: int,
        reserve_mem_gb: float,
        max_prealloc_gb: float,
        start_used_mib: int,
    ) -> None:
        super().__init__(daemon=True)
        self.gpu = gpu_idx
        self.threshold = threshold
        self.target_margin = 8
        self.reserve = reserve_mem_gb
        self.max_prealloc = max_prealloc_gb
        self.start_used_mib = start_used_mib
        # Thread.join() calls Thread._stop(); never shadow that internal method.
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        torch.cuda.init()
        device = torch.device(f"cuda:{self.gpu}")
        torch.cuda.set_device(device)
        nv = NV(self.gpu)
        utilization_window = UtilWin(window_s=7200)
        duty = 0.0
        burner = None
        self_pid = os.getpid()
        main_pid = None
        last_pick = 0.0
        last_control = 0.0
        last_sample = 0.0

        while not self._stop_event.is_set():
            current = time.time()
            if current - last_sample < self.SAMPLE_INTERVAL:
                time.sleep(0.05)
                continue
            last_sample = current
            total_util = nv.util_total()
            total, used, free = nv.mem()
            used_mib = used // (1024**2)
            if used_mib < self.start_used_mib:
                print(
                    f"[{now_hms()}] gpu:{self.gpu} | waiting_for_model_memory "
                    f"used:{used_mib} MiB < start:{self.start_used_mib} MiB",
                    flush=True,
                )
                time.sleep(self.SAMPLE_INTERVAL)
                continue
            if burner is None:
                burner = Burner(
                    device,
                    reserve_gb=self.reserve,
                    max_prealloc_gb=self.max_prealloc,
                    dtype=torch.bfloat16,
                    streams=8,
                )
            utilization_window.add(current, float(total_util))
            if current - last_pick > 5.0 or main_pid is None:
                main_pid = nv.pick_main_pid(exclude_pid=self_pid)
                last_pick = current
            main_util = nv.proc_util(main_pid) if main_pid is not None else None
            average = utilization_window.avg()
            target = self.threshold + self.target_margin
            stop_on_average = average >= target + self.UPPER_SLACK
            main_busy = main_util is not None and main_util >= self.threshold
            burn_now = False
            burn_seconds = 0.0
            intensity = 1

            if current - last_control >= self.CONTROL_PERIOD:
                last_control = current
                if not (main_busy or stop_on_average):
                    deficit = max(0.0, target - average)
                    duty = 0.9 if average < 0.6 * target else max(0.0, min(1.0, duty + self.KP * deficit))
                    burn_seconds = self.CONTROL_PERIOD * duty
                    intensity = 1 + int(min(self.INTENSITY_MAX - 1, (deficit + 2) // 3))
                    burn_now = burn_seconds > 0.02
                else:
                    duty = max(0.0, duty - 0.25)

            free_gb = b2g(free)
            if free_gb < self.reserve + 0.2:
                burn_now = False
                duty = 0.0
                burner.release()
            else:
                burner.maybe_alloc(free_gb)
            if burn_now:
                burner.burn(burn_seconds, intensity=intensity)
            main_text = f"{main_util:3d}%" if main_util is not None else "N/A"
            print(
                f"[{now_hms()}] gpu:{self.gpu} | main:{main_text} | total:{total_util:3d}% | "
                f"mem:{b2g(used):5.1f}/{b2g(total):.1f} GB | avg2h:{average:5.1f}% | "
                f"duty:{duty:4.2f} | burning:{'Y' if burn_now else 'N'} | "
                f"main_pid:{main_pid if main_pid else 'N/A'}",
                flush=True,
            )


def parse_gpus_arg(value: str, maximum: int) -> List[int]:
    value = value.strip().lower()
    if value == "all":
        return list(range(maximum))
    indices = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        index = int(token)
        if index < 0 or index >= maximum:
            raise ValueError(f"GPU index {index} out of range [0, {maximum - 1}]")
        if index not in indices:
            indices.append(index)
    return indices


def parse_reserve_arg(value: str, count: int) -> List[float]:
    value = value.strip()
    if "," not in value:
        return [float(value)] * count
    values = [float(token.strip()) for token in value.split(",") if token.strip()]
    if len(values) != count:
        raise ValueError(f"--reserve expects 1 value or {count} values; got {len(values)}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=int, default=70)
    parser.add_argument("--gpus", default="all")
    parser.add_argument("--reserve", default="5.0")
    parser.add_argument("--max-prealloc", type=float, default=2.5)
    parser.add_argument("--start-used-mib", type=int, default=8192)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")
    gpu_list = parse_gpus_arg(args.gpus, torch.cuda.device_count())
    if not gpu_list:
        raise SystemExit("No GPUs selected")
    reserves = parse_reserve_arg(args.reserve, len(gpu_list))
    controllers = [
        Controller(
            gpu,
            args.threshold,
            reserve,
            max_prealloc_gb=args.max_prealloc,
            start_used_mib=args.start_used_mib,
        )
        for gpu, reserve in zip(gpu_list, reserves)
    ]
    for controller in controllers:
        controller.start()
    stop_event = threading.Event()

    def handle_signal(_signal, _frame) -> None:
        print("Stopping controllers...", file=sys.stderr)
        stop_event.set()
        for controller in controllers:
            controller.stop()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    try:
        while not stop_event.is_set():
            dead = [controller.gpu for controller in controllers if not controller.is_alive()]
            if dead:
                raise RuntimeError(f"CUDA keeper controller stopped unexpectedly: gpus={dead}")
            time.sleep(0.5)
    finally:
        for controller in controllers:
            controller.stop()
        for controller in controllers:
            controller.join(timeout=5.0)


if __name__ == "__main__":
    main()
