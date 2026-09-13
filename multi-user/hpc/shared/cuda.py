#!/usr/bin/env python3
"""Dynamic low-utilization CUDA keeper used by long Torch jobs."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from datetime import datetime
import math
import signal
import sys
import threading
import time
from typing import Deque, List, Tuple

try:
    import pynvml as nvml
except Exception:
    nvml = None

try:
    import torch
except Exception:
    print("Please install PyTorch", file=sys.stderr)
    raise

GB = 1024**3


def lowest_stream_priority() -> int:
    """Return the lowest CUDA stream priority across supported Torch builds."""

    get_range = getattr(torch.cuda, "get_stream_priority_range", None)
    if get_range is None:
        # Older Torch wheels do not expose the CUDA priority-range helper.
        # Priority zero is the normal (and lowest on standard CUDA builds)
        # stream priority, so it remains safe for opportunistic keeper work.
        return 0
    least_priority, _ = get_range()
    return int(least_priority)


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
        if nvml is None:
            raise RuntimeError("Please install NVML bindings: pip install nvidia-ml-py3")
        nvml.nvmlInit()
        self.h = nvml.nvmlDeviceGetHandleByIndex(index)

    def util_total(self) -> int:
        return int(nvml.nvmlDeviceGetUtilizationRates(self.h).gpu)

    def mem(self):
        memory = nvml.nvmlDeviceGetMemoryInfo(self.h)
        return memory.total, memory.used, memory.free


class Burner:
    def __init__(
        self,
        device,
        reserve_gb: float,
        max_prealloc_gb: float,
        dtype=torch.bfloat16,
    ) -> None:
        self.device = device
        self.reserve_gb = reserve_gb
        self.max_prealloc_gb = max_prealloc_gb
        self.dtype = dtype
        # CUDA uses numerically larger values for lower-priority streams.  A
        # single low-priority stream limits queued keeper work so real model
        # kernels can take over after at most one matrix multiplication.
        self.stream = torch.cuda.Stream(
            device=device,
            priority=lowest_stream_priority(),
        )
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

    def allocated_gb(self, bytes_per_value: int = 2) -> float:
        if self.size <= 0:
            return 0.0
        return 3.0 * self.size * self.size * bytes_per_value / GB

    def maybe_alloc(self, free_gb: float) -> None:
        # Treat existing keeper tensors as reclaimable.  Without this, their
        # own allocation can push free memory below the reserve and cause an
        # allocate/release loop near the HBM boundary.
        target = self._choose_size(free_gb + self.allocated_gb())
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
    def burn(self, seconds: float) -> None:
        deadline = time.time() + max(0.0, seconds)
        while time.time() < deadline:
            with torch.cuda.stream(self.stream):
                torch.mm(self.left, self.right, out=self.output)
                self.output[:1, :1].add_(1)
            self.stream.synchronize()


@dataclass(frozen=True)
class BurnDecision:
    burn: bool
    reason: str
    pulse_seconds: float = 0.0


def decide_burn(
    *,
    current_total_util: float,
    rolling_average_util: float,
    target_util: float,
    idle_util_threshold: float,
    idle_seconds: float,
    idle_confirm_seconds: float,
    reclaimable_free_gb: float,
    reserve_gb: float,
    max_pulse_seconds: float,
) -> BurnDecision:
    """Return a keeper decision using only current device-wide utilization.

    Process-level NVML utilization is intentionally excluded: on the H200
    serving workload it was delayed/cumulative and reported a busy model while
    device-wide utilization was zero.  The keeper only fills confirmed idle
    gaps and emits short pulses so it cannot monopolize the GPU.
    """

    if reclaimable_free_gb < reserve_gb + 0.2:
        return BurnDecision(False, "memory_guard")
    if current_total_util > idle_util_threshold:
        return BurnDecision(False, "gpu_busy")
    if rolling_average_util >= target_util:
        return BurnDecision(False, "window_target_met")
    if idle_seconds < idle_confirm_seconds:
        return BurnDecision(False, "confirming_idle")
    deficit = max(0.0, target_util - rolling_average_util)
    pulse_seconds = min(max_pulse_seconds, 0.05 + 0.01 * deficit)
    return BurnDecision(True, "confirmed_idle_below_target", pulse_seconds)


class Controller(threading.Thread):
    SAMPLE_INTERVAL = 1.0

    def __init__(
        self,
        gpu_idx: int,
        threshold: int,
        reserve_mem_gb: float,
        max_prealloc_gb: float,
        start_used_mib: int,
        window_seconds: int,
        target_margin: int,
        idle_util_threshold: int,
        idle_confirm_seconds: float,
        max_pulse_seconds: float,
    ) -> None:
        super().__init__(daemon=True)
        self.gpu = gpu_idx
        self.threshold = threshold
        self.reserve = reserve_mem_gb
        self.max_prealloc = max_prealloc_gb
        self.start_used_mib = start_used_mib
        self.window_seconds = window_seconds
        self.target_margin = target_margin
        self.idle_util_threshold = idle_util_threshold
        self.idle_confirm_seconds = idle_confirm_seconds
        self.max_pulse_seconds = max_pulse_seconds
        # Thread.join() calls Thread._stop(); never shadow that internal method.
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        torch.cuda.init()
        device = torch.device(f"cuda:{self.gpu}")
        torch.cuda.set_device(device)
        nv = NV(self.gpu)
        utilization_window = UtilWin(window_s=self.window_seconds)
        burner = None
        last_sample = 0.0
        idle_since = None

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
                )
            utilization_window.add(current, float(total_util))
            average = utilization_window.avg()
            target = self.threshold + self.target_margin
            if total_util <= self.idle_util_threshold:
                idle_since = current if idle_since is None else idle_since
            else:
                idle_since = None
            idle_seconds = 0.0 if idle_since is None else current - idle_since

            free_gb = b2g(free)
            reclaimable_free_gb = free_gb + burner.allocated_gb()
            decision = decide_burn(
                current_total_util=total_util,
                rolling_average_util=average,
                target_util=target,
                idle_util_threshold=self.idle_util_threshold,
                idle_seconds=idle_seconds,
                idle_confirm_seconds=self.idle_confirm_seconds,
                reclaimable_free_gb=reclaimable_free_gb,
                reserve_gb=self.reserve,
                max_pulse_seconds=self.max_pulse_seconds,
            )
            if decision.reason == "memory_guard":
                burner.release()
            elif decision.burn:
                burner.maybe_alloc(free_gb)
                burner.burn(decision.pulse_seconds)
            print(
                f"[{now_hms()}] gpu:{self.gpu} | total:{total_util:3d}% | "
                f"mem:{b2g(used):5.1f}/{b2g(total):.1f} GB | "
                f"avg{self.window_seconds}s:{average:5.1f}% | "
                f"idle:{idle_seconds:4.1f}s | pulse:{decision.pulse_seconds:4.2f}s | "
                f"burning:{'Y' if decision.burn else 'N'} | reason:{decision.reason}",
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
    parser.add_argument("--window-seconds", type=int, default=300)
    parser.add_argument("--target-margin", type=int, default=8)
    parser.add_argument("--idle-util-threshold", type=int, default=10)
    parser.add_argument("--idle-confirm-seconds", type=float, default=1.0)
    parser.add_argument("--max-pulse-seconds", type=float, default=0.2)
    args = parser.parse_args()
    if nvml is None:
        raise SystemExit("Please install NVML bindings: pip install nvidia-ml-py3")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")
    if args.window_seconds <= 0:
        raise SystemExit("--window-seconds must be positive")
    if not 0 <= args.idle_util_threshold < args.threshold <= 100:
        raise SystemExit("utilization values must satisfy 0 <= idle < threshold <= 100")
    if args.target_margin < 0 or args.threshold + args.target_margin > 100:
        raise SystemExit("threshold plus target margin must be between 0 and 100")
    if args.idle_confirm_seconds < 0 or args.max_pulse_seconds <= 0:
        raise SystemExit("idle confirmation must be nonnegative and max pulse positive")
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
            window_seconds=args.window_seconds,
            target_margin=args.target_margin,
            idle_util_threshold=args.idle_util_threshold,
            idle_confirm_seconds=args.idle_confirm_seconds,
            max_pulse_seconds=args.max_pulse_seconds,
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
