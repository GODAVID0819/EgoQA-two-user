#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import signal
import time
from pathlib import Path

import psutil


_RUNNING = True


def _stop(_signum, _frame):
    global _RUNNING
    _RUNNING = False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--interval', type=float, default=0.5)
    args = parser.parse_args()
    if args.interval <= 0:
        raise ValueError('interval must be positive')

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Prime psutil's percentage calculation.
    psutil.cpu_percent(interval=None)
    with args.output.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow([
            'wall_time_epoch_s',
            'monotonic_ns',
            'cpu_busy_pct',
            'cpu_idle_pct',
            'ram_used_gib',
            'ram_available_gib',
            'load1',
            'load5',
            'load15',
        ])
        handle.flush()
        while _RUNNING:
            busy = psutil.cpu_percent(interval=args.interval)
            vm = psutil.virtual_memory()
            load1, load5, load15 = psutil.getloadavg()
            writer.writerow([
                f'{time.time():.6f}',
                time.monotonic_ns(),
                f'{busy:.3f}',
                f'{100.0 - busy:.3f}',
                f'{vm.used / 1024**3:.6f}',
                f'{vm.available / 1024**3:.6f}',
                f'{load1:.4f}',
                f'{load5:.4f}',
                f'{load15:.4f}',
            ])
            handle.flush()


if __name__ == '__main__':
    main()
