"""启动期临时保护交接到训练进程，避免额外常驻CUDA上下文。"""
import argparse
import atexit
import json
import os
from pathlib import Path
import signal
import threading
import time
from .utilization_guard import Feedback, checked_config

_active = None
REVISION = 'pointwise-2-instant'


def auxiliary_operands(torch, device):
    """两个小缓冲区共1.125MiB，避开矩阵库的额外工作区。"""
    source = torch.ones((384,384), device=device, dtype=torch.float32)
    return source, torch.empty_like(source)


def auxiliary_step(torch, operands):
    torch.sin(operands[0],out=operands[1])


class Guard:
    def __init__(self, config, log, *, stop_file=None):
        self.config = checked_config(config)
        self.log = Path(log)
        self.stop_file = Path(stop_file) if stop_file else None
        self.stop = threading.Event()
        self.ready = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.error = None
        self.graph_live = False

    def record(self, **values):
        self.log.parent.mkdir(parents=True, exist_ok=True)
        with self.log.open('a', encoding='utf-8') as f:
            f.write(json.dumps({'epoch': time.time(), 'pid': os.getpid(), 'revision':REVISION, **values})+'\n')

    def start(self):
        self.thread.start()
        if not self.ready.wait(120):
            raise RuntimeError('利用率保护启动未返回状态')
        if self.error:
            raise RuntimeError('利用率保护启动失败：'+self.error)
        return self

    def close(self):
        self.stop.set()
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            raise RuntimeError('利用率保护未停止，禁止开始下一次捕获')

    def run(self):
        left = right = output = graph = None
        try:
            import torch
            import pynvml as nv
            from hpc.shared.cuda_device_identity import handle_for_cuda_device
            torch.cuda.set_device(0)
            nv.nvmlInit()
            handle = handle_for_cuda_device(0, torch.cuda, nv)
            stream = torch.cuda.Stream(device=0, priority=0)
            stream.wait_stream(torch.cuda.default_stream(0))
            feedback = Feedback(**self.config)
            history = os.environ.get('EGOQA_UTILIZATION_METRICS')
            if history and Path(history).is_file():
                import csv
                from datetime import datetime
                samples = []
                with Path(history).open() as f:
                    for row in csv.DictReader(f):
                        timestamp = datetime.strptime(row['timestamp'],'%Y/%m/%d %H:%M:%S.%f').timestamp()
                        value = float(row[' utilization.gpu [%]'].strip().split()[0])
                        samples.append((timestamp,value))
                feedback.seed(samples)
            def allocate(capture=False):
                # 捕获只在start()阻塞主线程时进行；运行期间绝不并发捕获图。
                with torch.cuda.stream(stream), torch.inference_mode():
                    a,c = auxiliary_operands(torch,'cuda')
                    b = None
                    for _ in range(3): auxiliary_step(torch,(a,c))
                    stream.synchronize()
                    g = None
                    if capture:
                        g = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(g, stream=stream):
                            for _ in range(128): auxiliary_step(torch,(a,c))
                return a,b,c,g
            mem = nv.nvmlDeviceGetMemoryInfo(handle)
            if mem.free/1024**2 >= self.config['reserve_gib']*1024+self.config['max_prealloc_mib']:
                before = torch.cuda.memory_allocated()
                reserved_before = torch.cuda.memory_reserved()
                left,right,output,graph = allocate(capture=True)
                allocated = torch.cuda.memory_allocated()-before
                reserved = torch.cuda.memory_reserved()-reserved_before
                self.record(event='auxiliary_allocated', allocated_mib=allocated/1024**2,
                            reserved_mib=reserved/1024**2)
                if max(allocated,reserved) > self.config['max_prealloc_mib']*1024**2:
                    raise RuntimeError(f'辅助图分配超过预算：allocated={allocated/1024**2:.3f}MiB '
                                       f'reserved={reserved/1024**2:.3f}MiB max={self.config["max_prealloc_mib"]}MiB')
                self.graph_live = True
            self.record(event='monitor_ready', config=self.config, stream_priority=stream.priority)
            self.ready.set()
            last_sample = 0.
            action = {'duty': 0., 'memory_safe': False}
            while not self.stop.is_set() and not (self.stop_file and self.stop_file.exists()):
                now = time.time()
                if now-last_sample >= 1:
                    last_sample = now
                    mem = nv.nvmlDeviceGetMemoryInfo(handle)
                    utilization = nv.nvmlDeviceGetUtilizationRates(handle).gpu
                    action = feedback.update(now=now, utilization=utilization, free_mib=mem.free/1024**2)
                    self.record(event='feedback', total_utilization=utilization,
                                used_mib=mem.used/1024**2, free_mib=mem.free/1024**2,
                                graph_live=self.graph_live, **action)
                if not action['memory_safe']:
                    if left is not None:
                        stream.synchronize()
                        left = right = output = graph = None
                        self.graph_live = False
                        torch.cuda.empty_cache()
                        self.record(event='memory_yield')
                    self.stop.wait(.05)
                    continue
                if action['duty'] <= 0:
                    self.stop.wait(.05)
                    continue
                if left is None:
                    left,right,output,graph = allocate(capture=False)
                started = time.monotonic()
                deadline = started+.05*action['duty']
                while time.monotonic() < deadline and not self.stop.is_set():
                    # 每个图只提交有限工作；只等待自己的低优先级流。
                    with torch.cuda.stream(stream), torch.inference_mode():
                        if graph is not None: graph.replay()
                        else:
                            for _ in range(32): auxiliary_step(torch,(left,output))
                    stream.synchronize()
                self.stop.wait(max(0., .05-(time.monotonic()-started)))
        except BaseException as exc:
            self.error = f'{type(exc).__name__}: {exc}'
            self.record(event='failed', error=self.error)
            self.ready.set()
        finally:
            if left is not None:
                stream.synchronize()
            left = right = output = graph = None
            self.graph_live = False
            self.record(event='stopped')


def enable_training_guard():
    global _active
    if not os.environ.get('EGOQA_UTILIZATION_GUARD'):
        return None
    if _active is not None:
        return _active
    handoff = Path(os.environ['EGOQA_UTILIZATION_HANDOFF'])
    handoff.write_text('trainer_ready\n')
    stopped = Path(str(handoff)+'.stopped')
    deadline = time.monotonic()+120
    while not stopped.exists():
        if time.monotonic() >= deadline:
            raise RuntimeError('启动期利用率保护未释放CUDA上下文，禁止显存重叠')
        time.sleep(.1)
    status = json.loads(stopped.read_text())
    if status['error']:
        raise RuntimeError('启动保护失败：'+status['error'])
    import torch
    import pynvml as nv
    from hpc.shared.cuda_device_identity import handle_for_cuda_device
    nv.nvmlInit()
    handle = handle_for_cuda_device(0,torch.cuda,nv)
    while any(p.pid==status['pid'] for p in nv.nvmlDeviceGetComputeRunningProcesses(handle)):
        if time.monotonic()>=deadline:
            raise RuntimeError('启动保护的CUDA上下文尚未释放')
        time.sleep(.1)
    _active = Guard(json.loads(os.environ['EGOQA_UTILIZATION_GUARD']),
                    os.environ['EGOQA_UTILIZATION_LOG']).start()
    atexit.register(_active.close)
    return _active


def rearm_training_guard():
    """只在主线程已卸载并同步Policy、Judge尚未唤醒的边界重建辅助图。"""
    global _active
    if _active is None or _active.graph_live:
        return
    if _active.error:
        raise RuntimeError('利用率保护异常：'+_active.error)
    previous = _active
    previous.close()
    _active = Guard(previous.config, previous.log).start()
    atexit.register(_active.close)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--log', required=True)
    parser.add_argument('--handoff', required=True)
    args = parser.parse_args()
    guard = Guard(json.loads(args.config), args.log, stop_file=args.handoff).start()
    signal.signal(signal.SIGTERM, lambda *_: guard.stop.set())
    signal.signal(signal.SIGINT, lambda *_: guard.stop.set())
    while guard.thread.is_alive(): time.sleep(.1)
    guard.close()
    # 此标记只在进程退出前发出；训练侧还需等NVML确认临时PID释放。
    Path(args.handoff+'.stopped').write_text(json.dumps({'pid':os.getpid(),'error':guard.error}))
    if guard.error: raise RuntimeError(guard.error)


if __name__ == '__main__': main()
