"""低利用率反馈与显存边界；纯CPU逻辑可独立验证。"""
from collections import deque
import math


def checked_config(value):
    result = {'cancel': 60., 'target': 80., 'reserve_gib': 8., 'max_prealloc_mib': 16.,
              'max_duty': .9, **value}
    if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v)
           for v in result.values()):
        raise ValueError('利用率保护参数必须是有限数字')
    if not 0 <= result['cancel'] < result['target'] < 100:
        raise ValueError('保护目标必须高于取消阈值且小于100%')
    if result['reserve_gib'] <= 0 or not 0 < result['max_prealloc_mib'] <= 64:
        raise ValueError('辅助显存必须受限，且保留正数显存余量')
    if not 0 < result['max_duty'] <= .95:
        raise ValueError('辅助计算占空比必须有明确上限')
    return result


class Feedback:
    def __init__(self, **config):
        self.config = checked_config(config)
        self.samples = deque()
        self.sum = self.count = 0
        self.duration = self.weighted = 0.
        self.previous = None
        self.duty = 0.

    def seed(self, samples):
        if not samples: return
        if any(b[0]<a[0] for a,b in zip(samples,samples[1:])):
            raise ValueError('历史采样时间不能倒退')
        self.duration = samples[-1][0]-samples[0][0]
        self.weighted = sum((b[0]-a[0])*a[1] for a,b in zip(samples,samples[1:]))
        self.previous = samples[-1]
        self.samples = deque(samples)
        cutoff = samples[-1][0]-7200
        while len(self.samples)>1 and self.samples[1][0]<=cutoff: self.samples.popleft()

    def update(self, *, now, utilization, free_mib):
        if not 0 <= utilization <= 100 or free_mib < 0:
            raise ValueError('NVML样本无效')
        if self.previous is not None:
            previous_time, previous_util = self.previous
            if now < previous_time:
                raise ValueError('采样时间不能倒退')
            self.duration += now-previous_time
            self.weighted += (now-previous_time)*previous_util
        self.previous = (now, utilization)
        self.samples.append((now, utilization))
        while len(self.samples)>1 and self.samples[1][0] <= now - 7200:
            self.samples.popleft()
        windows = {}
        for seconds in (60,600,1800,3600,7200):
            duration = weighted = 0.
            end = now
            for timestamp, value in reversed(self.samples):
                start = max(timestamp, now-seconds)
                length = max(0.,end-start)
                duration += length
                weighted += length*value
                end = timestamp
                if timestamp <= now-seconds: break
            windows[str(seconds)] = weighted/duration if duration else utilization
        windows['cumulative'] = self.weighted/self.duration if self.duration else utilization
        safe = free_mib >= self.config['reserve_gib']*1024 + self.config['max_prealloc_mib']
        limiting = min(windows.values())
        deficit = self.config['target'] - limiting
        if not safe:
            self.duty = 0.
        elif utilization < self.config['cancel']:
            # 短时下探不能等小时均值下降；仍由显存余量和既有占空比上限约束。
            self.duty = self.config['max_duty']
        elif deficit > 0:
            self.duty = min(self.config['max_duty'], self.duty + max(.05, deficit/100))
        else:
            self.duty = max(0., self.duty - .25)
        return {'duty': self.duty, 'memory_safe': safe, 'windows': windows,
                'allocation_mib': self.config['max_prealloc_mib'] if safe else 0.,
                'target': self.config['target'], 'cancel': self.config['cancel']}
