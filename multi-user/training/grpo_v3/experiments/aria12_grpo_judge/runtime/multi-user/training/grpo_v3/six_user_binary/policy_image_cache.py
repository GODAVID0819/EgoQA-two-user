"""为同进程的 Policy 编码复用缩放后图像；缓存不改变原模板的后续处理。"""
from collections import OrderedDict
from functools import wraps
import os
import math
from pathlib import Path
import threading
import time


_active_cache = None


def configured_cache_bytes(gigabytes):
    if gigabytes is None:
        return 0
    if isinstance(gigabytes, bool):
        raise ValueError('图像缓存预算不能是布尔值')
    value = float(gigabytes)
    if not math.isfinite(value) or value < 0:
        raise ValueError('图像缓存预算必须为有限非负数')
    return int(value * 1024**3)


def enable_from_environment():
    size = configured_cache_bytes(os.environ.get('EGOQA_POLICY_IMAGE_CACHE_GB'))
    return install_policy_image_cache(max_bytes=size) if size else None


class PolicyImageCache:
    def __init__(self, max_bytes):
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
            raise ValueError('图像缓存字节预算必须为正整数')
        self.max_bytes = max_bytes
        self.items = OrderedDict()
        self.bytes = self.hits = self.misses = 0
        self.decode_resize_seconds = 0.
        self.lock = threading.RLock()

    def get(self, path, template, resize):
        info = Path(path).stat()
        key = (path, info.st_mtime_ns, info.st_size, template.max_pixels, type(template))
        with self.lock:
            if key in self.items:
                self.hits += 1
                image, _ = self.items[key]
                self.items.move_to_end(key)
                return image.copy()
            self.misses += 1
            started = time.perf_counter()
            image = resize(template._load_image(path, True), template.max_pixels)
            self.decode_resize_seconds += time.perf_counter() - started
            size = len(image.tobytes())
            if size <= self.max_bytes:
                while self.items and self.bytes + size > self.max_bytes:
                    _, (_, removed_size) = self.items.popitem(last=False)
                    self.bytes -= removed_size
                self.items[key] = (image, size)
                self.bytes += size
            # 后续模板或调用者可修改返回值，不能污染下一候选的缓存。
            return image.copy()

    def snapshot(self):
        with self.lock:
            return {'version': 2, 'hits': self.hits, 'misses': self.misses, 'bytes': self.bytes,
                    'max_bytes': self.max_bytes, 'items': len(self.items),
                    'decode_resize_seconds': self.decode_resize_seconds}


def install_policy_image_cache(*, max_bytes, template_cls=None, resize_fn=None):
    """只处理绝对路径图片；对象坐标和保持路径的模板仍走原实现。"""
    global _active_cache
    if template_cls is None:
        from swift.template.base import Template
        template_cls = Template
    if resize_fn is None:
        from swift.template.vision_utils import rescale_image
        resize_fn = rescale_image
    original = template_cls._preprocess_inputs
    previous = getattr(original, '_egoqa_policy_image_cache', None)
    if previous is not None:
        if previous.max_bytes != max_bytes:
            raise ValueError('同一进程不能混用不同的图像缓存预算')
        _active_cache = previous
        return previous
    cache = PolicyImageCache(max_bytes)

    @wraps(original)
    def preprocess(self, inputs):
        images = inputs.images
        # ms-swift 的数据加载器把文件路径规范化为 bytes/path 字典。
        # 有嵌入字节时必须继续读取那些字节，不能误用同字典中的文件路径。
        paths = [(image.get('path') if isinstance(image, dict) and 'bytes' in image and not image['bytes'] else image)
                 for image in images] if images else []
        keep_images = self.load_images or self.mode in {'vllm', 'lmdeploy'}
        if (images and keep_images and self.max_pixels and not inputs.objects
                and all(isinstance(path, str) and os.path.isabs(path) for path in paths)):
            inputs.images = [cache.get(path, self, resize_fn) for path in paths]
        return original(self, inputs)

    preprocess._egoqa_policy_image_cache = cache
    template_cls._preprocess_inputs = preprocess
    _active_cache = cache
    return cache


def policy_cache_metrics():
    return _active_cache.snapshot() if _active_cache is not None else None
