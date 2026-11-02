"""用 CUDA 实际可见设备身份定位 NVML，兼容 Slurm GPU 重编号。"""
import uuid


def handle_for_cuda_device(index, cuda, nvml):
    value = getattr(cuda.get_device_properties(index), "uuid", None)
    if value is None:
        raise RuntimeError("CUDA 未暴露设备 UUID，不能安全映射 NVML 设备")
    if isinstance(value, bytes):
        if len(value) == 16 and not value.startswith((b"GPU-", b"MIG-")):
            value = uuid.UUID(bytes=value)
        else:
            value = value.decode("ascii")
    text = str(value)
    if not text.startswith(("GPU-", "MIG-")):
        try:
            # PyTorch 可能返回不带 NVML 前缀的 UUID 字符串或 UUID 对象。
            text = "GPU-" + str(uuid.UUID(text))
        except (ValueError, AttributeError) as exc:
            raise RuntimeError("CUDA 设备 UUID 格式无法识别，不能使用物理编号猜测") from exc
    return nvml.nvmlDeviceGetHandleByUUID(text.encode("ascii"))
