"""CUDA 逻辑编号不可直接当作 NVML 物理编号。"""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
import uuid


class GPUIdentityTests(unittest.TestCase):
    def test_torch_uuid_without_nvml_prefix_is_normalized(self):
        path = Path(__file__).resolve().parents[5] / "hpc/shared/cuda_device_identity.py"
        spec = importlib.util.spec_from_file_location("keeper_device_identity", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        value = uuid.UUID("00112233-4455-6677-8899-aabbccddeeff")
        calls = []
        nvml = SimpleNamespace(nvmlDeviceGetHandleByUUID=lambda item: calls.append(item) or "correct_gpu")
        for variant in (str(value), value, value.bytes):
            cuda = SimpleNamespace(get_device_properties=lambda i, v=variant: SimpleNamespace(uuid=v))
            self.assertEqual(module.handle_for_cuda_device(0, cuda, nvml), "correct_gpu")
        self.assertEqual(calls, [b"GPU-00112233-4455-6677-8899-aabbccddeeff"] * 3)

    def test_uses_visible_device_uuid_instead_of_physical_index(self):
        path = Path(__file__).resolve().parents[5] / "hpc/shared/cuda_device_identity.py"
        self.assertTrue(path.is_file(), "缺少 GPU 身份映射辅助函数")
        spec = importlib.util.spec_from_file_location("keeper_device_identity", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        calls = []
        cuda = SimpleNamespace(get_device_properties=lambda i: SimpleNamespace(uuid=f"GPU-physical-{i+2}"))
        nvml = SimpleNamespace(nvmlDeviceGetHandleByUUID=lambda value: calls.append(value) or "handle")
        self.assertEqual(module.handle_for_cuda_device(0, cuda, nvml), "handle")
        self.assertEqual(calls, [b"GPU-physical-2"])
        cuda = SimpleNamespace(get_device_properties=lambda i: SimpleNamespace())
        with self.assertRaises(RuntimeError):
            module.handle_for_cuda_device(0, cuda, nvml)


if __name__ == "__main__":
    unittest.main()
