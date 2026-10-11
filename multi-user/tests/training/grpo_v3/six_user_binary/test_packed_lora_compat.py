"""用实际安装的vLLM方法和CPU张量复现部分融合LoRA加载错误。"""
import ast
import __future__
import os
from pathlib import Path
import sysconfig
from types import SimpleNamespace
import unittest


def installed_expand():
    path = Path(sysconfig.get_paths()['purelib']) / 'vllm/lora/layers/column_parallel_linear.py'
    if not path.is_file():
        raise unittest.SkipTest('需要当前Python环境实际安装的vLLM源码')
    tree = ast.parse(path.read_text())
    methods = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'expand_packed_lora']
    if len(methods) != 1:
        raise RuntimeError('实际vLLM方法不唯一')
    namespace = {}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), 'exec',
                 flags=__future__.annotations.compiler_flag), namespace)
    original = namespace['expand_packed_lora']
    # 默认验证修复；显式设0时保留原方法以复现历史错误。
    if os.environ.get('EGOQA_TEST_PACKED_COMPAT', '1') == '1':
        from training.grpo_v3.six_user_binary.packed_lora_compat import wrap_expand
        return wrap_expand(original)
    return original


class PackedLoRACompatibilityTests(unittest.TestCase):
    def setUp(self):
        try:
            import torch
        except ModuleNotFoundError as exc:
            if exc.name != 'torch':
                raise
            self.skipTest('需要当前Python环境安装PyTorch才能运行数值兼容检查')
        self.torch = torch
        self.expand = installed_expand()
        self.layer = SimpleNamespace(n_slices=4, output_sizes=[2, 2, 6, 6])
        self.a = torch.arange(12, dtype=torch.float32).reshape(2, 6)
        self.b = torch.arange(20, dtype=torch.float32).reshape(10, 2)

    def test_partial_qkv_preserves_rows_and_leaves_z_unadapted(self):
        a, b = self.expand(self.layer, [self.a, None], [self.b, None])
        self.assertEqual(len(a), 4)
        self.assertIsNone(a[3]); self.assertIsNone(b[3])
        self.assertTrue(self.torch.equal(self.torch.cat(b[:3]), self.b))
        x = self.torch.arange(6, dtype=self.a.dtype)
        actual = self.torch.cat([bi @ ai @ x for ai, bi in zip(a[:3], b[:3])])
        self.assertTrue(self.torch.equal(actual, self.b @ self.a @ x))

    def test_complete_group_preserves_both_adapters(self):
        z_a = self.a + 2; z_b = self.torch.ones(6, 2)
        a, b = self.expand(self.layer, [self.a, z_a], [self.b, z_b])
        self.assertTrue(self.torch.equal(self.torch.cat(b[:3]), self.b))
        self.assertIs(a[3], z_a)
        self.assertTrue(self.torch.equal(b[3], z_b))

    def test_invalid_qkv_row_count_still_fails(self):
        with self.assertRaises(ValueError):
            self.expand(self.layer, [self.a, None], [self.torch.ones(11, 2), None])

    def test_unrelated_fused_group_keeps_original_result(self):
        layer = SimpleNamespace(n_slices=2, output_sizes=[4, 6])
        a, b = self.expand(layer, [self.a], [self.b])
        self.assertEqual(len(b), 2)
        self.assertTrue(self.torch.equal(self.torch.cat(b), self.b))


if __name__ == '__main__': unittest.main()
