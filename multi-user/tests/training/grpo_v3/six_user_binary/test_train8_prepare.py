"""不同packet标识也不能掩盖训练与验证媒体的时间重叠。"""
import importlib.util
from pathlib import Path
import unittest


class Train8PreparationTests(unittest.TestCase):
    def test_actual_ranges_reject_overlap_with_independent_validation(self):
        path=Path(__file__).resolve().parents[4]/'hpc/grpo_v3/six_user_binary/prepare_train8.py'
        self.assertTrue(path.is_file(),'需要实际媒体准备脚本')
        spec=importlib.util.spec_from_file_location('train8_prepare',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        val=[{'day':'DAY4','time_token':'21400000','duration_seconds':600}]
        train=[{'day':'DAY2','time_token':'16000000','duration_seconds':600}]
        module.verify_disjoint_ranges(train,val)
        with self.assertRaises(ValueError):
            module.verify_disjoint_ranges([{'day':'DAY4','time_token':'21350000','duration_seconds':600}],val)
        with self.assertRaises(ValueError):
            module.verify_disjoint_ranges(train+[{'day':'DAY2','time_token':'16050000','duration_seconds':600}],val)


if __name__=='__main__':unittest.main()
