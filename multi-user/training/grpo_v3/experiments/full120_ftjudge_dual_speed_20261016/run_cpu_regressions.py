"""只运行本快照的轻量CPU回归，不加载模型、不调用GPU或提交作业。"""
from pathlib import Path
import os
import sys
import unittest


def main():
    here = Path(__file__).resolve().parent
    runtime = here / "runtime"
    package = next(p for p in here.parents if p.name == "multi-user")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(runtime), str(package)]

    # 保持归档源码原样；依赖沿用仓库，六用户实现取实际运行快照。
    import training
    import training.grpo_v3
    training.grpo_v3.__path__.insert(
        0, str(runtime / "multi-user/training/grpo_v3")
    )
    modules = [
        "test_answerability_batch", "test_decode_pipeline",
        "test_predictor_pipeline", "test_speed_sampling",
        "test_package_decode_import", "test_guard_deferred_rebuild",
        "test_recovery_prefix",
    ]
    suite = unittest.TestSuite(
        unittest.defaultTestLoader.loadTestsFromName(name) for name in modules
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
