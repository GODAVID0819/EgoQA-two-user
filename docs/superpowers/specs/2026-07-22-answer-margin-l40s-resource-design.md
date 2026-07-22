# Answer-Margin L40S 资源设计

## 目标

Combined-Video Answer-Margin Convergence Sweep v1 默认使用兼顾排队、运行速度和 OOM 重提成本的 L40S 48GB，不再为 24GB GPU 建立正式资源路径。

## 默认资源

- `scorer_probe`：1 张 L40S 48GB。
- `calibration`、`smoke1`、`smoke5`、`probe40`、`fixed_eval`：2 张 L40S 48GB；GPU0 运行 policy，GPU1 运行冻结 scorer。
- 所有作业在模型加载前保存实际 GPU 名称、显存、驱动和 CUDA 版本。
- L40S 比 H100 慢，适当增加 walltime，但不修改任何研究参数。

## 升级边界

只有 CUDA OOM、L40S BF16/runtime 不兼容，或同一硬件故障可复现时，才把同一 Gate 单变量升级到 A100 80GB 或 H100 80GB。依赖、视频、数据、路径和 scorer 语义错误不得靠升级 GPU 掩盖。升级时保持模型、batch、`num_generations`、视频输入、像素、dtype、temperature、步数和 reward 完全不变。

## 验证

- 静态测试检查六个脚本的 L40S 数量、GPU 证据位置和双 GPU 隔离。
- 远端运行 `bash -n`，并用 `scontrol` 核对实际 GRES。
- OOM 后保留原 JobID、stdout、stderr、`gpu_environment.csv`，再提交新的同 Gate 作业。
