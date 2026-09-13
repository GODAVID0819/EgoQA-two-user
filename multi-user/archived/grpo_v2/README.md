# GRPO v2 归档

本目录保存 judge reward v2 与 GRPO v2 LoRA smoke 相关代码。

状态：

- historical / not active
- 不被活跃 GRPOv3 引用
- `grpo_judge_reward/` 仍作为 v2 归档包保留
- `training/` 与 `hpc/` 保留历史复跑入口

归档运行语境：

```bash
set PYTHONPATH=%CD%\\archived\\grpo_v2;%CD%;%PYTHONPATH%
python -m pytest archived/grpo_v2/tests
```

Linux/Slurm 上对应语义是把 `${PROJECT_ROOT}/archived/grpo_v2` 放在 `PYTHONPATH` 最前面。
