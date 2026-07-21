# Torch 执行手册

执行前必须完整阅读并遵守 [Torch 实验 Meta 规则](../../../../TORCH_EXPERIMENT_META_RULES_CN.md)。本手册只供人工操作；远端脚本和测试不得依赖 Markdown。

## 1. 上传与登录节点审计

在本地仓库根目录上传本分支源码（排除 `.git`、输出和缓存）：

```powershell
sftp <torch-host>
put -r training /scratch/<user>/projects/EgoQA-two-user/
put -r hpc /scratch/<user>/projects/EgoQA-two-user/
```

远端分别审计训练与 scorer 环境：

```bash
cd /scratch/$USER/projects/EgoQA-two-user
/scratch/$USER/envs/egoqa-ms-swift-v4.2.2-vllm024/bin/python -m pip check
/scratch/$USER/envs/egoqa-answer-scorer/bin/python -m pip check
/scratch/$USER/envs/egoqa-ms-swift-v4.2.2-vllm024/bin/python -c 'import torch,transformers,peft,swift; print(torch.__version__,transformers.__version__)'
/scratch/$USER/envs/egoqa-answer-scorer/bin/python -c 'import torch,transformers; print(torch.__version__,transformers.__version__)'
command -v gcc; command -v g++; command -v ninja
bash -n hpc/grpo_v3_answer_margin_{scorer_probe,calibration,smoke1,smoke5,probe40,fixed_eval}.sbatch
```

确认模型、Gate 0 数据、`gate2_result.json`、`run_manifest.json` 和 `checkpoint-1` 均在 scratch；不得用 `latest` 指针替代 manifest 与哈希清单的最终验收。

## 2. 严格依次提交

先创建 Slurm 日志目录：

```bash
mkdir -p logs outputs/grpo_v3
```

每次只提交一个 Gate，并等待其 JSON 为 `passed` 后继续：

```bash
sbatch hpc/grpo_v3_answer_margin_scorer_probe.sbatch
sbatch hpc/grpo_v3_answer_margin_calibration.sbatch
sbatch hpc/grpo_v3_answer_margin_smoke1.sbatch
sbatch hpc/grpo_v3_answer_margin_smoke5.sbatch
sbatch hpc/grpo_v3_answer_margin_probe40.sbatch
sbatch hpc/grpo_v3_answer_margin_fixed_eval.sbatch
```

监控与调度核对：

```bash
squeue -u "$USER"
sacct -j <jobid> --format=JobID,State,ExitCode,Elapsed,AllocTRES%80
scontrol show job -dd <jobid>
```

Gate 文件分别检查：`scorer_probe_result.json`、`calibration_result.json`、`answer_margin_smoke1_result.json`、`answer_margin_smoke5_result.json`、`answer_margin_probe40_result.json`、`fixed_eval_summary.json`。每个作业还必须有 `storage_preflight.json`；训练 Gate 必须有 reward trace、独立环境审计、父 checkpoint 哈希清单、adapter/processor 与 reload 证据。

若 scorer、CUDA、缓存、JIT 或 GPU 可见性失败，这是基础设施失败，修复后回到触发失败的最小 Gate。若 1-step 未过，禁止提交 5-step；5-step 未过，禁止提交 40-step。`not_converged` 是有效研究结果且 fixed-eval 作业退出码为 0；`invalid` 必须非零退出。

## 3. 下载验收证据

```powershell
sftp <torch-host>
get -r /scratch/<user>/projects/EgoQA-two-user/outputs/grpo_v3/answer_margin_scorer_probe_<jobid> ./torch_results/
get -r /scratch/<user>/projects/EgoQA-two-user/outputs/grpo_v3/answer_margin_calibration_<jobid> ./torch_results/
get -r /scratch/<user>/projects/EgoQA-two-user/outputs/grpo_v3/answer_margin_smoke1_<jobid> ./torch_results/
get -r /scratch/<user>/projects/EgoQA-two-user/outputs/grpo_v3/answer_margin_smoke5_<jobid> ./torch_results/
get -r /scratch/<user>/projects/EgoQA-two-user/outputs/grpo_v3/answer_margin_probe40_<jobid> ./torch_results/
get -r /scratch/<user>/projects/EgoQA-two-user/outputs/grpo_v3/answer_margin_fixed_eval_<jobid> ./torch_results/
```

下载后按调度、基础设施、reward 语义、训练四层分别汇报，不得把本地通过、作业完成或 `not_converged` 写成已收敛。
