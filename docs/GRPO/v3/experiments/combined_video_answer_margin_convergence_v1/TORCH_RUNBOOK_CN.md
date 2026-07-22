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

若为了先跑通链路而让 policy 与 scorer 共用已验证环境，必须显式传入相同路径，并在结果中记录 `dependency_environment_mode=shared`；这不改变进程、模型冻结和 GPU 隔离，但不能声称依赖环境物理独立：

```bash
export TRAIN_ENV=/scratch/$USER/envs/egoqa-ms-swift-v4.2.2-vllm024
export SCORER_ENV="$TRAIN_ENV"
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

## 3. 每个作业结束后的日志、错误与产物验证

### 3.1 不要在失败作业中依赖 `latest_*` 指针

`latest_answer_margin_*_output.txt` 只会在对应 Gate 成功走到脚本末尾后写入。作业失败时该文件可能不存在，或仍指向更早的成功作业。失败诊断必须用 Slurm job ID 推导本次输出目录。

先设置本次 Gate 和 job ID。`GATE` 只能取 `scorer_probe`、`calibration`、`smoke1`、`smoke5`、`probe40`、`fixed_eval`：

```bash
cd /scratch/$USER/projects/EgoQA-two-user
JOB_ID=<本次jobid>
GATE=<本次gate>
GRPO_V3_ROOT="$PWD/outputs/grpo_v3"

case "$GATE" in
  scorer_probe)
    OUTPUT_DIR="$GRPO_V3_ROOT/answer_margin_scorer_probe_${JOB_ID}"
    STDOUT="logs/grpo-v3-answer-margin-scorer-${JOB_ID}.out"
    STDERR="logs/grpo-v3-answer-margin-scorer-${JOB_ID}.err"
    RESULT_FILE="scorer_probe_result.json"
    ;;
  calibration)
    OUTPUT_DIR="$GRPO_V3_ROOT/answer_margin_calibration_${JOB_ID}"
    STDOUT="logs/grpo-v3-answer-margin-calibration-${JOB_ID}.out"
    STDERR="logs/grpo-v3-answer-margin-calibration-${JOB_ID}.err"
    RESULT_FILE="calibration_result.json"
    ;;
  smoke1)
    OUTPUT_DIR="$GRPO_V3_ROOT/answer_margin_smoke1_${JOB_ID}"
    STDOUT="logs/grpo-v3-answer-margin-smoke1-${JOB_ID}.out"
    STDERR="logs/grpo-v3-answer-margin-smoke1-${JOB_ID}.err"
    RESULT_FILE="answer_margin_smoke1_result.json"
    ;;
  smoke5)
    OUTPUT_DIR="$GRPO_V3_ROOT/answer_margin_smoke5_${JOB_ID}"
    STDOUT="logs/grpo-v3-answer-margin-smoke5-${JOB_ID}.out"
    STDERR="logs/grpo-v3-answer-margin-smoke5-${JOB_ID}.err"
    RESULT_FILE="answer_margin_smoke5_result.json"
    ;;
  probe40)
    OUTPUT_DIR="$GRPO_V3_ROOT/answer_margin_probe40_${JOB_ID}"
    STDOUT="logs/grpo-v3-answer-margin-probe40-${JOB_ID}.out"
    STDERR="logs/grpo-v3-answer-margin-probe40-${JOB_ID}.err"
    RESULT_FILE="answer_margin_probe40_result.json"
    ;;
  fixed_eval)
    OUTPUT_DIR="$GRPO_V3_ROOT/answer_margin_fixed_eval_${JOB_ID}"
    STDOUT="logs/grpo-v3-answer-margin-fixed-eval-${JOB_ID}.out"
    STDERR="logs/grpo-v3-answer-margin-fixed-eval-${JOB_ID}.err"
    RESULT_FILE="fixed_eval_summary.json"
    ;;
  *) echo "未知 GATE=$GATE" >&2; exit 2 ;;
esac

printf 'OUTPUT_DIR=%s\nSTDOUT=%s\nSTDERR=%s\nRESULT=%s\n' \
  "$OUTPUT_DIR" "$STDOUT" "$STDERR" "$OUTPUT_DIR/$RESULT_FILE"
```

### 3.2 调度层检查

作业运行中：

```bash
squeue -j "$JOB_ID" -o '%.18i %.12P %.30j %.2t %.10M %.10l %.6D %R'
scontrol show job -dd "$JOB_ID"
tail -n 100 -F "$STDOUT" "$STDERR"
```

作业结束后停止 `tail -F`，再执行：

```bash
sacct -j "$JOB_ID" \
  --format=JobID,JobName%30,State,ExitCode,Elapsed,Start,End,NodeList,AllocTRES%80
```

只有顶层作业和 batch step 均为 `COMPLETED`、`ExitCode=0:0`，才说明调度与 shell 层完成；这仍不等于研究门槛通过。

### 3.3 stdout、stderr 和内部服务日志

无论成功或失败都执行：

```bash
echo '===== stdout tail ====='
tail -n 200 "$STDOUT" 2>&1 || true

echo '===== stderr tail ====='
tail -n 200 "$STDERR" 2>&1 || true

echo '===== output inventory ====='
find "$OUTPUT_DIR" -maxdepth 2 -type f -printf '%P %s bytes\n' 2>&1 | sort || true

echo '===== scorer service tail ====='
tail -n 300 "$OUTPUT_DIR/scorer_service.log" 2>&1 || true
```

快速定位常见根因：

```bash
grep -RniE \
  'Traceback|ERROR|Error|Exception|CUDA out of memory|OutOfMemory|No such file|HTTP 500|unhealthy|NaN|Inf' \
  "$STDOUT" "$STDERR" "$OUTPUT_DIR" 2>/dev/null | tail -n 300 || true
```

scorer 的 HTTP 500 只是客户端看到的外层错误；真正的 processor、视频解码、CUDA 或模型 forward traceback 在 `scorer_service.log`，必须优先读取该文件。

### 3.4 基础设施证据检查

```bash
for file in \
  storage_preflight.json \
  policy_environment.txt \
  scorer_environment.txt \
  scorer_health.json; do
  if [[ -f "$OUTPUT_DIR/$file" ]]; then
    echo "===== $file ====="
    if [[ "$file" == *.json ]]; then
      python -m json.tool "$OUTPUT_DIR/$file" || cat "$OUTPUT_DIR/$file"
    else
      head -n 30 "$OUTPUT_DIR/$file"
    fi
  else
    echo "MISSING $OUTPUT_DIR/$file"
  fi
done
```

训练 Gate 还要检查：

```bash
for file in \
  reward_trace.jsonl \
  checkpoint_inventory.json \
  adapter_reload.json; do
  [[ -s "$OUTPUT_DIR/$file" ]] \
    && echo "PRESENT $file $(wc -l < "$OUTPUT_DIR/$file") lines" \
    || echo "MISSING_OR_EMPTY $file"
done

find "$OUTPUT_DIR" -name trainer_state.json -o -name adapter_config.json \
  -o -name adapter_model.safetensors -o -name processor_config.json
```

### 3.5 Gate 结果 JSON 验证

结果文件存在时执行：

```bash
if [[ -s "$OUTPUT_DIR/$RESULT_FILE" ]]; then
  python -m json.tool "$OUTPUT_DIR/$RESULT_FILE"
  python - "$OUTPUT_DIR/$RESULT_FILE" <<'PY'
import json, math, sys
from pathlib import Path

path = Path(sys.argv[1])
value = json.loads(path.read_text(encoding="utf-8"))
status = value.get("status")
run_status = value.get("run_status")
conclusion = value.get("experiment_conclusion")
print("result_file =", path)
print("status =", status)
print("run_status =", run_status)
print("experiment_conclusion =", conclusion)

def walk(item, where="root"):
    if isinstance(item, float) and not math.isfinite(item):
        raise SystemExit(f"non-finite value at {where}: {item}")
    if isinstance(item, dict):
        for key, child in item.items():
            walk(child, f"{where}.{key}")
    elif isinstance(item, list):
        for index, child in enumerate(item):
            walk(child, f"{where}[{index}]")

walk(value)
print("finite_json_check = passed")
PY
else
  echo "RESULT_MISSING: $OUTPUT_DIR/$RESULT_FILE"
fi
```

各 Gate 的继续条件：

| Gate | 结果文件 | 允许继续的条件 |
|---|---|---|
| scorer probe | `scorer_probe_result.json` | `status=passed`；两次请求一致；零可训练参数；泄露扫描通过 |
| calibration | `calibration_result.json` | `status=passed`；32/32 有限；零 mask；研究阈值通过 |
| smoke1 | `answer_margin_smoke1_result.json` | `run_status=passed` 且 1-step 完整 |
| smoke5 | `answer_margin_smoke5_result.json` | `run_status=passed` 且 5-step 完整 |
| probe40 | `answer_margin_probe40_result.json` | `run_status=passed` 且 40-step/160 traces 完整；不等于已收敛 |
| fixed eval | `fixed_eval_summary.json` | `experiment_conclusion=passed` 才证明 v1 收敛；`not_converged` 是有效负结果；`invalid` 无研究结论 |

任何结果文件缺失、空文件、非有限值、`masked>0`、adapter/reload 缺失或基础设施状态失败，都禁止提交下一 Gate。

### 3.6 本次 scorer probe 失败的正确定位示例

若 job ID 为 `14497742`，不要读取不存在的成功指针，直接执行：

```bash
JOB_ID=14497742
OUTPUT_DIR="/scratch/$USER/projects/EgoQA-two-user/outputs/grpo_v3/answer_margin_scorer_probe_${JOB_ID}"

find "$OUTPUT_DIR" -maxdepth 1 -type f -printf '%f %s bytes\n' | sort
tail -n 300 "$OUTPUT_DIR/scorer_service.log"
grep -nE 'Traceback|ERROR|Error|Exception|CUDA|video|processor' \
  "$OUTPUT_DIR/scorer_service.log" | tail -n 200 || true
```

Gate 文件分别检查：`scorer_probe_result.json`、`calibration_result.json`、`answer_margin_smoke1_result.json`、`answer_margin_smoke5_result.json`、`answer_margin_probe40_result.json`、`fixed_eval_summary.json`。每个作业还必须有 `storage_preflight.json`；训练 Gate 必须有 reward trace、环境审计、父 checkpoint 哈希清单、adapter/processor 与 reload 证据。

若 scorer、CUDA、缓存、JIT 或 GPU 可见性失败，这是基础设施失败，修复后回到触发失败的最小 Gate。若 1-step 未过，禁止提交 5-step；5-step 未过，禁止提交 40-step。`not_converged` 是有效研究结果且 fixed-eval 作业退出码为 0；`invalid` 必须非零退出。

## 4. 下载验收证据

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
