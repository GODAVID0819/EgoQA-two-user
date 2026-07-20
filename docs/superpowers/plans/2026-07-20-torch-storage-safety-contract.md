# Torch Storage Safety Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent Torch jobs from writing caches, configuration, datasets, or temporary files under quota-constrained `/home/${USER}` and fail cheaply before model loading when storage isolation is invalid.

**Architecture:** A focused Python preflight validates a declared set of environment paths against one allowed scratch root, probes writability, and writes JSON evidence. The formality Smoke/Probe scripts define the complete scratch-first environment internally and invoke the preflight before reviewer or trainer startup; tests enforce ordering and variable coverage. Meta rules and the Runbook make the contract mandatory for future Torch documentation without making remote tests depend on Markdown.

**Tech Stack:** Python 3.11 standard library, `unittest`, Bash/Slurm, Markdown.

---

### Task 1: Storage preflight module

**Files:**
- Create: `training/torch_storage_preflight.py`
- Create: `tests/training/test_torch_storage_preflight.py`

- [ ] **Step 1: Write failing tests for safe and unsafe path sets**

Create tests that call:

```python
result = validate_storage_environment(
    allowed_root=root,
    environ={"HOME": str(root / "home"), "TMPDIR": str(root / "tmp")},
    required_variables=("HOME", "TMPDIR"),
)
```

Assert that safe paths pass, relative paths and paths outside `allowed_root` fail, a pre-existing sentinel file remains unchanged, and every result contains `path`, `within_allowed_root`, `writable`, and filesystem-space fields.

- [ ] **Step 2: Run the new test and verify RED**

Run:

```bash
python -m unittest tests.training.test_torch_storage_preflight -v
```

Expected: import failure because `training.torch_storage_preflight` does not exist.

- [ ] **Step 3: Implement the minimal preflight and CLI**

Implement:

```python
REQUIRED_STORAGE_VARIABLES = (
    "HOME", "XDG_CACHE_HOME", "HF_HOME", "HF_DATASETS_CACHE",
    "MODELSCOPE_CACHE", "TORCH_HOME", "TRITON_CACHE_DIR",
    "TORCHINDUCTOR_CACHE_DIR", "VLLM_CACHE_ROOT", "CUDA_CACHE_PATH",
    "FLASHINFER_WORKSPACE_BASE", "TMPDIR", "TMP", "TEMP",
)

def validate_storage_environment(
    *, allowed_root: Path, environ: Mapping[str, str],
    required_variables: Sequence[str] = REQUIRED_STORAGE_VARIABLES,
) -> dict[str, Any]:
    allowed = allowed_root.resolve()
    entries = []
    failed_checks = []
    for name in required_variables:
        raw = environ.get(name, "")
        path = Path(raw)
        absolute = path.is_absolute()
        resolved = path.resolve() if absolute else path
        within = absolute and resolved.is_relative_to(allowed)
        writable = False
        error = None
        if within:
            try:
                resolved.mkdir(parents=True, exist_ok=True)
                probe = resolved / f".egoqa-storage-probe-{uuid.uuid4().hex}"
                probe.write_bytes(b"ok")
                probe.unlink()
                writable = True
            except OSError as exc:
                error = f"{type(exc).__name__}: {exc}"
        if not absolute or not within or not writable:
            failed_checks.append(name)
        entries.append({"variable": name, "path": raw, "absolute": absolute,
                        "within_allowed_root": within, "writable": writable,
                        "error": error})
    return {"status": "passed" if not failed_checks else "failed",
            "allowed_root": str(allowed), "failed_checks": failed_checks,
            "paths": entries}
```

For each path: require absolute path, require `path.resolve().is_relative_to(allowed_root.resolve())`, create the directory, write and unlink one uniquely named small probe, and record `shutil.disk_usage(path)`. Return `status=failed` with `failed_checks` instead of deleting any existing data. The CLI accepts `--allowed-root` and `--output`, prints JSON, writes the same JSON, and exits 2 on failure.

- [ ] **Step 4: Run tests and verify GREEN**

Run the command from Step 2. Expected: all storage preflight tests pass.

### Task 2: Slurm scratch-first integration

**Files:**
- Modify: `tests/training/test_grpo_v3_formality_slurm.py`
- Modify: `hpc/grpo_v3_formality_smoke.sbatch`
- Modify: `hpc/grpo_v3_formality_probe.sbatch`

- [ ] **Step 1: Write failing Slurm contract tests**

For both scripts assert presence of the complete environment list from Task 1 plus:

```python
preflight = "training.torch_storage_preflight"
self.assertLess(text.index(preflight), text.index('"${VLLM}" serve'))
self.assertLess(text.index(preflight), text.index('"${SWIFT}" rlhf'))
self.assertIn('"${OUTPUT_DIR}/storage_preflight.json"', text)
self.assertIn("VLLM_NO_USAGE_STATS=1", text)
```

- [ ] **Step 2: Run the Slurm test and verify RED**

Run:

```bash
python -m unittest tests.training.test_grpo_v3_formality_slurm -v
```

Expected: failure for missing `HOME`, ModelScope/HF datasets/temp variables, and storage preflight invocation.

- [ ] **Step 3: Implement the complete environment in both scripts**

Derive all paths from:

```bash
JOB_SCRATCH_ROOT="${JOB_SCRATCH_ROOT:-/scratch/${USER}/formality_job_runtime}"
export HOME="${HOME_OVERRIDE:-${JOB_SCRATCH_ROOT}/home}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-${JOB_SCRATCH_ROOT}/modelscope}"
export VLLM_NO_USAGE_STATS=1
```

Set every required variable, create the directories, and before existing `grpo_v3_preflight` run:

```bash
"${PYTHON}" -m training.torch_storage_preflight \
  --allowed-root "${JOB_SCRATCH_ROOT}" \
  --output "${OUTPUT_DIR}/storage_preflight.json"
```

Do not rely on `sbatch --export` for safety defaults.

- [ ] **Step 4: Run Slurm and full formality tests**

Run:

```bash
python -m unittest tests.training.test_grpo_v3_formality_slurm -v
python -m unittest discover -s tests/training -p 'test_grpo_v3_formality_*.py' -v
```

Expected: all tests pass.

### Task 3: Artifact evidence contract

**Files:**
- Modify: `tests/training/test_grpo_v3_formality_artifacts.py`
- Modify: `training/grpo_v3_formality_artifacts.py`

- [ ] **Step 1: Write failing artifact tests**

Update the fixture to write:

```python
(output / "storage_preflight.json").write_text(
    json.dumps({"status": "passed", "allowed_root": "/scratch/test"}),
    encoding="utf-8",
)
```

Assert validation fails when the file is missing or has `status=failed`, and assert `run_manifest.json` embeds the storage preflight report.

- [ ] **Step 2: Run artifact tests and verify RED**

Run:

```bash
python -m unittest tests.training.test_grpo_v3_formality_artifacts -v
```

Expected: current validator passes without the report and manifest omits it.

- [ ] **Step 3: Require and summarize storage evidence**

Add:

```python
storage_preflight = _read_json(output_dir / "storage_preflight.json")
checks["storage_preflight_passed"] = storage_preflight.get("status") == "passed"
```

Embed the same object in `resolved_config` and `manifest` so successful and failed artifact summaries preserve the storage boundary.

- [ ] **Step 4: Run artifact and full formality tests**

Run artifact tests, then the full formality pattern. Expected: all pass.

### Task 4: Meta rules and human Runbook

**Files:**
- Create: `docs/AGENTS.md`
- Modify: `docs/TORCH_EXPERIMENT_META_RULES_CN.md`
- Modify: `docs/GRPO/v3/experiments/qa_formality_only_convergence_v1/TORCH_RUNBOOK_CN.md`

- [ ] **Step 1: Add the cross-experiment hard gate**

Add a new Meta section that states every Torch Runbook author must read the Meta file first, link it from new Runbooks, define all scratch-first variables inside `.sbatch`, run storage preflight before model loading, never auto-delete user data, and classify disk/quota errors as infrastructure failures.

Create `docs/AGENTS.md` with the repository-local authoring instruction:

```markdown
# 文档编写规则

- 创建或修改任何 Torch Runbook、Slurm 执行手册或远端实验文档前，必须先完整阅读 `TORCH_EXPERIMENT_META_RULES_CN.md`。
- Torch 手册必须遵守其中的 scratch-first 存储契约、前置检查、失败分层和证据口径。
- Markdown 操作手册默认只供本地人工阅读，不得让远端训练测试依赖其存在。
```

- [ ] **Step 2: Update the formality Runbook**

Add the Meta link near the top, add `training/torch_storage_preflight.py` and its test to the SFTP list, replace submission-shell cache fragments with the unified environment contract, add login-node preflight commands, require `storage_preflight.json` in Smoke/Probe acceptance, and document the ModelScope/vLLM home-write incident.

- [ ] **Step 3: Run documentation scans**

Run:

```powershell
rg -n "TORCH_EXPERIMENT_META_RULES_CN|storage_preflight|MODELSCOPE_CACHE|VLLM_NO_USAGE_STATS" docs/TORCH_EXPERIMENT_META_RULES_CN.md docs/GRPO/v3/experiments/qa_formality_only_convergence_v1/TORCH_RUNBOOK_CN.md
rg -n --glob '*.md' '\bexit\s+[0-9]+' docs
```

Expected: required contract terms appear and interactive Markdown contains no numeric `exit`.

### Task 5: Final verification and delivery

**Files:**
- Verify all files from Tasks 1-4.

- [ ] **Step 1: Run complete local verification**

```bash
python -m unittest tests.training.test_torch_storage_preflight -v
python -m unittest discover -s tests/training -p 'test_grpo_v3_formality_*.py' -v
python -m compileall training tests/training
git diff --check
```

Expected: tests pass, compilation succeeds, and diff check is clean.

- [ ] **Step 2: Inspect scope and prepare the remote handoff**

List only changed files, distinguish unrelated dirty-tree changes, and provide exact SFTP commands for the new preflight module/test plus updated Smoke/Probe/formality files. State explicitly that Markdown remains local and that Torch must still run `bash -n`, storage preflight, and a fresh 1-step Smoke before claiming the remote issue resolved.
