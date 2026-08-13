# Torch 实验总规则

本文件是本仓库所有 Torch、Slurm 和远程实验 Runbook 的上位规则。具体实验手册可以补充更严格的 Gate，但不得削弱本文件的同步、安全和证据边界。

## 1. 登录 shell 必须可继续使用

- 可复制到 SSH 登录节点的命令块不得主动结束当前 shell。
- 失败时打印 `STOP` 或 `MISSING`，设置就绪变量为 0，并跳过依赖步骤。
- 不把 Slurm `COMPLETED 0:0` 等同于训练、Gate、产物或科研结论成功；每层证据必须分别验收。
- JobID 必须由 `sbatch --parsable` 自动捕获并写入时间戳 manifest，不要求用户记忆或长期记录。

## 2. 本地开发仓库与远程实验 worktree 使用不同清洁度合同

Windows 本地开发仓库在提交或推送前可以使用完整 `git status` 检查，包括未跟踪源码和文档。

远程实验 worktree 会合法地产生数据、日志和模型产物。Git 快进同步只应阻止两类状态：

1. 已跟踪文件存在未暂存修改；
2. 暂存区存在未提交修改。

统一同步 Gate：

```bash
if git -C "${PROJECT_ROOT}" diff --quiet \
  && git -C "${PROJECT_ROOT}" diff --cached --quiet; then
  git -C "${PROJECT_ROOT}" merge --ff-only "origin/${BRANCH}"
else
  echo "STOP: remote worktree has tracked or staged changes"
  git -C "${PROJECT_ROOT}" status --short --untracked-files=no
  READY=0
fi
```

不得用“工作树必须完全没有未跟踪文件”作为远程快进条件，也不得通过全局隐藏所有未跟踪文件来绕过检查。未跟踪文件若会覆盖远端新增的同名路径，`merge --ff-only` 仍会拒绝覆盖。

## 3. 运行时目录使用 repository-local exclude

创建或首次初始化 Torch worktree 后，将明确的运行时目录写入该仓库的 Git exclude。`info/exclude` 位于 `$GIT_COMMON_DIR`，因此同一仓库的 linked worktrees 共享这些规则；它不进入提交、不影响用户的全局 Git 配置，也不删除目录内容：

```bash
LOCAL_EXCLUDE=$(git -C "${PROJECT_ROOT}" rev-parse --git-path info/exclude)
mkdir -p "$(dirname "${LOCAL_EXCLUDE}")"

for PATTERN in \
  "/data_RLHF/" \
  "/outputs/" \
  "/logs/"; do
  if grep -Fqx "${PATTERN}" "${LOCAL_EXCLUDE}" 2>/dev/null; then
    echo "ALREADY_EXCLUDED: ${PATTERN}"
  else
    printf '%s\n' "${PATTERN}" >> "${LOCAL_EXCLUDE}"
    echo "ADDED_LOCAL_EXCLUDE: ${PATTERN}"
  fi
done
```

只有整个仓库都明确约定为运行时产物、且不应进入 Git 的目录才能加入这里。新的源码、测试、配置、Runbook 和 manifest 模板不能因为“方便”而被排除。不要仅为这些目录启用 `extensions.worktreeConfig`；真正阻止实验数据卡住同步的是上一节的 tracked/staged Gate。

## 4. 同步后的固定验收

每次远程同步至少检查：

```bash
LOCAL_HEAD=$(git -C "${PROJECT_ROOT}" rev-parse HEAD)
REMOTE_HEAD=$(git -C "${PROJECT_ROOT}" rev-parse "origin/${BRANCH}")

if [ "${LOCAL_HEAD}" = "${REMOTE_HEAD}" ]; then
  echo "GIT_SYNC_PASSED branch=${BRANCH} head=${LOCAL_HEAD}"
else
  echo "STOP: worktree head does not match origin branch"
fi
```

提交 Slurm 任务前还必须运行目标脚本的 `bash -n`、验证输入文件非空、确认模型和环境路径，并把 resolved command、环境、JobID 和唯一输出目录持久化。

## 5. 证据与汇报边界

每次汇报必须区分：

- 本地静态检查；
- Torch 远程语法与路径预检；
- Slurm 调度和进程退出状态；
- checkpoint、manifest、指标和 Gate 产物完整性；
- 科研问题是否得到支持。

任何上游失败都不得手工强制启动依赖其完整 checkpoint 的下游任务。诊断包只收集日志和小型结构化证据，不默认打包模型、视频、缓存或全部输出目录。
