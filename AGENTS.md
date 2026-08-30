# EgoQA-two-user Agent 规则

## 通用要求

- 默认使用中文回答、写文档和汇报；公式可使用 LaTeX。
- 默认采用轻量、敏捷的开发方式。写计划或新增测试前，先判断它是否会降低真实风险、锁定训练语义或防止已知回归；不为流程完整而创建仪式性的计划书、测试或重复文档。
- 小范围、低风险、可直接验证的修改可以边实现边验证；涉及 reward 公式、judge schema、数据划分、训练入口和远程 Gate 的行为变化，保留最小且有针对性的回归测试。
- 一份仍然准确的设计或运行文档优先原地更新，不为同一结论创建多个近似版本。
- 本项目的真实 Git 根目录是本文件所在的 `EgoQA-two-user`，不要把外层 `multiuser` 当作项目仓库。
- 修改前先检查真实分支、worktree、提交关系和 dirty state；禁止用 `git reset --hard`、强制覆盖或清理命令处理研究工作区。
- 保留本地未跟踪、忽略的研究文件和大型输出。源码、测试、Slurm 脚本和运行约束进入 Git；大型实验产物通过 `rsync`、归档或选择性下载管理。

## GRPO 与 Torch 硬约束

- 进行任何 GRPO、Slurm、Torch、reviewer、scorer 或远程同步任务前，必须完整阅读：
  - `training/grpo_v3/REMOTE_EXECUTION_GUARDRAILS_CN.md`
  - 与当前实验对应的 `training/grpo_v3/experiments/<experiment>/`、`tests/` 和 `hpc/` 文件。
- 本地测试通过、远端作业完成、代理奖励收敛、固定端点评估通过、真实 QA 质量改善是五种不同证据，不得互相替代。
- 按 Gate 顺序执行并停在第一个失败 Gate。未经新证据和明确决定，不要同时修改模型、数据、reward、学习率、温度和训练步数。
- 每个新增或修改的 `.sbatch` 必须：
  - 在模型加载前运行 `training.torch_storage_preflight`；
  - 将全部缓存和临时目录固定到 job-specific scratch；
  - 为原生视频任务同时设置 FFmpeg runtime 的 `PATH` 与 `LD_LIBRARY_PATH`；
  - 显式固定并预检任务实际使用的视频后端；视频像素上、下界必须成对显式传入，不依赖第三方包隐藏默认值；不得把未被应用选择的 decoder 设为硬依赖；
  - 从真实 `${SLURM_JOB_ID}` 派生产物路径，不用 `latest_*` 作为结论依据。
- reviewer HTTP 200 只证明服务可访问，不证明 schema、奖励语义或训练正确。至少检查真实调用形状、reward trace、梯度/参数变化、checkpoint 和 adapter reload。
- 汇报实验时必须明确：
  - 能证明什么；
  - 不能证明什么；
  - 当前失败属于调度、基础设施、reward 语义、训练还是研究 Gate；
  - 下一步需要越过哪条最小边界。

## 当前 cross-view GRPO 边界

- 当前主线是 `qa_cross_view_relation`。其 text-only judge 负责格式、自然度、内部一致性、非浅层关系和文本关系强度；除非实验版本明确改变，不负责视频 groundedness 或真实 answerability。
- `deterministic.py` 是候选资格检查/路由，不是奖励公式中的独立加权项。
- 任何阻断性内部一致性错误、问题—答案类型错位、语义重复选项或浅层活动问题，都必须通过可审计规则限制最终奖励，不能仅依赖 judge 的总体印象。
- 训练数据必须报告独立 `evidence_id` 数量、question type 分布和 held-out 划分。单一视频对上的 reward 上升不能称为跨 clip 泛化。
- 失败实验应保留在明确的 archived 路径；活跃实验不得反向依赖 archived 实现。
## Torch 视频结果下载规则

- 从 Torch 下载视频用于人工审核时，只下载已生成并拼接完成的 `stitched` 目录及其中的成片；不要下载原始分段、缓存、临时文件或整个候选媒体树。
- 本地视频目标目录应保持类似 `review_artifacts\\six_user_qa_3min_resubmissions_20260827\\media_DAY6_20060000\\stitched` 的结构，便于人工审核工具直接打开。
- Markdown 审核文档与视频成片分开管理；若用户只要求 Markdown，则不下载 JSON 或视频。
