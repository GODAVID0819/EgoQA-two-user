# 120窗、r64、两学习率各300步：实验脚本快照

本目录归档2026-10-13至14日实际使用的实验辅助脚本、公开源索引与条件划分。原本地来源是 `.codex_runtime/grpo_pair300_20261013/`，原训练私有根目录是 `/scratch/xl6775/projects/EgoQA-six-user-baseline-grpo-20260929/variants/full120-r64-lrpair-20261013`。

这是一份有明确环境依赖的实验快照。上传代码不代表GPU检查、300步训练或终点评价已经完成。它也不是下载仓库后即可在任意机器运行的一键训练包。

## 实验设置

| 项目 | 本轮设置 |
|---|---|
| 任务 | 六视角、指定speaker的跨视角选择题QA生成；固定基座Judge评分 |
| 数据 | 120个同步十分钟窗口，每个输入仍含六视角 |
| 划分 | 每窗5训练/1固定speaker验证；600训练、120验证；六人各100/20 |
| 重叠处理 | 同日完整替换，按新增配对重叠秒数、移动距离、起点顺序选择；重叠连通组共用留出speaker |
| 每次更新 | 4个不同窗口，各一个speaker条件；每条件G4，共16候选 |
| 预算 | 每组300个优化器更新、1200条件访问、4800训练候选；600条件各访问2次 |
| LoRA | r64、alpha128、dropout0.05；64语言层的原q_proj/v_proj/in_proj_qkv；冻结视觉和aligner |
| 学习率 | 1e-5与5e-6；完整300步余弦退火，warmup10，最低为峰值10% |
| GRPO | 条件内组归一化、KL0.04、epsilon0.2、num_iterations1；微批1、梯度累积16 |
| 奖励 | $R=0.2F+0.4E+0.4A(1-S)$；形式、证据、六视角可答性、speaker-only可答性 |
| 媒体 | 0.5fps，speaker完整300帧，provider沿用原剪枝；每图max_pixels24576，上下文65536 |
| 生成 | temperature0.85、top_p0.95、top_k40、max_completion_length2048 |
| 随机种子 | seed/data_seed/validation_seed均42 |
| 验证与保存 | 第300更新完整120条件G4；固定基座/adapter配对验证；每2更新完整保存，保留3个 |
| 资源 | 正式每组1H200、16CPU、500G、168h；Policy/Judge同GPU交替运行 |

本轮模型目录名为 `Qwen3.8-27B`，原模型配置记录属于Qwen3.5架构；目录名不构成另一个官方模型版本的证据。原环境记录为ms-swift4.2.2、vLLM0.24.0、TRL0.29.1。

固定Judge奖励是代理指标。这项划分衡量同窗口的留出speaker条件；六视角视频内容仍可能在训练输入出现，不能据此声称未见视频泛化或人工正确率改善。

## 文件与用途

| 文件 | 用途 |
|---|---|
| `submit_pair.py` | 原初始提交器：生成两学习率配置及CPU→一次GPU检查→两组正式任务的Slurm依赖链 |
| `prepare_full120.py` | 公开源复用/下载、六视角拼接、CPU抽帧与CLIP、真实600/120训练数据构建 |
| `repair_sources.py` | 源视觉时长审计、最小新增重叠完整窗替换、连通组留出重分配及旧审计复用 |
| `pair_data.py` | 600/120划分校验和四不同窗口的确定性采样顺序 |
| `media_quality.py` | 真实视频流时长检查，防止音轨时长掩盖缺失视频 |
| `validate_full120.py` | 实际JSONL条件、六视角、speaker、四窗口顺序核对 |
| `audit_runtime.py` | 真实TRL重复采样、CLI参数、输入长度、r64模块覆盖的CPU审计 |
| `install_pair_runtime.py` | 本轮私有副本的三处源码调整：关闭再次打乱、vLLM rank上限、600/120验证路由 |
| `source_manifest.json` | **实际替换后的120窗**、公开URL、分段顺序与官方大小 |
| `condition_definition.json` | **实际固定划分**；train部分已按四窗口更新顺序排列 |
| `conditions/` | 可读600训练、120验证及300更新采样CSV |
| `official_files_sizes.json` | 源替换使用的官方公开视频文件索引，包含同日候选范围 |
| `window_replacements_report.json` | 当前作业结构化报告派生的13项替换、连通组及留出分配；不是远端原件下载 |
| `prerequisites/` | 此前同视频speaker留出扩展的原安装脚本、辅助模块及测试 |
| `history/` | 替换前输入与原恢复提交器，用于追溯失败/恢复来源 |

没有包含视频、帧、模型权重、checkpoint、认证日志、设备认证码或后台连接程序。

## 实际替换与划分

13个缺损窗口的起点均提前30秒，全部重新核对六视角真实完整性。复用11952条已审计源记录，新增审计6条，总11958条；不补帧，不继承旧QA/证据。

替换选择按窗口顺序进行，并非对所有替代组合的全局最优证明。各窗口对的新增重叠累计420秒，不能解释为独立视频时长。最终有1个三窗连通组、28个双窗连通组和61个单窗组。

为保证重叠组同speaker及总体均衡，下列4个固定留出与原划分不同：

| 窗口 | 原留出 | 当前留出 |
|---|---|---|
| DAY3_20393000 | Lucia | Tasha |
| DAY4_22000000 | Lucia | Katrina |
| DAY6_19400000 | Katrina | Lucia |
| DAY7_14100000 | Tasha | Lucia |

## 本地检查

以下命令只检查纯Python数据逻辑，不提交任务、不加载GPU模型：

```bash
cd multi-user/training/grpo_v3/experiments/full120_r64_lrpair_20261013
python -m unittest test_pair_data test_media_quality test_overlap_policy -v
cd prerequisites
python -m unittest test_holdout_runtime test_speaker_holdout_split -v
```

完整CPU模块审计需要对应Torch环境、模型和准备后的图像；GPU更新、保存重载及终点评价必须另有真实运行证据。

## 环境依赖与复现边界

1. 原运行基于成功完成19460878的私有源码副本。Git分支基线 `a5fb5c65aa3c44108d1601038fec4740578836dd` 尚未包含该私有副本的全部同speaker留出扩展。本目录保留 `prerequisites/install_holdout_runtime.py` 及其辅助模块，不能将“脚本已上传”表述成整个训练源码已经与远端逐文件同步。
2. `submit_pair.py` 读取实验根的 `source_workflow.json`。该文件包含实际Python环境、Policy/Judge模型、Judge配置、CLIP、FFmpeg、Slurm账号和已验证的媒体/奖励设置。本快照没有凭空生成缺失的环境配置；须从对应已核验私有实验读取，再根据实际环境适配。
3. 源修复还读取 `existing_segments.json`；这是旧文件复用映射。没有旧分段时应明确建立空映射，而不是把其他实验的视频或QA当成本轮输入。
4. 提交器、安装器和历史恢复脚本中有本人账号、私有根路径及精确旧JobID防重复检查。这些是原实验来源，不能在任意目录无修改直接执行。新环境必须逐项适配路径和真实资源，保护已有提交清单。
5. 使用当前替换后的清单复现时，保留 `window_replacements_report.json` 的来源，直接准备当前媒体。不要把原缺损替换流程再执行一遍并覆盖已经固定的替换记录。需要复核原替换选择时使用 `history/` 中的替换前输入，在另一个私有目录完成。
6. `prerequisites/install_holdout_runtime.py` 是此前8窗speaker留出扩展的原脚本，会读取当时的 `preflight/formal200/workflow.json`；它不是本轮新的600/120配置生成器。应先理解该扩展或从已安装的来源复用，再应用本轮 `install_pair_runtime.py`。

## 提交来源

2026-10-14恢复提交的本轮清单：

| 阶段 | JobID |
|---|---:|
| CPU源修复、帧准备与模块审计 | 19754024 |
| 唯一r64/四窗口实际检查 | 19754025 |
| 1e-5、300更新 | 19754028 |
| 5e-6、300更新 | 19754030 |

旧19717659因公开源短视频失败；旧19722680因严格无新增重叠替换不可行而失败。两批旧GPU任务均由系统取消且0运行，旧JobID作为来源保留。本目录的上传操作不提交、恢复或取消任何Torch任务。
