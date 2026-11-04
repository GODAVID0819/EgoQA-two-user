# Aria 十二路异步窗口 GRPO 与 Judge 标注

本目录以三人 Aria 录像的不同相对时段构造十二路输入。每窗 5 分钟、75 帧；四包轮换提问者得到 48 个条件。以 checkpoint-348 作为策略和冻结参考的初值，两卡每步各处理一个 G4，共 24 步、192 槽。固定 F/E 与 C630；仅当 E>0.5 时调用 C，跳过记为 null；无多样性和单视角扣分。

## 训练源码
- `runtime/` 是独立运行目录的源码，包含组包、训练采样检查、评分服务、奖励、双卡入口、调度和终态验收。E 提示词采用作业 21758184 的修正版：11 个提供窗口、0.25 FPS、全部未剪枝采样帧。原始 21741478 使用旧 E 说明。
- 在集群上使用独立的可写运行目录。先配置 `configure_aria12.py` 中的已有运行根、Python 环境、模型与 scratch 路径；参考模板保留了历史实验路径，仓库不包含远端依赖、断点或数据。创建 logs/preflight/submissions 后运行 `python configure_aria12.py`。固定 `ARIA_ROOT` 为运行目录，`ARIA_PYTHON` 为训练环境 Python。
- `slurm/cpu_aria12.sbatch` 通过 `cpu_aria12.py` 复用既有三视角 JPEG 和特征，准备数据并验证 48 行、900 帧评审和实际训练器采样顺序。通过后，`slurm/train.sbatch` 通过 bootstrap 启动训练。时限、账户、存储和资源以实际环境为准。
- `analysis/` 包含训练指标采集与绘图。采集脚本指向 21758184 的原路径，新运行需替换作业号与目录。统计遵循全槽分母，跳过 C 不等于 C 作出了否决。

## 人工标注工具
- `review/sample.py 私有候选.json 入选题.json --count 200`：输入为记录数组，含 id、family（已审定的同事实组）、scores；保留其他字段。先保留三项全过与事实组去重后的 C 正例，再按 F/E/C 状态均衡抽样。C 未评与未通过分开。此通用入口不保证重建历史首批的具体选题顺序。
- `review/build_review.py 入选题.json metadata.jsonl 审核.html --dataset 数据集名 --revision 固定版本 --assignment 任务名`：输入还须含 seq、speaker、windows 和 qa 对象（question/options/correct/answer）。可用 --translations 提供以候选 id 为键的中译。模板只嵌入审核所需的题目、原视频分段链接与声明答案，不嵌入自动分数或助手标签。浏览器导出人工标签。
- 人工确认的标签单独保存。此分支并未声称在这些标签上完成新的 Judge 训练。私有 QA、审核结论、视频、权重以及设备认证代码均未打包。
