# MiniCPM-V 4.5 / EgoLife 131题：已完成

正式作业21448178已COMPLETED/0:0，独立审计VERIFIED_393。北京时间2026-10-31 03:55:09完成全部393条预测。

| 条件 | 正确数 | 正确率 |
|---|---:|---:|
| evidence2 | 97/131 | 74.05% |
| speaker1 | 63/131 | 48.09% |
| all6 | 89/131 | 67.94% |

证据视角比六视角高6.11个百分点；排除17题复审fail标记后114题仍高6.14个百分点。69窗口重采样区间[0.00,12.59]个百分点，点估计达到5，稳定至少5仍未获证明。完整结果、格式核验、成本和限制见RESULTS_CN.md；精确统计见FINAL_STATISTICS.json。

## 本次输入

模型openbmb/MiniCPM-V-4_5，版本daef484c35ec93210ec93c5e901f8f3e9b78ee34。Qwen3语言骨干+SigLIP2+3D-Resampler，补充现有模型的视觉压缩路线。

题集tianxia2/egolife-six-user-mcqa-textwrong131，版本90e6e9bfbbe4b2f009bb37fdf25c1920e972a4a5，全131题，无二次选项重排。

沿用原grpo_frames帧清单和文本：speaker300帧，provider原keep-mask；原156预缩放，官方处理器实际448×448。每用户独立每6帧一组，余下不足6帧也保留；原文件名时间按0.1秒编码。共同视角在三条件中的分组、时间、尺寸和缓存特征一致。原生压缩会改变内部图像编码方式，跨模型结果属于整套模型配置比较。

Answer+Reason，关闭深度思考、greedy、beam1、repetition_penalty1.0、max_new_tokens512、seed0、batch4。原生上下文40960，实际最大19434输入+512输出=19946，无截断或上下文扩展。

## 入口和代码

- packing.py：用户边界、六帧组和真实时间。
- native_inputs.py：调用官方文字转换，核对图像占位符；缓存生成路径移除无用时间/尺寸字段。
- prepare_cpu.py：全393输入容量、官方真实图片路径与缓存文字路径等价、远端帧路径/尺寸检查。
- evaluate.py：官方视觉编码缓存和batch4答题，逐批保存原始输出、gold、帧路径和token数量。
- audit.py：独立重算393条、配对差值、130/114子集和窗口重采样区间。
- cpu_prepare.sbatch、formal.sbatch、formal_retry.sbatch：本次准确资源和环境设置。
- test_packing.py、test_native_cache.py、test_cached_device.py：CPU目标检查；无预训练权重或GPU测试。
- upstream/：对应模型版本的官方源码、处理器和分词器；reference_grpo/：原GRPO帧版参考代码。
- requirements-inference.txt、ENVIRONMENT.json：已核验核心版本。当前CPU脚本复用旧CUDA环境的只读包，在独立新环境安装Transformers4.51.3和tokenizers0.21.4。

## 路径和追溯

远端根：/scratch/xl6775/experiments/vqa_minicpmv45_grpo131_20261031

原始预测：outputs/formal_21448178/下三个predictions.jsonl及final_audit.json、paired_summary.json、run_config.json、run_status.json、model_loading.json。原始预测未下载，本地统计是远端聚合查询产生的衍生文件。

帧缓存只读复用：/scratch/xl6775/experiments/vqa_videollama3_grpo131_20261030/frames_156

新模型环境：/scratch/xl6775/envs/minicpmv45_grpo131_20261031

CPU21444333完成4:33；首跑21444890因设备搬运接口错误失败1:40、0预测，CPU复现修复后完整正式21448178完成42:27。失败代码在archive_21444890保留。正式任务和失败首跑合计约0.735 GPU小时。

提交记录jobs/cpu_submission.json、jobs/formal_submission.json、jobs/formal_retry_21444890_submission.json；本地亦保存对应清单。所有脚本保留防重复提交记录，复现时使用独立新实验目录。共享Torch连接与认证状态不随代码包交付。

CPU预取.1已完成；同GPU尾部预编码.2因主进度接近，0组安全退出，无预测或缓存贡献。主任务完成不代表长期低利用率保护已获验证，详细GPU采样与Slurm记账区别见结果报告。

## 验收

最终独立审计核对131个同题号配对、原始帧/用户/gold、解析、真实输入长度/尺寸和模型版本。两个主对比条件均131条严格Answer格式、0未解析；speaker-only有1条主动表示无法确定，按原规则计错。393条均有Reason段，1条证据视角输出达到512 token上限。

结果报告RESULTS_CN.md和完整代码minicpmv45_grpo131_complete_code.zip可直接交付；代码包包含题集元数据，不含权重、媒体、环境、运行缓存或凭据。
## GitHub 版本的输入准备

在本目录执行以下命令，获取此次运行使用的两个题集输入文件。媒体和模型权重按 experiment_config.json 中的版本另行准备。运行脚本保留正式实验的 Torch 路径。

```bash
hf download tianxia2/egolife-six-user-mcqa-textwrong131 dataset.json vqa_shuffled.json --repo-type dataset --revision 90e6e9bfbbe4b2f009bb37fdf25c1920e972a4a5 --local-dir .
```
