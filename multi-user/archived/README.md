# 归档代码

这里保存历史 GRPO 代际代码。归档区用于复现和查证历史实验，不属于当前活跃 QA 生成-评审主流程，也不属于活跃 GRPOv3 训练路径。

当前活跃入口：

- QA 主流程：顶层 `egolife_two_user_qa/` 与 `hpc/qa/`
- GRPOv3：`training/grpo_v3/` 与 `hpc/grpo_v3/`

归档规则：

- active 代码不得 import `archived/`。
- 归档代码只保证核心历史测试和历史入口尽量可复跑。
- 归档代码不保证继续适配未来主线 API。
