# EgoLife Video-First QA Judger 设计说明

这份 note 记录为什么要把 EgoLife two-user QA 的 judger 拆成多个 blocking dimensions。它的目标不是替代 answerability test，而是在更早阶段发现坏问题，并给 generator 提供可执行的修改反馈。

## 为什么单个 Judge Bit 太弱

当前任务比普通 VQA 质量控制更严格。一个合格的问题必须同时满足：

- 问法自然，像第一视角用户自己或 AR assistant 会问的问题；
- 答案必须来自原始 egocentric videos，而不是 caption、外部知识或上帝视角描述；
- 至少两个 required users 的视频都必须贡献必要证据；
- 单个 required user 的视频不能完整回答；
- 合并全部 required users 的视频后，必须能唯一推出正确选项。

如果 judger 只输出一个 `review_passed`，失败原因会被压缩在一个 bool 里。Generator 得到的反馈就会很模糊，比如只知道“这题不好”，但不知道是问题口吻不自然、题型不匹配、单视频泄漏答案，还是 evidence 不够具体。这样 retry 很容易重复同样错误。

新版设计参考了三个方向：

- LLM-as-judge 工作，例如 G-Eval，通常会用 form-filling/rubric，而不是一个不结构化的整体偏好判断。
- Self-refinement 工作说明，反馈循环只有在 feedback 明确指出要改什么时才有用。
- 长视频 QA benchmark，例如 CinePile 和 EgoSchema，强调“困难性”不能只靠视频长度假设，而要验证 evidence scope。对我们来说，这个 evidence scope 不是单纯长时间上下文，而是多用户视频范围。

## 推荐的 Blocking Judge Dimensions

### 1. `agent_perspective`

检查问题是否是自然的第一视角问法。

它主要阻止：

- “in the video / in the frame / what does the camera show” 这种数据集观察者口吻；
- 直接把 required users 都写成第三人称，而不是合理使用 `I/me/my/we/our`；
- 从上帝视角描述画面，而不是像用户回忆或 AR assistant 提问。

### 2. `source_scope`

检查答案是否只能从原始视频和 packet metadata 判断。

它主要阻止：

- 依赖 caption 或 pre-written observation；
- 依赖外部常识才能确认答案；
- 猜测人物身份、心理状态、意图；
- 使用 off-screen facts 或视频中不可见的信息。

### 3. `question_type_semantics`

检查题型语义是否真的匹配 `question_type`。

- `commonality` 应该问两个或多个用户共同看到、共同经历、或共同验证的事实；
- `difference` 应该问用户视角之间的差异、不对称信息，或互补细节。

这个 check 主要防止 generator 标了 `commonality`，但实际问的是单用户细节；或者标了 `difference`，但其实只是普通共同事件。

### 4. `multi_video_necessity`

检查这道题是否真的需要多个视频。

它要求：

- 每个 required user 都贡献必要且非重复的 visual clue；
- 任何单个 required user 的视频都不能完整回答；
- 如果是同一天同一时间 token 的 clips，问题应该围绕同一个事件、地点、任务或相关互动，而不是硬拼两个无关视频。

这是目前最关键的维度，因为我们的目标不是“把两个 user 放进题目里”，而是构造必须联合两个 user 才能答的问题。

### 5. `visual_grounding`

检查正确答案是否真的有视频证据支撑。

它要求：

- 正确选项能对应到具体可见时刻；
- `per_user_evidence_claims` 要说明每个 user 的视频分别贡献了什么；
- `referred_timestamps` 如果存在，要足够具体，方便后续人眼核查。

这个 check 用来减少“听起来合理但视频里看不清”的题。

### 6. `mcq_option_quality`

检查五选一 MCQ 的选项质量。

它要求：

- 正好五个非空选项；
- 只有一个正确答案；
- distractors 要合理但不能也正确；
- 选项之间不能因为语义太接近而产生歧义。

### 7. `gaze_safety`

检查 gaze 相关表述是否保守。

EgoLife EyeGaze 默认是 Project Aria CPF yaw/pitch/depth，不是 image pixel gaze。所以如果 `projection_status` 不是 `projected`，题目不能说“我看向某个物体中心”或“gaze point 靠近某个 bbox”这类 pixel-level claim。

### 8. `human_auditability`

检查这条 QA 是否给后续人工审核保留了足够信息。

它要求 row 里有：

- 对应用户的视频路径或视频 URL；
- day / time token / clip clock；
- referred timestamps；
- generator rationale；
- why_two_users_needed；
- per_user_evidence_claims；
- generation/judge/answerability 的 intermediate trace。

这不是单纯格式问题，而是保证后续我们能人工判断这条题到底为什么被接受或拒绝。

## 和 Answerability Evaluation 的关系

Judger 是 semantic + format gate。它回答的是：

> 这道题从定义上、问法上、证据说明上，是否应该被认为是合格 two-user QA？

Answerability evaluation 是 behavioral gate。它回答的是：

> 在不同视频可见范围下，模型实际能不能答对？

具体测试条件是：

- `single_user::<user>`：只给某一个用户的视频，应该答错或返回 insufficient；
- `proper_subset::<users>`：对 3 个及以上 user 的题，只给部分 users，应该答错或 insufficient；
- `combined_all_users::<users>`：给全部 required users 的视频，应该选中正确答案。

最终一条 QA 只有同时通过 judger 和 answerability gate，才会被写入 accepted `qa_mcq.jsonl`。

## 当前代码中的实现

当前实现主要落在这些文件：

- `prompts.py`
  - 定义 generator prompt、structured judger prompt、answerability prompt；
  - `build_judger_prompt()` 要求模型输出 9 个 checks，每个 check 都有 `status/reason/fix`。

- `video_qa_loop.py`
  - `judge_gate()` 读取 structured judger output；
  - 如果任意 blocking check 不是 `PASS`，即使 `review_passed=True` 也会拒绝；
  - `generation_trace` 保存每次 attempt 的 prompt、raw output、judge feedback 和 answerability 结果。

- `schema.py`
  - strict validation 要求 `review.judger.checks` 存在；
  - strict validation 要求 `video_evidence`、`human_audit`、`generation_trace` 存在。

- `tests/test_core.py`
  - 测试 structured judge checks；
  - 测试 `judge_gate` 会拒绝失败维度；
  - 测试 dry-run QA 会保存 video provenance 和 trace。

## References

- G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment:
  https://arxiv.org/abs/2303.16634
- Self-Refine: Iterative Refinement with Self-Feedback:
  https://arxiv.org/abs/2303.17651
- Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena:
  https://arxiv.org/abs/2306.05685
- Large Language Models are not Fair Evaluators:
  https://arxiv.org/abs/2305.17926
- CinePile: A Long Video Question Answering Dataset and Benchmark:
  https://arxiv.org/abs/2405.08813
- EgoSchema: A Diagnostic Benchmark for Very Long-form Video Language Understanding:
  https://arxiv.org/abs/2308.09126
- EgoTaskQA: Understanding Human Tasks in Egocentric Videos:
  https://arxiv.org/abs/2210.03929
- MA-EgoQA: Question Answering over Egocentric Videos from Multiple Embodied Agents:
  https://arxiv.org/abs/2603.09827
