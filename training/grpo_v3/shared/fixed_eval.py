"""GRPO v3 固定集生成的共享辅助函数。"""

from __future__ import annotations

import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any


CHECKPOINT_STEPS = (0, 40)
FIXED_EVAL_MIN_VIDEO_PIXELS = 4 * 32 * 32


def set_generation_seed(seed: int, *, torch_module: Any | None = None) -> None:
    """在每个候选生成前重设所有可控随机源。"""

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32))
    except ModuleNotFoundError:
        pass
    if torch_module is None:
        try:
            import torch as torch_module
        except ModuleNotFoundError:
            torch_module = None
    if torch_module is not None:
        torch_module.manual_seed(seed)
        cuda = getattr(torch_module, "cuda", None)
        if cuda is not None and cuda.is_available():
            cuda.manual_seed_all(seed)


def fixed_eval_prompt(row: Mapping[str, Any]) -> str:
    """移除双视频占位符，返回实际送入 runner 的文本。"""

    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) != 1:
        raise ValueError("fixed eval 数据必须恰好包含一条 user message")
    content = str(messages[0].get("content") or "")
    marker = "<video><video>\n"
    if not content.startswith(marker):
        raise ValueError("fixed eval prompt 必须以两个有序 <video> 占位符开头")
    return content[len(marker) :]


def load_multi_adapter_runner(
    *,
    model_path: str,
    adapters: Mapping[int, Path],
    max_new_tokens: int,
    max_image_pixels: int,
    dtype: str = "bfloat16",
) -> Any:
    """加载共享基座以及 step 0/40 两个只读 LoRA adapter。"""

    from peft import PeftModel
    from qwen3vl_runner import Qwen3VLTransformersRunner

    if set(adapters) != set(CHECKPOINT_STEPS):
        raise ValueError("runner adapters 必须严格为 step 0 和 step 40")
    if max_image_pixels < FIXED_EVAL_MIN_VIDEO_PIXELS:
        raise ValueError("fixed eval max_image_pixels must cover the Qwen3-VL video minimum")
    runner = Qwen3VLTransformersRunner(
        model_id=model_path,
        max_new_tokens=max_new_tokens,
        max_image_pixels=max_image_pixels,
        min_video_pixels=FIXED_EVAL_MIN_VIDEO_PIXELS,
        dtype=dtype,
    )
    runner.model = PeftModel.from_pretrained(
        runner.model,
        str(adapters[0]),
        adapter_name="step_0",
        is_trainable=False,
    )
    runner.model.load_adapter(
        str(adapters[40]), adapter_name="step_40", is_trainable=False
    )
    runner.model.eval()
    runner.device = next(runner.model.parameters()).device
    return runner
