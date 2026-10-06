from __future__ import annotations

import os
import sys
from pathlib import Path


def pop_resume_checkpoint() -> Path:
    flag = "--resume-from-checkpoint"

    if flag not in sys.argv:
        raise RuntimeError(
            "--resume-from-checkpoint is required"
        )

    i = sys.argv.index(flag)

    if i + 1 >= len(sys.argv):
        raise RuntimeError(
            "--resume-from-checkpoint requires a path"
        )

    checkpoint = (
        Path(sys.argv[i + 1])
        .expanduser()
        .resolve()
    )

    del sys.argv[i:i + 2]

    if not checkpoint.is_dir():
        raise RuntimeError(
            f"checkpoint not found: {checkpoint}"
        )

    return checkpoint


def main() -> None:
    checkpoint = pop_resume_checkpoint()

    from peft import PeftModel
    from transformers import Trainer

    # Import the real base training module first so we can patch the
    # exact globals used by load_model_and_processor().
    from training.judge_sft import train as base

    # ------------------------------------------------------------
    # 1. Load adapter BEFORE LoRA TP materialization.
    #
    # Normal training does:
    #   get_peft_model(...)
    #   synchronize_lora_initialization(...)
    #   materialize_lora_tensor_parallelism(...)
    #
    # For resume we replace get_peft_model(...) with
    # PeftModel.from_pretrained(checkpoint). At this point LoRA factors
    # are still ordinary tensors, so PEFT can load them safely.
    # The existing training code will then synchronize + shard them.
    # ------------------------------------------------------------

    def get_peft_model_from_checkpoint(
        model,
        peft_config,
        *args,
        **kwargs,
    ):
        del peft_config, args, kwargs

        rank = os.environ.get("RANK", "?")
        print(
            f"[rank{rank}] "
            f"TP_RESUME_ADAPTER_PRE_SHARD={checkpoint}",
            flush=True,
        )

        return PeftModel.from_pretrained(
            model,
            str(checkpoint),
            is_trainable=True,
            autocast_adapter_dtype=False,
        )

    base.get_peft_model = get_peft_model_from_checkpoint

    # ------------------------------------------------------------
    # 2. Trainer.train(resume_from_checkpoint=...) must still be used
    #    so optimizer/scheduler/trainer-state/RNG are restored.
    #
    # But Trainer's first resume action is model.load_adapter(...),
    # which would try to load the same full Tensor checkpoint AGAIN,
    # now after LoRA has become DTensor.
    #
    # Skip ONLY that model reload. Leave the rest of HF resume intact.
    # ------------------------------------------------------------

    def skip_hf_model_reload(
        self,
        *args,
        **kwargs,
    ):
        rank = os.environ.get("RANK", "?")
        print(
            f"[rank{rank}] "
            "TP_RESUME_SKIP_HF_MODEL_RELOAD="
            f"{checkpoint}",
            flush=True,
        )
        return None

    Trainer._load_from_checkpoint = skip_hf_model_reload

    # ------------------------------------------------------------
    # 3. Inject resume checkpoint into Trainer.train().
    # ------------------------------------------------------------

    original_train = Trainer.train

    def train_with_resume(
        self,
        *args,
        **kwargs,
    ):
        kwargs["resume_from_checkpoint"] = str(
            checkpoint
        )

        if self.is_world_process_zero():
            print(
                "TP_TRUE_RESUME_CHECKPOINT="
                f"{checkpoint}",
                flush=True,
            )

        return original_train(
            self,
            *args,
            **kwargs,
        )

    Trainer.train = train_with_resume

    # Import after patches so the 0.25-FPS/MB1 launcher uses them.
    from training.judge_sft import (
        train_025fps_mb1,
    )

    train_025fps_mb1.main()


if __name__ == "__main__":
    main()
