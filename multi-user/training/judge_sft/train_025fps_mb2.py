from __future__ import annotations

import os

from . import train as _base
from .collator_025fps_mb2 import JudgeFrameCollator
from .trainer_epoch_eval import (
    audit_tensor_parallel_sampler,
    build_verdict_trainer_class,
)


# Replace only the components used by this experimental entry point.
_base.JudgeFrameCollator = JudgeFrameCollator
_base.build_verdict_trainer_class = build_verdict_trainer_class
_base.audit_tensor_parallel_sampler = audit_tensor_parallel_sampler

_original_training_kwargs = _base._training_argument_kwargs


def _training_argument_kwargs(*args, **kwargs):
    out = _original_training_kwargs(*args, **kwargs)

    out["per_device_train_batch_size"] = 2
    out["per_device_eval_batch_size"] = 2

    eval_name = (
        "eval_strategy"
        if "eval_strategy" in out
        else "evaluation_strategy"
    )

    if os.environ.get("JUDGE_EVAL_MANIFEST"):
        out[eval_name] = "epoch"

    return out


_base._training_argument_kwargs = _training_argument_kwargs


def main():
    _base.main()


if __name__ == "__main__":
    main()
