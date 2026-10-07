"""Deprecated resume entry point; kept so existing launch scripts still work.

Resume is now built into train.py: ``--resume-from-checkpoint DIR`` loads the
LoRA adapter before TP sharding, verifies it against the checkpoint file, and
lets Trainer restore optimizer/scheduler/RNG/trainer state. This module simply
forwards to train_025fps_mb1 with the original command line.

The previous implementation patched ``train.get_peft_model``, but train.py
imports ``get_peft_model`` from peft inside the function, so the patch never
applied: runs "resumed" from freshly initialized LoRA (B = 0) while restoring
the old optimizer state.
"""

from __future__ import annotations

import sys


def main() -> None:
    if "--resume-from-checkpoint" not in sys.argv:
        raise RuntimeError("--resume-from-checkpoint is required")
    from . import train_025fps_mb1

    train_025fps_mb1.main()


if __name__ == "__main__":
    main()
