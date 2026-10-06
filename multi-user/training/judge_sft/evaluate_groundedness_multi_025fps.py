"""Multi-checkpoint Groundedness eval using the 0.25 FPS collator used in training."""

from . import evaluate_groundedness_checkpoints_multi as _multi
from .collator_025fps_mb2 import JudgeFrameCollator

_multi.JudgeFrameCollator = JudgeFrameCollator

if __name__ == "__main__":
    _multi.main()
