from __future__ import annotations

import sys
from pathlib import Path


def pop_resume_checkpoint() -> str:
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

    checkpoint = str(
        Path(sys.argv[i + 1])
        .expanduser()
        .resolve()
    )

    del sys.argv[i:i + 2]

    return checkpoint


def main():
    checkpoint = pop_resume_checkpoint()

    if not Path(checkpoint).is_dir():
        raise RuntimeError(
            f"checkpoint not found: {checkpoint}"
        )

    # Inject true HF Trainer resume without modifying
    # the already-working launcher.
    from transformers import Trainer

    original_train = Trainer.train

    def resume_train(self, *args, **kwargs):
        kwargs["resume_from_checkpoint"] = checkpoint

        if self.is_world_process_zero():
            print(
                f"AUTO_RESUME_CHECKPOINT={checkpoint}",
                flush=True,
            )

        return original_train(
            self,
            *args,
            **kwargs,
        )

    Trainer.train = resume_train

    from training.judge_sft import train_025fps_mb1

    train_025fps_mb1.main()


if __name__ == "__main__":
    main()
