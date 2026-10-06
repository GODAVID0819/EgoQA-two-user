from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .data import JudgeDataset, load_normalized_manifest
from .trainer import (
    TensorParallelReplicatedSampler,
    audit_tensor_parallel_sampler,
    build_verdict_trainer_class as _build_base_trainer,
)


def _metrics(eval_pred):
    import numpy as np

    logits = np.asarray(eval_pred.predictions)
    labels = np.asarray(eval_pred.label_ids)

    y = labels[:, 0].astype(int)
    pred = logits.argmax(axis=-1).astype(int)

    acc = float((pred == y).mean())

    recalls = []
    for cls in (0, 1):
        mask = y == cls
        if mask.any():
            recalls.append(float((pred[mask] == cls).mean()))

    return {
        "accuracy": acc,
        "balanced_accuracy": float(np.mean(recalls)),
        "pred_pass_rate": float((pred == 1).mean()),
        "gold_pass_rate": float((y == 1).mean()),
    }


def build_verdict_trainer_class():
    Base = _build_base_trainer()

    class VerdictTokenTrainerMB2(Base):
        def _get_train_sampler(self, train_dataset=None):
            dataset = (
                self.train_dataset
                if train_dataset is None
                else train_dataset
            )

            if not isinstance(dataset, JudgeDataset):
                raise TypeError("expected JudgeDataset")

            seed = (
                self.args.data_seed
                if self.args.data_seed is not None
                else self.args.seed
            )

            return TensorParallelReplicatedSampler(
                dataset,
                seed=int(seed),
            )

        def _get_eval_sampler(self, eval_dataset):
            seed = (
                self.args.data_seed
                if self.args.data_seed is not None
                else self.args.seed
            )
            return TensorParallelReplicatedSampler(
                eval_dataset,
                seed=int(seed),
            )

        def _assert_replicated_tp_example(self, fingerprint):
            import torch

            if (
                not torch.distributed.is_available()
                or not torch.distributed.is_initialized()
            ):
                if self.tensor_parallel_size != 1:
                    raise RuntimeError("TP process group is not initialized")
                return

            local = fingerprint.reshape(-1).contiguous()

            gathered = [
                torch.empty_like(local)
                for _ in range(torch.distributed.get_world_size())
            ]
            torch.distributed.all_gather(gathered, local)

            values = [
                tuple(int(x) for x in tensor.tolist())
                for tensor in gathered
            ]

            if len(set(values)) != 1:
                raise RuntimeError(
                    f"TP ranks received different microbatches: {values}"
                )

        def __init__(self, *args, **kwargs):
            # Automatically attach the test manifest when provided.
            if kwargs.get("eval_dataset") is None:
                manifest = os.environ.get("JUDGE_EVAL_MANIFEST")

                if manifest:
                    train_dataset = kwargs.get("train_dataset")
                    train_task_ids = {
                        int(train_dataset[i].task_id)
                        for i in range(len(train_dataset))
                    }

                    eval_examples = [
                        x
                        for x in load_normalized_manifest(Path(manifest))
                        if int(x.task_id) in train_task_ids
                    ]

                    kwargs["eval_dataset"] = JudgeDataset(eval_examples)
                    kwargs["compute_metrics"] = _metrics

            super().__init__(*args, **kwargs)

    return VerdictTokenTrainerMB2
