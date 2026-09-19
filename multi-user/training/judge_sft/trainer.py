"""Trainer that optimizes only the PASS/FAIL vocabulary-token margin."""

from __future__ import annotations

import inspect
import math
import random
from collections import defaultdict
from collections.abc import Iterator
from typing import Any

from .contracts import Verdict
from .data import JudgeDataset, JudgeExample
from .loss import verdict_bce_from_pair_logits, verdict_pair_logits


def model_execution_signature(example: JudgeExample) -> tuple[int, int]:
    """Return the branch-shaping media signature used by the Qwen forward pass."""

    return len(example.frame_sets), example.frame_count


class RankAlignedExecutionSampler:
    """Keep every data-parallel microstep on one ZeRO-3 module path.

    Accelerate shards consecutive per-device batches across ranks. This sampler
    therefore emits global chunks of ``world_size`` examples with an identical
    media signature. Short final chunks use negative indices, which
    ``JudgeDataset`` resolves to zero-loss padding copies of the same signature.
    """

    def __init__(
        self,
        dataset: JudgeDataset,
        *,
        world_size: int,
        seed: int,
    ) -> None:
        if world_size < 1:
            raise ValueError("world_size must be positive")
        if not dataset.examples:
            raise ValueError("rank-aligned sampler requires a non-empty dataset")
        self.dataset = dataset
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.epoch = 0
        grouped: dict[tuple[int, int], list[int]] = defaultdict(list)
        for index, example in enumerate(dataset.examples):
            grouped[model_execution_signature(example)].append(index)
        self._grouped_indices = dict(grouped)
        self._length = sum(
            math.ceil(len(indices) / self.world_size) * self.world_size
            for indices in self._grouped_indices.values()
        )

    @staticmethod
    def _padding_index(source_index: int) -> int:
        return -int(source_index) - 1

    def __len__(self) -> int:
        return self._length

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        generator = random.Random(self.seed + self.epoch)
        global_microbatches: list[list[int]] = []
        for signature in sorted(self._grouped_indices):
            indices = list(self._grouped_indices[signature])
            generator.shuffle(indices)
            for start in range(0, len(indices), self.world_size):
                chunk = indices[start : start + self.world_size]
                real_count = len(chunk)
                if real_count < self.world_size:
                    chunk.extend(
                        self._padding_index(chunk[offset % real_count])
                        for offset in range(self.world_size - real_count)
                    )
                global_microbatches.append(chunk)
        generator.shuffle(global_microbatches)
        for chunk in global_microbatches:
            yield from chunk


def audit_rank_aligned_sampler(
    dataset: JudgeDataset,
    *,
    world_size: int,
    seed: int,
    epochs: int = 2,
) -> dict[str, Any]:
    """Validate rank alignment before the expensive model load."""

    if epochs < 1:
        raise ValueError("sampler audit epochs must be positive")
    sampler = RankAlignedExecutionSampler(
        dataset,
        world_size=world_size,
        seed=seed,
    )
    padding_slots = 0
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        indices = list(sampler)
        if len(indices) % world_size:
            raise RuntimeError("rank-aligned sampler length is not world-size divisible")
        real_indices = [index for index in indices if index >= 0]
        if sorted(real_indices) != list(range(len(dataset))):
            raise RuntimeError(
                "rank-aligned sampler must include every real example exactly once per epoch"
            )
        for start in range(0, len(indices), world_size):
            chunk = [dataset[index] for index in indices[start : start + world_size]]
            signatures = {model_execution_signature(example) for example in chunk}
            if len(signatures) != 1:
                raise RuntimeError(
                    "rank-aligned sampler mixed model execution paths in one "
                    f"global microbatch: {sorted(signatures)}"
                )
        padding_slots = sum(index < 0 for index in indices)
    signature_counts: dict[str, int] = defaultdict(int)
    for example in dataset.examples:
        frame_sets, frames = model_execution_signature(example)
        signature_counts[f"{frame_sets}_frame_sets::{frames}_frames"] += 1
    return {
        "status": "passed",
        "world_size": world_size,
        "audited_epochs": epochs,
        "real_examples_per_epoch": len(dataset),
        "sampler_slots_per_epoch": len(sampler),
        "zero_loss_padding_slots_per_epoch": padding_slots,
        "execution_signature_counts": dict(sorted(signature_counts.items())),
    }


def logits_to_keep_argument(model: Any) -> str:
    """Find the model argument that avoids materializing long-sequence logits."""

    candidates = [model]
    get_base_model = getattr(model, "get_base_model", None)
    if callable(get_base_model):
        candidates.append(get_base_model())
    base_model = getattr(model, "base_model", None)
    if base_model is not None:
        candidates.append(base_model)
    for candidate in candidates:
        try:
            parameters = inspect.signature(candidate.forward).parameters
        except (TypeError, ValueError):
            continue
        if "logits_to_keep" in parameters:
            return "logits_to_keep"
        if "num_logits_to_keep" in parameters:
            return "num_logits_to_keep"
    raise RuntimeError(
        "model.forward must support logits_to_keep or num_logits_to_keep. "
        "Materializing vocabulary logits for every token of a 1,800-frame context is unsafe."
    )


def final_logits(outputs: Any) -> Any:
    logits = outputs.logits
    if logits.ndim != 3 or logits.shape[1] != 1:
        raise RuntimeError(
            "expected exactly one retained logits position; "
            f"received shape={tuple(logits.shape)}"
        )
    return logits[:, 0, :]


def build_verdict_trainer_class() -> type:
    from transformers import Trainer

    class VerdictTokenTrainer(Trainer):
        def __init__(self, *args: Any, verdict_token_ids: dict[Verdict, int], **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.verdict_token_ids = verdict_token_ids
            self._logits_to_keep_name = logits_to_keep_argument(self.model)

        def _get_train_sampler(self, train_dataset: Any | None = None) -> Any:
            dataset = self.train_dataset if train_dataset is None else train_dataset
            if not isinstance(dataset, JudgeDataset):
                raise TypeError(
                    "verdict-token training requires JudgeDataset for rank-aligned "
                    "ZeRO-3 sampling"
                )
            if int(self.args.per_device_train_batch_size) != 1:
                raise ValueError(
                    "rank-aligned ZeRO-3 sampling requires per-device batch size 1"
                )
            data_seed = self.args.data_seed
            if data_seed is None:
                data_seed = self.args.seed
            return RankAlignedExecutionSampler(
                dataset,
                world_size=max(1, int(self.args.world_size)),
                seed=int(data_seed),
            )

        def _verdict_forward(self, model: Any, inputs: dict[str, Any]) -> tuple[Any, Any, Any]:
            model_inputs = dict(inputs)
            labels = model_inputs.pop("labels")
            sample_weights = model_inputs.pop("sample_weights")
            model_inputs[self._logits_to_keep_name] = 1
            outputs = model(**model_inputs, return_dict=True)
            pair_logits = verdict_pair_logits(
                final_logits(outputs),
                fail_token_id=self.verdict_token_ids[Verdict.FAIL],
                pass_token_id=self.verdict_token_ids[Verdict.PASS],
            )
            loss = verdict_bce_from_pair_logits(
                pair_logits,
                labels[:, 0],
                sample_weights=sample_weights,
            )
            return loss, pair_logits, labels

        def compute_loss(
            self,
            model: Any,
            inputs: dict[str, Any],
            return_outputs: bool = False,
            num_items_in_batch: Any | None = None,
        ) -> Any:
            del num_items_in_batch
            loss, pair_logits, _ = self._verdict_forward(model, inputs)
            if return_outputs:
                return loss, {"logits": pair_logits}
            return loss

        def prediction_step(
            self,
            model: Any,
            inputs: dict[str, Any],
            prediction_loss_only: bool,
            ignore_keys: list[str] | None = None,
        ) -> tuple[Any, Any, Any]:
            del ignore_keys
            inputs = self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():
                import torch

                with torch.no_grad():
                    loss, pair_logits, labels = self._verdict_forward(model, inputs)
            if prediction_loss_only:
                return loss.detach(), None, None
            return loss.detach(), pair_logits.detach(), labels.detach()

    return VerdictTokenTrainer
