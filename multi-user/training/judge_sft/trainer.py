"""Trainer that optimizes only the PASS/FAIL vocabulary-token margin."""

from __future__ import annotations

import inspect
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


class TensorParallelReplicatedSampler:
    """Emit one deterministic order identically on every tensor-parallel rank."""

    def __init__(
        self,
        dataset: JudgeDataset,
        *,
        seed: int,
    ) -> None:
        if not dataset.examples:
            raise ValueError("tensor-parallel sampler requires a non-empty dataset")
        self.dataset = dataset
        self.seed = int(seed)
        self.epoch = 0
        grouped: dict[tuple[int, int], list[int]] = defaultdict(list)
        for index, example in enumerate(dataset.examples):
            grouped[model_execution_signature(example)].append(index)
        self._grouped_indices = dict(grouped)

    def __len__(self) -> int:
        return len(self.dataset)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        generator = random.Random(self.seed + self.epoch)
        signature_groups: list[list[int]] = []
        for signature in sorted(self._grouped_indices):
            indices = list(self._grouped_indices[signature])
            generator.shuffle(indices)
            signature_groups.append(indices)
        generator.shuffle(signature_groups)
        for group in signature_groups:
            yield from group


def audit_tensor_parallel_sampler(
    dataset: JudgeDataset,
    *,
    tensor_parallel_size: int,
    seed: int,
    epochs: int = 2,
) -> dict[str, Any]:
    """Validate identical, lossless sample order across simulated TP ranks."""

    if tensor_parallel_size < 2:
        raise ValueError("tensor_parallel_size must be at least two")
    if epochs < 1:
        raise ValueError("sampler audit epochs must be positive")
    rank_samplers = [
        TensorParallelReplicatedSampler(dataset, seed=seed)
        for _ in range(tensor_parallel_size)
    ]
    for epoch in range(epochs):
        rank_orders = []
        for sampler in rank_samplers:
            sampler.set_epoch(epoch)
            rank_orders.append(list(sampler))
        if any(order != rank_orders[0] for order in rank_orders[1:]):
            raise RuntimeError("tensor-parallel ranks received different sample orders")
        indices = rank_orders[0]
        if sorted(indices) != list(range(len(dataset))):
            raise RuntimeError(
                "tensor-parallel sampler must include every example exactly once per epoch"
            )
    signature_counts: dict[str, int] = defaultdict(int)
    for example in dataset.examples:
        frame_sets, frames = model_execution_signature(example)
        signature_counts[f"{frame_sets}_frame_sets::{frames}_frames"] += 1
    return {
        "status": "passed",
        "tensor_parallel_size": tensor_parallel_size,
        "audited_epochs": epochs,
        "real_examples_per_epoch": len(dataset),
        "sampler_slots_per_rank_per_epoch": len(rank_samplers[0]),
        "identical_order_on_every_tp_rank": True,
        "zero_loss_padding_slots_per_epoch": 0,
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
        def __init__(
            self,
            *args: Any,
            verdict_token_ids: dict[Verdict, int],
            tensor_parallel_size: int,
            **kwargs: Any,
        ) -> None:
            super().__init__(*args, **kwargs)
            self.verdict_token_ids = verdict_token_ids
            self.tensor_parallel_size = int(tensor_parallel_size)
            self._logits_to_keep_name = logits_to_keep_argument(self.model)

        def save_model(
            self,
            output_dir: str | None = None,
            _internal_call: bool = False,
        ) -> None:
            """Collectively gather TP LoRA shards and write them on rank zero.

            Transformers 5.16 gates its normal ``_save`` call behind
            ``args.should_save``. That is correct for replicated DDP weights,
            but PEFT's TP adapter gathering calls DTensor collectives from
            ``save_pretrained``. Every TP rank must therefore enter
            ``save_pretrained`` even though only rank zero writes files.
            """

            if self.tensor_parallel_size <= 1:
                return super().save_model(output_dir, _internal_call)

            import os

            import torch
            from peft import PeftModel
            from transformers.trainer import TRAINING_ARGS_NAME

            if not torch.distributed.is_available() or not torch.distributed.is_initialized():
                raise RuntimeError("TP adapter saving requires an initialized process group")
            world_size = torch.distributed.get_world_size()
            if world_size != self.tensor_parallel_size:
                raise RuntimeError(
                    "TP adapter saving requires WORLD_SIZE == tensor_parallel_size: "
                    f"world_size={world_size} tp_size={self.tensor_parallel_size}"
                )
            if not isinstance(self.model, PeftModel):
                raise TypeError("TP adapter saving requires a PeftModel")

            output_dir = output_dir or self.args.output_dir
            is_writer = bool(self.args.should_save)
            if is_writer:
                os.makedirs(output_dir, exist_ok=True)
            torch.distributed.barrier()

            adapter_state_dict = {
                name: parameter
                for name, parameter in self.model.named_parameters()
                if "lora_" in name.lower()
            }
            if not adapter_state_dict:
                raise RuntimeError("TP adapter save found no LoRA parameters")
            # get_peft_model_state_dict runs before PEFT's is_main_process write
            # gate, so all ranks participate in DTensor.full_tensor() gathers.
            # Supplying an adapter-only state dict also prevents PEFT 0.21 from
            # gathering the frozen 27B base model to CPU at every checkpoint.
            self.model.save_pretrained(
                output_dir,
                state_dict=adapter_state_dict,
                safe_serialization=bool(
                    getattr(self.args, "save_safetensors", True)
                ),
                is_main_process=is_writer,
            )
            if is_writer:
                if self.processing_class is not None:
                    self.processing_class.save_pretrained(output_dir)
                elif (
                    self.data_collator is not None
                    and hasattr(self.data_collator, "tokenizer")
                    and self.data_collator.tokenizer is not None
                ):
                    self.data_collator.tokenizer.save_pretrained(output_dir)
                torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
            torch.distributed.barrier()

            if self.args.push_to_hub and not _internal_call and is_writer:
                raise RuntimeError(
                    "push_to_hub is disabled for the custom collective TP save path"
                )

        def _get_train_sampler(self, train_dataset: Any | None = None) -> Any:
            dataset = self.train_dataset if train_dataset is None else train_dataset
            if not isinstance(dataset, JudgeDataset):
                raise TypeError(
                    "verdict-token training requires JudgeDataset for replicated "
                    "tensor-parallel sampling"
                )
            if int(self.args.per_device_train_batch_size) != 1:
                raise ValueError(
                    "tensor-parallel judge sampling requires per-device batch size 1"
                )
            data_seed = self.args.data_seed
            if data_seed is None:
                data_seed = self.args.seed
            return TensorParallelReplicatedSampler(dataset, seed=int(data_seed))

        def _assert_replicated_tp_example(self, fingerprint: Any) -> None:
            """Fail before forward if TP ranks were accidentally given different rows."""

            import torch

            if not torch.distributed.is_available() or not torch.distributed.is_initialized():
                if self.tensor_parallel_size != 1:
                    raise RuntimeError("TP training requires an initialized process group")
                return
            world_size = torch.distributed.get_world_size()
            if world_size != self.tensor_parallel_size:
                raise RuntimeError(
                    "pure TP requires WORLD_SIZE == tensor_parallel_size: "
                    f"world_size={world_size} tp_size={self.tensor_parallel_size}"
                )
            local = fingerprint.reshape(1)
            gathered = [torch.empty_like(local) for _ in range(world_size)]
            torch.distributed.all_gather(gathered, local)
            values = [int(value.item()) for value in gathered]
            if len(set(values)) != 1:
                raise RuntimeError(
                    "tensor-parallel ranks received different examples: "
                    f"fingerprints={values}"
                )

        def _verdict_forward(self, model: Any, inputs: dict[str, Any]) -> tuple[Any, Any, Any]:
            model_inputs = dict(inputs)
            labels = model_inputs.pop("labels")
            sample_weights = model_inputs.pop("sample_weights")
            fingerprint = model_inputs.pop("tp_example_fingerprint")
            self._assert_replicated_tp_example(fingerprint)
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
