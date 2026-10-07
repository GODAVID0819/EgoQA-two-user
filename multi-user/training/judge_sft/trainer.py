"""Trainer that optimizes only the PASS/FAIL vocabulary-token margin."""

from __future__ import annotations

import inspect
import json
import os
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path
from collections.abc import Iterator
from typing import Any

from .contracts import Verdict
from .data import JudgeDataset, JudgeExample
from .loss import verdict_bce_from_pair_logits, verdict_pair_logits


def model_execution_signature(example: JudgeExample) -> tuple[int, int]:
    """Return the branch-shaping media signature used by the Qwen forward pass."""

    return len(example.frame_sets), example.frame_count


def media_locality_key(
    example: JudgeExample,
) -> tuple[str, tuple[tuple[str, ...], ...]]:
    """Identify examples that can reuse the same decoded CPU images."""

    return (
        example.group_id,
        tuple(frame_set.frames for frame_set in example.frame_sets),
    )


class _LazyBatchWindow:
    """Stream one GA window without materializing all microbatches."""

    def __init__(self, epoch_iterator, count: int) -> None:
        self._epoch_iterator = epoch_iterator
        self._count = int(count)

        if self._count < 1:
            raise ValueError(
                "lazy batch window requires a positive count"
            )

    def __len__(self) -> int:
        return self._count

    def __iter__(self):
        for index in range(self._count):
            try:
                yield next(self._epoch_iterator)
            except StopIteration as error:
                raise RuntimeError(
                    "epoch iterator ended before expected GA window "
                    f"completed: expected={self._count}, "
                    f"observed={index}"
                ) from error


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

        grouped: dict[
            tuple[int, int],
            dict[tuple[str, tuple[tuple[str, ...], ...]], list[int]],
        ] = defaultdict(lambda: defaultdict(list))
        for index, example in enumerate(dataset.examples):
            grouped[model_execution_signature(example)][
                media_locality_key(example)
            ].append(index)
        self._grouped_indices = {
            signature: list(locality_buckets.values())
            for signature, locality_buckets in grouped.items()
        }

    def __len__(self) -> int:
        return len(self.dataset)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        generator = random.Random(self.seed + self.epoch)
        signature_groups: list[list[int]] = []

        for signature in sorted(self._grouped_indices):
            buckets = [list(bucket) for bucket in self._grouped_indices[signature]]
            generator.shuffle(buckets)
            indices: list[int] = []
            for bucket in buckets:
                generator.shuffle(bucket)
                indices.extend(bucket)
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
            raise RuntimeError(
                "tensor-parallel ranks received different sample orders"
            )

        indices = rank_orders[0]

        if sorted(indices) != list(range(len(dataset))):
            raise RuntimeError(
                "tensor-parallel sampler must include every example "
                "exactly once per epoch"
            )

    signature_counts: dict[str, int] = defaultdict(int)

    for example in dataset.examples:
        frame_sets, frames = model_execution_signature(example)
        signature_counts[
            f"{frame_sets}_frame_sets::{frames}_frames"
        ] += 1

    return {
        "status": "passed",
        "tensor_parallel_size": tensor_parallel_size,
        "audited_epochs": epochs,
        "real_examples_per_epoch": len(dataset),
        "sampler_slots_per_rank_per_epoch": len(rank_samplers[0]),
        "identical_order_on_every_tp_rank": True,
        "zero_loss_padding_slots_per_epoch": 0,
        "media_locality_buckets": sum(
            len(buckets)
            for buckets in rank_samplers[0]._grouped_indices.values()
        ),
        "execution_signature_counts": dict(
            sorted(signature_counts.items())
        ),
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
        "model.forward must support logits_to_keep or "
        "num_logits_to_keep. "
        "Materializing vocabulary logits for every token of a "
        "1,800-frame context is unsafe."
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
            probe_instrumentation: bool = False,
            probe_log_dir: str | os.PathLike[str] | None = None,
            preloaded_adapter_checkpoint: str | os.PathLike[str] | None = None,
            **kwargs: Any,
        ) -> None:
            super().__init__(*args, **kwargs)

            self.preloaded_adapter_checkpoint = (
                None
                if preloaded_adapter_checkpoint is None
                else Path(preloaded_adapter_checkpoint).resolve()
            )
            self.verdict_token_ids = verdict_token_ids
            self.tensor_parallel_size = int(tensor_parallel_size)
            self._logits_to_keep_name = logits_to_keep_argument(
                self.model
            )

            self.probe_instrumentation = bool(probe_instrumentation)
            self._probe_rank = int(os.environ.get("RANK", "0"))
            self._probe_log_dir = Path(
                probe_log_dir if probe_log_dir is not None else self.args.output_dir
            )
            self._probe_log_path = (
                self._probe_log_dir / f"probe_timeline_rank{self._probe_rank}.jsonl"
            )
            self._probe_records: list[dict[str, Any]] = []
            self._probe_micro_step = 0
            self._probe_current_step_index: int | None = None
            self._probe_current_batch_meta: dict[str, Any] | None = None
            self._probe_h2d_ms = 0.0
            self._probe_forward_ms = 0.0
            self._probe_loss_ms = 0.0
            self._probe_backward_ms = 0.0
            self._probe_prev_step_start_ns: int | None = None
            self._probe_prev_step_end_ns: int | None = None
            self._probe_last_optimizer_end_ns: int | None = None
            self._probe_step_memory: dict[str, Any] = {}
            self._probe_optimizer_wrapped = False

            if self.probe_instrumentation:
                self._probe_log_dir.mkdir(parents=True, exist_ok=True)
                self._probe_log_path.write_text("", encoding="utf-8")
                self._install_probe_backward_timer()

        def _probe_sync(self) -> None:
            if not self.probe_instrumentation:
                return
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()

        def _probe_memory_snapshot(self) -> dict[str, Any]:
            if not self.probe_instrumentation:
                return {}
            import torch

            if not torch.cuda.is_available():
                return {}
            device = torch.cuda.current_device()
            gib = float(1024**3)
            return {
                "device": int(device),
                "allocated_gib": torch.cuda.memory_allocated(device) / gib,
                "reserved_gib": torch.cuda.memory_reserved(device) / gib,
                "max_allocated_gib": torch.cuda.max_memory_allocated(device) / gib,
                "max_reserved_gib": torch.cuda.max_memory_reserved(device) / gib,
            }

        def _probe_write(self, record: dict[str, Any]) -> None:
            if not self.probe_instrumentation:
                return
            payload = {"rank": self._probe_rank} | record
            self._probe_records.append(payload)
            with self._probe_log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, sort_keys=True) + "\n")

        def _install_probe_backward_timer(self) -> None:
            original_backward = self.accelerator.backward

            def timed_backward(*args: Any, **kwargs: Any) -> Any:
                self._probe_sync()
                started_ns = time.monotonic_ns()
                result = original_backward(*args, **kwargs)
                self._probe_sync()
                ended_ns = time.monotonic_ns()
                self._probe_backward_ms = (ended_ns - started_ns) / 1e6
                self._probe_step_memory["after_backward"] = (
                    self._probe_memory_snapshot()
                )
                return result

            self.accelerator.backward = timed_backward

        def create_optimizer(self):
            optimizer = super().create_optimizer()
            if (
                self.probe_instrumentation
                and not self._probe_optimizer_wrapped
                and optimizer is not None
            ):
                original_step = optimizer.step

                def timed_optimizer_step(*args: Any, **kwargs: Any) -> Any:
                    self._probe_sync()
                    started_ns = time.monotonic_ns()
                    result = original_step(*args, **kwargs)
                    self._probe_sync()
                    ended_ns = time.monotonic_ns()
                    optimizer_ms = (ended_ns - started_ns) / 1e6
                    self._probe_last_optimizer_end_ns = ended_ns
                    self._probe_write(
                        {
                            "event": "optimizer_step",
                            "micro_step": self._probe_current_step_index,
                            "start_ns": started_ns,
                            "end_ns": ended_ns,
                            "optimizer_ms": optimizer_ms,
                            "memory": self._probe_memory_snapshot(),
                        }
                    )
                    return result

                optimizer.step = timed_optimizer_step
                self._probe_optimizer_wrapped = True
            return optimizer

        def _load_from_checkpoint(
            self,
            resume_from_checkpoint: str,
            model: Any | None = None,
        ) -> None:
            """Skip Trainer's adapter reload: train.py already loaded it pre-shard.

            Reloading through Trainer would copy full tensors into TP-sharded
            LoRA factors after sharding, which fails. Optimizer, scheduler, RNG
            and trainer state are still restored by Trainer as usual.
            """

            requested = Path(resume_from_checkpoint).resolve()
            if self.preloaded_adapter_checkpoint != requested:
                raise RuntimeError(
                    "TP resume must load the adapter before sharding; pass "
                    "--resume-from-checkpoint to train.py instead of calling "
                    f"Trainer.train(resume_from_checkpoint={requested}) directly "
                    f"(preloaded={self.preloaded_adapter_checkpoint})"
                )
            if self.is_world_process_zero():
                print(
                    f"resume_skip_trainer_adapter_reload={requested}",
                    flush=True,
                )

        def _prepare_inputs(self, inputs: dict[str, Any]) -> dict[str, Any]:
            if not self.probe_instrumentation:
                return super()._prepare_inputs(inputs)

            local_inputs = dict(inputs)
            probe_meta = local_inputs.pop("_probe_meta", None)
            self._probe_current_batch_meta = (
                dict(probe_meta) if isinstance(probe_meta, dict) else None
            )

            self._probe_sync()
            started_ns = time.monotonic_ns()
            prepared = super()._prepare_inputs(local_inputs)
            self._probe_sync()
            ended_ns = time.monotonic_ns()
            self._probe_h2d_ms = (ended_ns - started_ns) / 1e6
            self._probe_step_memory["after_h2d"] = self._probe_memory_snapshot()
            return prepared

        @staticmethod
        def _probe_overlap_ms(
            left_start_ns: int,
            left_end_ns: int,
            right_start_ns: int,
            right_end_ns: int,
        ) -> float:
            overlap_ns = max(
                0,
                min(left_end_ns, right_end_ns)
                - max(left_start_ns, right_start_ns),
            )
            return overlap_ns / 1e6

        def training_step(
            self,
            model: Any,
            inputs: dict[str, Any],
            num_items_in_batch: Any | None = None,
        ) -> Any:
            if not self.probe_instrumentation:
                return super().training_step(
                    model,
                    inputs,
                    num_items_in_batch=num_items_in_batch,
                )

            import torch

            self._probe_micro_step += 1
            self._probe_current_step_index = self._probe_micro_step
            self._probe_current_batch_meta = None
            self._probe_h2d_ms = 0.0
            self._probe_forward_ms = 0.0
            self._probe_loss_ms = 0.0
            self._probe_backward_ms = 0.0
            self._probe_step_memory = {}
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            self._probe_sync()
            step_start_ns = time.monotonic_ns()
            result = super().training_step(
                model,
                inputs,
                num_items_in_batch=num_items_in_batch,
            )
            self._probe_sync()
            step_end_ns = time.monotonic_ns()

            meta = self._probe_current_batch_meta or {}
            collate_start_ns = meta.get("collate_start_ns")
            collate_end_ns = meta.get("collate_end_ns")
            previous_gpu_end_ns = self._probe_last_optimizer_end_ns
            if previous_gpu_end_ns is None:
                previous_gpu_end_ns = self._probe_prev_step_end_ns

            overlap_ms = None
            hidden_pct = None
            cpu_exposed_wait_ms = None
            batch_ready_ahead_ms = None
            post_gpu_gap_ms = None
            if (
                collate_start_ns is not None
                and collate_end_ns is not None
                and self._probe_prev_step_start_ns is not None
                and previous_gpu_end_ns is not None
            ):
                overlap_ms = self._probe_overlap_ms(
                    int(collate_start_ns),
                    int(collate_end_ns),
                    self._probe_prev_step_start_ns,
                    previous_gpu_end_ns,
                )
                collate_ms = max(
                    0.0,
                    (int(collate_end_ns) - int(collate_start_ns)) / 1e6,
                )
                if collate_ms > 0:
                    hidden_pct = 100.0 * overlap_ms / collate_ms
                cpu_exposed_wait_ms = max(
                    0.0,
                    (int(collate_end_ns) - previous_gpu_end_ns) / 1e6,
                )
                batch_ready_ahead_ms = max(
                    0.0,
                    (previous_gpu_end_ns - int(collate_end_ns)) / 1e6,
                )
                post_gpu_gap_ms = max(
                    0.0,
                    (step_start_ns - previous_gpu_end_ns) / 1e6,
                )

            record = {
                "event": "train_step",
                "micro_step": self._probe_current_step_index,
                "start_ns": step_start_ns,
                "end_ns": step_end_ns,
                "train_step_ms": (step_end_ns - step_start_ns) / 1e6,
                "h2d_ms": self._probe_h2d_ms,
                "forward_ms": self._probe_forward_ms,
                "loss_ms": self._probe_loss_ms,
                "backward_ms": self._probe_backward_ms,
                "cpu_collate_overlap_prev_gpu_ms": overlap_ms,
                "cpu_collate_hidden_pct": hidden_pct,
                "cpu_exposed_wait_after_prev_gpu_ms": cpu_exposed_wait_ms,
                "batch_ready_before_prev_gpu_end_ms": batch_ready_ahead_ms,
                "post_prev_gpu_to_step_start_gap_ms": post_gpu_gap_ms,
                "batch": meta,
                "memory": self._probe_step_memory
                | {"after_training_step": self._probe_memory_snapshot()},
            }
            self._probe_write(record)
            self._probe_prev_step_start_ns = step_start_ns
            self._probe_prev_step_end_ns = step_end_ns
            return result

        def finalize_probe(self) -> dict[str, Any] | None:
            if not self.probe_instrumentation:
                return None

            step_records = [
                record
                for record in self._probe_records
                if record.get("event") == "train_step"
            ]
            optimizer_records = [
                record
                for record in self._probe_records
                if record.get("event") == "optimizer_step"
            ]
            steady = step_records[1:] if len(step_records) > 1 else step_records

            def mean_field(records: list[dict[str, Any]], key: str) -> float | None:
                values = [
                    float(record[key])
                    for record in records
                    if record.get(key) is not None
                ]
                return statistics.mean(values) if values else None

            peak_allocated = []
            peak_reserved = []
            for record in step_records:
                for snapshot in record.get("memory", {}).values():
                    if not isinstance(snapshot, dict):
                        continue
                    if "max_allocated_gib" in snapshot:
                        peak_allocated.append(float(snapshot["max_allocated_gib"]))
                    if "max_reserved_gib" in snapshot:
                        peak_reserved.append(float(snapshot["max_reserved_gib"]))
            for record in optimizer_records:
                snapshot = record.get("memory", {})
                if not isinstance(snapshot, dict):
                    continue
                if "max_allocated_gib" in snapshot:
                    peak_allocated.append(float(snapshot["max_allocated_gib"]))
                if "max_reserved_gib" in snapshot:
                    peak_reserved.append(float(snapshot["max_reserved_gib"]))

            summary = {
                "rank": self._probe_rank,
                "steps_completed": len(step_records),
                "steady_state_steps": len(steady),
                "mean_train_step_ms": mean_field(steady, "train_step_ms"),
                "mean_h2d_ms": mean_field(steady, "h2d_ms"),
                "mean_forward_ms": mean_field(steady, "forward_ms"),
                "mean_loss_ms": mean_field(steady, "loss_ms"),
                "mean_backward_ms": mean_field(steady, "backward_ms"),
                "mean_optimizer_ms": mean_field(optimizer_records[1:] or optimizer_records, "optimizer_ms"),
                "mean_cpu_collate_ms": (
                    statistics.mean(
                        [
                            float(record["batch"]["collate_ms"])
                            for record in steady
                            if isinstance(record.get("batch"), dict)
                            and record["batch"].get("collate_ms") is not None
                        ]
                    )
                    if any(
                        isinstance(record.get("batch"), dict)
                        and record["batch"].get("collate_ms") is not None
                        for record in steady
                    )
                    else None
                ),
                "mean_cpu_collate_hidden_pct": mean_field(
                    steady,
                    "cpu_collate_hidden_pct",
                ),
                "mean_cpu_worker_idle_before_ms": (
                    statistics.mean(
                        [
                            float(record["batch"]["worker_idle_before_ms"])
                            for record in steady
                            if isinstance(record.get("batch"), dict)
                            and record["batch"].get("worker_idle_before_ms") is not None
                        ]
                    )
                    if any(
                        isinstance(record.get("batch"), dict)
                        and record["batch"].get("worker_idle_before_ms") is not None
                        for record in steady
                    )
                    else None
                ),
                "decoded_image_cache_hit_rate": (
                    statistics.mean(
                        [
                            1.0 if bool(record["batch"].get("cache_hit")) else 0.0
                            for record in steady
                            if isinstance(record.get("batch"), dict)
                        ]
                    )
                    if steady
                    else None
                ),
                "mean_cpu_exposed_wait_after_prev_gpu_ms": mean_field(
                    steady,
                    "cpu_exposed_wait_after_prev_gpu_ms",
                ),
                "mean_post_prev_gpu_to_step_start_gap_ms": mean_field(
                    steady,
                    "post_prev_gpu_to_step_start_gap_ms",
                ),
                "peak_allocated_gib": max(peak_allocated) if peak_allocated else None,
                "peak_reserved_gib": max(peak_reserved) if peak_reserved else None,
                "timeline_jsonl": str(self._probe_log_path),
            }
            summary_path = (
                self._probe_log_dir
                / f"probe_summary_rank{self._probe_rank}.json"
            )
            summary_path.write_text(
                json.dumps(summary, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            return summary

        def get_batch_samples(
            self,
            epoch_iterator,
            num_batches: int,
            device,
        ):
            """
            Preserve GA semantics while streaming microbatches lazily.

            Transformers 5.16 normally materializes the entire
            gradient-accumulation window before training it. For our
            1,800-frame batches that is unsafe for CPU/GPU memory.

            num_items_in_batch is intentionally None because this
            custom verdict loss does not use it.
            """
            del device
            return _LazyBatchWindow(
                epoch_iterator,
                num_batches,
            ), None

        def get_train_dataloader(self):
            """
            Keep training batches on CPU until training_step() consumes them.

            Transformers 5.16 gathers an entire gradient-accumulation
            window before executing the individual forward/backward
            passes.

            Accelerate's DataLoaderShard normally moves every yielded
            batch to the target device immediately.

            For our 1,800-frame examples, pixel_values alone is about
            4.12 GiB. With GA=32, eager placement can therefore retain
            many huge visual batches on GPU simultaneously before the
            first forward pass.

            Setting dataloader.device = None disables eager device
            placement.

            Trainer.training_step() subsequently calls
            _prepare_inputs() on each individual microbatch, so the
            current microbatch is still transferred to GPU immediately
            before its forward/backward pass.
            """

            dataloader = super().get_train_dataloader()

            if not hasattr(dataloader, "device"):
                raise RuntimeError(
                    "Expected an Accelerate-prepared train dataloader "
                    "with a .device attribute"
                )

            # CRITICAL:
            #
            # Do not let Accelerate move every batch yielded during
            # Trainer.get_batch_samples() to GPU.
            #
            # The accumulation window can still contain 32 batches,
            # but those batches remain in CPU RAM.
            #
            # Trainer.training_step() moves only the currently consumed
            # microbatch to GPU through _prepare_inputs().
            dataloader.device = None

            return dataloader

        def save_model(
            self,
            output_dir: str | None = None,
            _internal_call: bool = False,
        ) -> None:
            """
            Collectively gather TP LoRA shards and write them on rank zero.

            Transformers 5.16 gates its normal _save call behind
            args.should_save. That is correct for replicated DDP
            weights, but PEFT's TP adapter gathering calls DTensor
            collectives from save_pretrained.

            Every TP rank must therefore enter save_pretrained even
            though only rank zero writes files.
            """

            if self.tensor_parallel_size <= 1:
                return super().save_model(
                    output_dir,
                    _internal_call,
                )

            import os

            import torch
            from peft import PeftModel
            from transformers.trainer import TRAINING_ARGS_NAME

            if (
                not torch.distributed.is_available()
                or not torch.distributed.is_initialized()
            ):
                raise RuntimeError(
                    "TP adapter saving requires an initialized "
                    "process group"
                )

            world_size = torch.distributed.get_world_size()

            if world_size != self.tensor_parallel_size:
                raise RuntimeError(
                    "TP adapter saving requires "
                    "WORLD_SIZE == tensor_parallel_size: "
                    f"world_size={world_size} "
                    f"tp_size={self.tensor_parallel_size}"
                )

            if not isinstance(self.model, PeftModel):
                raise TypeError(
                    "TP adapter saving requires a PeftModel"
                )

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
                raise RuntimeError(
                    "TP adapter save found no LoRA parameters"
                )

            # get_peft_model_state_dict runs before PEFT's
            # is_main_process write gate, so all ranks participate in DTensor.full_tensor() gathers.
            #
            # Supplying an adapter-only state dict also prevents PEFT
            # 0.21 from gathering the frozen 27B base model to CPU at
            # every checkpoint.
            self.model.save_pretrained(
                output_dir,
                state_dict=adapter_state_dict,
                safe_serialization=bool(
                    getattr(
                        self.args,
                        "save_safetensors",
                        True,
                    )
                ),
                is_main_process=is_writer,
            )

            if is_writer:
                if self.processing_class is not None:
                    self.processing_class.save_pretrained(
                        output_dir
                    )

                elif (
                    self.data_collator is not None
                    and hasattr(
                        self.data_collator,
                        "tokenizer",
                    )
                    and self.data_collator.tokenizer is not None
                ):
                    self.data_collator.tokenizer.save_pretrained(
                        output_dir
                    )

                torch.save(
                    self.args,
                    os.path.join(
                        output_dir,
                        TRAINING_ARGS_NAME,
                    ),
                )

            torch.distributed.barrier()

            if (
                self.args.push_to_hub
                and not _internal_call
                and is_writer
            ):
                raise RuntimeError(
                    "push_to_hub is disabled for the custom "
                    "collective TP save path"
                )

        def _get_train_sampler(
            self,
            train_dataset: Any | None = None,
        ) -> Any:
            dataset = (
                self.train_dataset
                if train_dataset is None
                else train_dataset
            )

            if not isinstance(dataset, JudgeDataset):
                raise TypeError(
                    "verdict-token training requires JudgeDataset "
                    "for replicated tensor-parallel sampling"
                )

            if int(self.args.per_device_train_batch_size) != 1:
                raise ValueError(
                    "tensor-parallel judge sampling requires "
                    "per-device batch size 1"
                )

            data_seed = self.args.data_seed

            if data_seed is None:
                data_seed = self.args.seed

            return TensorParallelReplicatedSampler(
                dataset,
                seed=int(data_seed),
            )

        def _assert_replicated_tp_example(
            self,
            fingerprint: Any,
        ) -> None:
            """
            Fail before forward if TP ranks were accidentally given
            different rows.
            """

            import torch

            if (
                not torch.distributed.is_available()
                or not torch.distributed.is_initialized()
            ):
                if self.tensor_parallel_size != 1:
                    raise RuntimeError(
                        "TP training requires an initialized "
                        "process group"
                    )

                return

            world_size = torch.distributed.get_world_size()

            if world_size != self.tensor_parallel_size:
                raise RuntimeError(
                    "pure TP requires "
                    "WORLD_SIZE == tensor_parallel_size: "
                    f"world_size={world_size} "
                    f"tp_size={self.tensor_parallel_size}"
                )

            local = fingerprint.reshape(1)

            gathered = [
                torch.empty_like(local)
                for _ in range(world_size)
            ]

            torch.distributed.all_gather(
                gathered,
                local,
            )

            values = [
                int(value.item())
                for value in gathered
            ]

            if len(set(values)) != 1:
                raise RuntimeError(
                    "tensor-parallel ranks received "
                    "different examples: "
                    f"fingerprints={values}"
                )

        def _verdict_forward(
            self,
            model: Any,
            inputs: dict[str, Any],
        ) -> tuple[Any, Any, Any]:
            model_inputs = dict(inputs)

            labels = model_inputs.pop("labels")
            sample_weights = model_inputs.pop(
                "sample_weights"
            )
            fingerprint = model_inputs.pop(
                "tp_example_fingerprint"
            )

            self._assert_replicated_tp_example(
                fingerprint
            )

            model_inputs[
                self._logits_to_keep_name
            ] = 1

            if self.probe_instrumentation:
                self._probe_sync()
                forward_start_ns = time.monotonic_ns()
            outputs = model(
                **model_inputs,
                return_dict=True,
            )
            if self.probe_instrumentation:
                self._probe_sync()
                forward_end_ns = time.monotonic_ns()
                self._probe_forward_ms = (
                    forward_end_ns - forward_start_ns
                ) / 1e6
                self._probe_step_memory["after_forward"] = (
                    self._probe_memory_snapshot()
                )
                loss_start_ns = time.monotonic_ns()

            pair_logits = verdict_pair_logits(
                final_logits(outputs),
                fail_token_id=self.verdict_token_ids[
                    Verdict.FAIL
                ],
                pass_token_id=self.verdict_token_ids[
                    Verdict.PASS
                ],
            )

            loss = verdict_bce_from_pair_logits(
                pair_logits,
                labels[:, 0],
                sample_weights=sample_weights,
            )
            if self.probe_instrumentation:
                self._probe_sync()
                loss_end_ns = time.monotonic_ns()
                self._probe_loss_ms = (
                    loss_end_ns - loss_start_ns
                ) / 1e6
                self._probe_step_memory["after_loss"] = (
                    self._probe_memory_snapshot()
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

            loss, pair_logits, _ = self._verdict_forward(
                model,
                inputs,
            )

            if return_outputs:
                return loss, {
                    "logits": pair_logits,
                }

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
                    (
                        loss,
                        pair_logits,
                        labels,
                    ) = self._verdict_forward(
                        model,
                        inputs,
                    )

            if prediction_loss_only:
                return (
                    loss.detach(),
                    None,
                    None,
                )

            return (
                loss.detach(),
                pair_logits.detach(),
                labels.detach(),
            )

    return VerdictTokenTrainer