"""Trainer that optimizes only the PASS/FAIL vocabulary-token margin."""

from __future__ import annotations

import inspect
from typing import Any

from .contracts import Verdict
from .loss import verdict_bce_from_pair_logits, verdict_pair_logits


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
