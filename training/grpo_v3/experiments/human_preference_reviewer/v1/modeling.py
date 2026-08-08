"""Three-head model contract for Reviewer v1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

OUTPUT_FIELDS = ("evidence_logits", "answerability_logits", "formality_logits")
HEAD_MODULES = {
    "evidence_quality": "evidence_head",
    "answerability": "answerability_head",
    "qa_formality": "formality_head",
}


def head_module_names(active_heads: Sequence[str]) -> tuple[str, ...]:
    unsupported = [name for name in active_heads if name not in HEAD_MODULES]
    if unsupported:
        raise ValueError(f"unsupported active head: {unsupported}")
    if not active_heads:
        raise ValueError("at least one active head is required")
    return tuple(HEAD_MODULES[name] for name in active_heads)


def last_nonpadding_indices_reference(attention_mask: Sequence[Sequence[int]]) -> list[int]:
    result: list[int] = []
    for row in attention_mask:
        active = [index for index, value in enumerate(row) if int(value) != 0]
        if not active:
            raise ValueError("attention mask row contains no active token")
        result.append(active[-1])
    return result


@dataclass
class ReviewerOutput:
    evidence_logits: Any | None = None
    answerability_logits: Any | None = None
    formality_logits: Any | None = None
    pooled_hidden: Any | None = None


try:
    import torch
    from torch import nn
except ImportError:  # Local contract tests do not require the H100 environment.
    torch = None
    nn = None


if nn is not None:
    class ReviewerV1(nn.Module):
        """Wrap a Qwen3-VL backbone with three independent classification heads."""

        def __init__(
            self,
            backbone: nn.Module,
            hidden_size: int,
            active_heads: Sequence[str] = ("evidence_quality", "answerability", "qa_formality"),
        ) -> None:
            super().__init__()
            if hidden_size <= 0:
                raise ValueError("hidden_size must be positive")
            self.backbone = backbone
            self.hidden_size = int(hidden_size)
            self.active_heads = tuple(active_heads)
            for module_name in head_module_names(self.active_heads):
                setattr(self, module_name, nn.Linear(hidden_size, 3))

        def _last_hidden(self, inputs: dict[str, Any]) -> Any:
            base_model = getattr(self.backbone, "model", None)
            if base_model is not None:
                outputs = base_model(**inputs, use_cache=False, return_dict=True)
                hidden = getattr(outputs, "last_hidden_state", None)
                if hidden is not None:
                    return hidden
            outputs = self.backbone(
                **inputs,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden = getattr(outputs, "last_hidden_state", None)
            if hidden is not None:
                return hidden
            hidden_states = getattr(outputs, "hidden_states", None)
            if not hidden_states:
                raise RuntimeError("backbone output does not expose a final hidden state")
            return hidden_states[-1]

        def forward(self, **inputs: Any) -> ReviewerOutput:
            attention_mask = inputs.get("attention_mask")
            if attention_mask is None:
                raise ValueError("attention_mask is required for Reviewer v1 pooling")
            if not torch.all(attention_mask.sum(dim=-1) > 0):
                raise ValueError("attention mask row contains no active token")
            hidden = self._last_hidden(dict(inputs))
            positions = torch.arange(attention_mask.shape[-1], device=attention_mask.device)
            positions = positions.unsqueeze(0).expand_as(attention_mask)
            last_indices = positions.masked_fill(attention_mask == 0, -1).max(dim=-1).values
            batch_indices = torch.arange(hidden.shape[0], device=hidden.device)
            pooled = hidden[batch_indices, last_indices.to(hidden.device)]
            # Keep numerically stable FP32 heads while preserving gradients into BF16 LoRA.
            first_head = getattr(self, head_module_names(self.active_heads)[0])
            head_input = pooled.to(dtype=first_head.weight.dtype)
            return ReviewerOutput(
                evidence_logits=(self.evidence_head(head_input) if hasattr(self, "evidence_head") else None),
                answerability_logits=(
                    self.answerability_head(head_input) if hasattr(self, "answerability_head") else None
                ),
                formality_logits=(self.formality_head(head_input) if hasattr(self, "formality_head") else None),
                pooled_hidden=pooled,
            )
else:
    class ReviewerV1:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any) -> None:
            raise RuntimeError("PyTorch is required to construct ReviewerV1")


def resolve_hidden_size(config: Any) -> int:
    text_config = getattr(config, "text_config", None)
    for value in (getattr(text_config, "hidden_size", None), getattr(config, "hidden_size", None)):
        if isinstance(value, int) and value > 0:
            return value
    raise ValueError("model config does not expose a positive text hidden_size")
