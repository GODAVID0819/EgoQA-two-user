"""Binary-first judge inference with continuous contract generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .contracts import VERDICT_ASSISTANT_PREFIX, Verdict, render_locked_verdict_prefix


@dataclass(frozen=True)
class VerdictDecision:
    verdict: Verdict
    pass_probability: float
    pass_minus_fail_logit: float
    unrestricted_top_token_id: int
    unrestricted_top_is_verdict: bool

    @property
    def locked_json_prefix(self) -> str:
        """Canonical prefix to use if explanation generation continues afterward."""

        return render_locked_verdict_prefix(self.verdict)


@dataclass(frozen=True)
class GeneratedJudgeOutput:
    """One continuous generation whose first token is the locked verdict."""

    decision: VerdictDecision
    text: str
    continuation: str
    generated_token_ids: tuple[int, ...]


class VerdictFirstTokenLogitsProcessor:
    """Choose PASS/FAIL from their logits at step one, then become a no-op.

    The original first-step scores are retained for audit. Only the selected
    verdict token remains finite on the first generation step; every later
    token is generated normally in the same ``model.generate`` call.
    """

    def __init__(
        self,
        *,
        prompt_length: int,
        fail_token_id: int,
        pass_token_id: int,
        pass_threshold: float = 0.5,
    ) -> None:
        if prompt_length <= 0:
            raise ValueError("prompt_length must be positive")
        if fail_token_id == pass_token_id:
            raise ValueError("PASS and FAIL token ids must differ")
        if not 0.0 <= pass_threshold <= 1.0:
            raise ValueError("pass_threshold must be between zero and one")
        self.prompt_length = int(prompt_length)
        self.fail_token_id = int(fail_token_id)
        self.pass_token_id = int(pass_token_id)
        self.pass_threshold = float(pass_threshold)
        self.first_step_scores: Any | None = None
        self.selected_token_ids: Any | None = None

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        if int(input_ids.shape[-1]) != self.prompt_length:
            return scores
        if self.first_step_scores is not None:
            raise RuntimeError("verdict logits processor was invoked twice at step one")

        import torch

        self.first_step_scores = scores.detach().clone()
        fail_scores = scores[:, self.fail_token_id]
        pass_scores = scores[:, self.pass_token_id]
        pass_probability = torch.sigmoid(pass_scores - fail_scores)
        pass_ids = torch.full_like(
            input=pass_probability,
            fill_value=self.pass_token_id,
            dtype=torch.long,
        )
        fail_ids = torch.full_like(
            input=pass_probability,
            fill_value=self.fail_token_id,
            dtype=torch.long,
        )
        selected = torch.where(
            pass_probability >= self.pass_threshold,
            pass_ids,
            fail_ids,
        )
        constrained = torch.full_like(scores, -torch.inf)
        constrained.scatter_(1, selected[:, None], 0.0)
        self.selected_token_ids = selected.detach().clone()
        return constrained


def select_verdict_from_next_token_logits(
    next_token_logits: Any,
    *,
    verdict_token_ids: Mapping[Verdict, int],
    pass_threshold: float = 0.5,
) -> VerdictDecision:
    """Make a two-label decision without trusting unconstrained generated text.

    The verdict is selected only from the PASS and FAIL vocabulary logits.  The
    unrestricted top token is returned as a diagnostic; it never changes or
    silently substitutes for the binary verdict.
    """

    import torch

    logits = next_token_logits
    if not isinstance(logits, torch.Tensor):
        logits = torch.as_tensor(logits)
    if logits.ndim != 1:
        raise ValueError(
            f"next_token_logits must have shape [vocab], got {tuple(logits.shape)}"
        )
    if not 0.0 <= pass_threshold <= 1.0:
        raise ValueError("pass_threshold must be between zero and one")
    fail_id = int(verdict_token_ids[Verdict.FAIL])
    pass_id = int(verdict_token_ids[Verdict.PASS])
    if fail_id == pass_id:
        raise ValueError("PASS and FAIL token ids must differ")
    if min(fail_id, pass_id) < 0 or max(fail_id, pass_id) >= logits.shape[0]:
        raise ValueError("verdict token id is outside the vocabulary logits")

    margin = logits[pass_id] - logits[fail_id]
    pass_probability = torch.sigmoid(margin)
    verdict = (
        Verdict.PASS
        if float(pass_probability.detach().cpu()) >= pass_threshold
        else Verdict.FAIL
    )
    top_id = int(torch.argmax(logits).detach().cpu())
    return VerdictDecision(
        verdict=verdict,
        pass_probability=float(pass_probability.detach().cpu()),
        pass_minus_fail_logit=float(margin.detach().cpu()),
        unrestricted_top_token_id=top_id,
        unrestricted_top_is_verdict=top_id in {fail_id, pass_id},
    )


def generate_complete_judge_output(
    model: Any,
    tokenizer: Any,
    model_inputs: Mapping[str, Any],
    *,
    verdict_token_ids: Mapping[Verdict, int],
    max_new_tokens: int = 160,
    pass_threshold: float = 0.5,
) -> GeneratedJudgeOutput:
    """Lock the first verdict token and continue the same generation to EOS.

    The caller must supply inputs that already end with ``{"verdict":"``.
    Only generation step one is constrained. The remaining JSON is generated
    autoregressively without a second prompt or second inference stage.
    """

    if max_new_tokens < 2:
        raise ValueError("max_new_tokens must allow a verdict and JSON continuation")
    input_ids = model_inputs.get("input_ids")
    if input_ids is None or getattr(input_ids, "ndim", None) != 2:
        raise ValueError("model_inputs must contain rank-two input_ids")
    if int(input_ids.shape[0]) != 1:
        raise ValueError("complete judge generation currently requires batch size one")

    fail_id = int(verdict_token_ids[Verdict.FAIL])
    pass_id = int(verdict_token_ids[Verdict.PASS])
    prompt_length = int(input_ids.shape[-1])
    locker = VerdictFirstTokenLogitsProcessor(
        prompt_length=prompt_length,
        fail_token_id=fail_id,
        pass_token_id=pass_id,
        pass_threshold=pass_threshold,
    )
    from transformers import LogitsProcessorList

    generated = model.generate(
        **dict(model_inputs),
        max_new_tokens=int(max_new_tokens),
        do_sample=False,
        num_beams=1,
        use_cache=True,
        logits_processor=LogitsProcessorList([locker]),
        return_dict_in_generate=True,
    )
    if locker.first_step_scores is None or locker.selected_token_ids is None:
        raise RuntimeError("generation ended before the verdict token was selected")

    generated_ids = generated.sequences[0, prompt_length:]
    if int(generated_ids.numel()) == 0:
        raise RuntimeError("generation returned no continuation tokens")
    selected_id = int(locker.selected_token_ids[0].detach().cpu())
    actual_first_id = int(generated_ids[0].detach().cpu())
    if actual_first_id != selected_id:
        raise RuntimeError(
            "generation did not honor the locked verdict token: "
            f"selected={selected_id} generated={actual_first_id}"
        )

    decision = select_verdict_from_next_token_logits(
        locker.first_step_scores[0],
        verdict_token_ids=verdict_token_ids,
        pass_threshold=pass_threshold,
    )
    continuation = str(
        tokenizer.decode(
            generated_ids.detach().cpu().tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    )
    return GeneratedJudgeOutput(
        decision=decision,
        text=VERDICT_ASSISTANT_PREFIX + continuation,
        continuation=continuation,
        generated_token_ids=tuple(
            int(value) for value in generated_ids.detach().cpu().tolist()
        ),
    )
