"""Experimental grouped gradient checkpointing for Qwen3.5 text decoder."""

from __future__ import annotations

import inspect
from types import MethodType
from typing import Any


def install_grouped_decoder_checkpointing(
    model: Any,
    *,
    first_trainable_layer: int,
    segment_size: int = 2,
) -> dict[str, Any]:
    """
    Replace per-layer checkpointing in the trainable Qwen3.5 decoder suffix
    with one non-reentrant checkpoint per N consecutive decoder layers.

    Frozen prefix layers run normally outside autograd.
    """

    import torch

    from transformers.cache_utils import DynamicCache
    from transformers.masking_utils import (
        create_causal_mask,
        create_recurrent_attention_mask,
    )
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5ModelOutputWithPast,
    )

    if segment_size < 1:
        raise ValueError("segment_size must be positive")

    # Locate the actual text transformer inside PEFT + multimodal wrappers.
    matches = [
        (name, module)
        for name, module in model.named_modules()
        if module.__class__.__name__ == "Qwen3_5TextModel"
    ]

    if len(matches) != 1:
        raise RuntimeError(
            "Expected exactly one Qwen3_5TextModel; "
            f"found={[name for name, _ in matches]}"
        )

    module_name, text_model = matches[0]

    layers = text_model.layers
    total_layers = len(layers)

    if total_layers != int(text_model.config.num_hidden_layers):
        raise RuntimeError(
            "Decoder depth mismatch: "
            f"module={total_layers} config={text_model.config.num_hidden_layers}"
        )

    if not 0 <= first_trainable_layer < total_layers:
        raise ValueError(
            f"bad first_trainable_layer={first_trainable_layer}, "
            f"total_layers={total_layers}"
        )

    trainable_count = total_layers - first_trainable_layer

    if trainable_count % segment_size != 0:
        raise ValueError(
            "Trainable decoder suffix must divide evenly into checkpoint "
            f"segments: trainable={trainable_count} segment_size={segment_size}"
        )

    # Fail closed if installed Transformers has a substantially different
    # Qwen3.5 forward implementation from the one this patch targets.
    source = inspect.getsource(text_model.__class__.forward)

    required_source_fragments = [
        "causal_mask_mapping",
        "create_causal_mask",
        "create_recurrent_attention_mask",
        "position_embeddings",
        "decoder_layer",
        "self.layers",
    ]

    missing = [
        fragment
        for fragment in required_source_fragments
        if fragment not in source
    ]

    if missing:
        raise RuntimeError(
            "Installed Qwen3.5 forward differs from expected implementation; "
            f"missing source markers={missing}"
        )

    # HF already configured this function with use_reentrant=False.
    checkpoint_func = getattr(
        layers[first_trainable_layer],
        "_gradient_checkpointing_func",
        None,
    )

    if checkpoint_func is None:
        raise RuntimeError(
            "Decoder layers have no _gradient_checkpointing_func. "
            "Call model.gradient_checkpointing_enable() before installing "
            "grouped checkpointing."
        )

    # IMPORTANT:
    # Disable decoder-layer-local checkpointing everywhere in the text stack.
    # The trainable suffix will instead be checkpointed explicitly below.
    #
    # Prefix 0..47 is frozen and remains outside autograd.
    original_gc_flags = []

    for index, layer in enumerate(layers):
        original_gc_flags.append(
            bool(getattr(layer, "gradient_checkpointing", False))
        )
        layer.gradient_checkpointing = False

    segment_starts = list(
        range(first_trainable_layer, total_layers, segment_size)
    )

    def grouped_forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        **kwargs,
    ):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Training path only. We deliberately do not maintain KV cache.
        if self.training:
            use_cache = False
        elif use_cache is None:
            use_cache = bool(getattr(self.config, "use_cache", False))

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        # Qwen3.5 multimodal RoPE uses four rows initially:
        # text, temporal, height, width.
        if position_ids is None:
            past_seen_tokens = (
                past_key_values.get_seq_length()
                if past_key_values is not None
                else 0
            )

            position_ids = (
                torch.arange(
                    inputs_embeds.shape[1],
                    device=inputs_embeds.device,
                )
                + past_seen_tokens
            )

            position_ids = position_ids.view(1, 1, -1).expand(
                4,
                inputs_embeds.shape[0],
                -1,
            )

        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(
                4,
                position_ids.shape[0],
                -1,
            )

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            rope_position_ids = position_ids[1:]
        else:
            text_position_ids = None
            rope_position_ids = position_ids

        if isinstance(attention_mask, dict):
            causal_mask_mapping = attention_mask
        else:
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": text_position_ids,
            }

            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "linear_attention": create_recurrent_attention_mask(
                    **mask_kwargs
                ),
            }

        hidden_states = inputs_embeds

        position_embeddings = self.rotary_emb(
            hidden_states,
            rope_position_ids,
        )

        # ------------------------------------------------------------
        # Frozen prefix: layers 0 .. first_trainable_layer-1
        #
        # No checkpointing. Parameters are frozen and the hidden state
        # entering layer 48 must still have requires_grad=False.
        # ------------------------------------------------------------
        for i in range(first_trainable_layer):
            decoder_layer = self.layers[i]

            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask_mapping[
                    self.config.layer_types[i]
                ],
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )

        # ------------------------------------------------------------
        # Trainable suffix:
        #
        #   [48,49] checkpoint
        #   [50,51] checkpoint
        #   ...
        #
        # One saved checkpoint boundary per segment rather than per layer.
        # ------------------------------------------------------------
        for start in segment_starts:
            end = start + segment_size

            segment_layers = tuple(self.layers[start:end])
            segment_indices = tuple(range(start, end))

            def run_segment(
                h,
                *,
                _layers=segment_layers,
                _indices=segment_indices,
            ):
                for layer, layer_index in zip(_layers, _indices):
                    h = layer(
                        h,
                        position_embeddings=position_embeddings,
                        attention_mask=causal_mask_mapping[
                            self.config.layer_types[layer_index]
                        ],
                        position_ids=text_position_ids,
                        past_key_values=past_key_values,
                        use_cache=False,
                        **kwargs,
                    )

                return h

            if self.training and torch.is_grad_enabled():
                hidden_states = checkpoint_func(
                    run_segment,
                    hidden_states,
                )
            else:
                hidden_states = run_segment(hidden_states)

        hidden_states = self.norm(hidden_states)

        return Qwen3_5ModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )

    text_model.forward = MethodType(grouped_forward, text_model)

    # Keep an explicit audit trail on the module itself.
    text_model._judge_grouped_checkpointing = {
        "segment_size": segment_size,
        "first_trainable_layer": first_trainable_layer,
        "trainable_layer_count": trainable_count,
        "segment_starts": segment_starts,
        "segments": [
            list(range(start, start + segment_size))
            for start in segment_starts
        ],
        "original_decoder_gc_flags": original_gc_flags,
    }

    return {
        "text_model_module": module_name,
        "total_decoder_layers": total_layers,
        "first_trainable_layer": first_trainable_layer,
        "trainable_layer_count": trainable_count,
        "segment_size": segment_size,
        "segments": [
            list(range(start, start + segment_size))
            for start in segment_starts
        ],
        "inner_decoder_checkpointing_disabled": True,
        "outer_nonreentrant_grouped_checkpointing": True,
    }
