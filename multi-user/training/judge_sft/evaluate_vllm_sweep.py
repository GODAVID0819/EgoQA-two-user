"""Fast vLLM checkpoint sweep for EgoLife binary judge evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from .collator import (
    _apply_chat_template,
    _assert_thinking_disabled,
    adaptive_image_max_pixels,
    qwen_vision_geometry,
    render_frame_order_blocks,
)
from .contracts import VERDICT_ASSISTANT_PREFIX


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8-sig") as f:
        for ln, line in enumerate(f, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{ln}: expected JSON object")
            out.append(obj)
    return out


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def packet_frames(
    row: dict[str, Any],
) -> tuple[list[str], list[tuple[str, int]]]:

    packet_dir = Path(
        str(row["frame_packet"])
    ).expanduser().resolve()

    packet = json.loads(
        (packet_dir / "packet.json").read_text(
            encoding="utf-8-sig"
        )
    )

    users = list(packet.get("users") or [])

    user_indices = [
        int(x)
        for x in row.get("frame_user_indices") or []
    ]

    labels = [
        str(x)
        for x in row.get("frame_order") or []
    ]

    if len(user_indices) != len(labels):
        raise ValueError(
            f"{row['example_id']}: "
            "frame_user_indices/frame_order mismatch"
        )

    paths = []
    blocks = []

    for user_index, label in zip(
        user_indices,
        labels,
        strict=True,
    ):
        user = users[user_index]
        frames = list(user.get("frames") or [])

        blocks.append(
            (label, len(frames))
        )

        for frame in frames:
            rel = Path(
                str(frame.get("path") or "")
            )

            if (
                rel.is_absolute()
                or ".." in rel.parts
            ):
                raise ValueError(
                    f"{row['example_id']}: "
                    f"invalid frame path {rel}"
                )

            p = (
                packet_dir / rel
            ).resolve()

            if not p.is_file():
                raise FileNotFoundError(p)

            paths.append(str(p))

    return paths, blocks


def media_key(
    row: dict[str, Any],
) -> tuple[Any, ...]:

    return (
        str(
            Path(
                str(row["frame_packet"])
            ).resolve()
        ),
        tuple(
            int(x)
            for x
            in row.get(
                "frame_user_indices"
            ) or []
        ),
        tuple(
            str(x)
            for x
            in row.get(
                "frame_order"
            ) or []
        ),
    )


def media_uuid(
    path: str,
    min_pixels: int,
    max_pixels: int,
) -> str:

    s = (
        "egolife-v1|"
        f"{Path(path).resolve()}|"
        f"{min_pixels}|"
        f"{max_pixels}"
    )

    return hashlib.sha256(
        s.encode()
    ).hexdigest()


def build_messages(
    row: dict[str, Any],
    frame_paths: list[str],
    blocks: list[tuple[str, int]],
    min_pixels: int,
    max_pixels: int,
) -> list[dict[str, Any]]:

    content = [
        {
            "type": "image",
            "image": p,
            "min_pixels": min_pixels,
            "max_pixels": max_pixels,
        }
        for p in frame_paths
    ]

    content.append(
        {
            "type": "text",
            "text": (
                render_frame_order_blocks(
                    blocks
                )
                + str(row["prompt"])
            ),
        }
    )

    return [
        {
            "role": "user",
            "content": content,
        }
    ]


def render_prompt(
    processor: Any,
    messages: list[dict[str, Any]],
) -> str:

    text = _apply_chat_template(
        processor,
        messages,
    )

    _assert_thinking_disabled(text)

    return (
        text
        + VERDICT_ASSISTANT_PREFIX
    )


def one_token_id(
    tokenizer: Any,
    text: str,
) -> int:

    ids = tokenizer.encode(
        text,
        add_special_tokens=False,
    )

    if len(ids) != 1:
        raise RuntimeError(
            f"{text!r} is not one token: {ids}"
        )

    return int(ids[0])


def logprob_value(x: Any) -> float:
    return float(
        x.logprob
        if hasattr(x, "logprob")
        else x
    )


def requested_logprob(
    output: Any,
    token_id: int,
) -> float:

    seq = output.outputs[0].logprobs

    if not seq:
        raise RuntimeError(
            "vLLM returned no "
            "output-token logprobs"
        )

    first = seq[0]

    if token_id in first:
        return logprob_value(
            first[token_id]
        )

    if str(token_id) in first:
        return logprob_value(
            first[str(token_id)]
        )

    raise RuntimeError(
        f"token {token_id} absent "
        "from returned logprobs; "
        f"keys={list(first)[:20]}"
    )


def bce(
    margin: float,
    target: int,
) -> float:

    return (
        max(margin, 0.0)
        - margin * target
        + math.log1p(
            math.exp(
                -abs(margin)
            )
        )
    )


def sigmoid(x: float) -> float:

    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)

    z = math.exp(x)
    return z / (1.0 + z)


def summarize(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:

    n = len(rows)

    tp = sum(
        r["gold"] == 1
        and r["pred"] == 1
        for r in rows
    )

    tn = sum(
        r["gold"] == 0
        and r["pred"] == 0
        for r in rows
    )

    fp = sum(
        r["gold"] == 0
        and r["pred"] == 1
        for r in rows
    )

    fn = sum(
        r["gold"] == 1
        and r["pred"] == 0
        for r in rows
    )

    recall_pass = (
        tp / (tp + fn)
        if tp + fn
        else 0.0
    )

    recall_fail = (
        tn / (tn + fp)
        if tn + fp
        else 0.0
    )

    return {
        "n": n,
        "raw_bce": (
            sum(r["bce"] for r in rows)
            / n
        ),
        "accuracy": (
            (tp + tn) / n
        ),
        "balanced_accuracy": (
            0.5
            * (
                recall_pass
                + recall_fail
            )
        ),
        "gold_pass_rate": (
            (tp + fn) / n
        ),
        "pred_pass_rate": (
            (tp + fp) / n
        ),
        "mean_margin": (
            sum(
                r["margin"]
                for r in rows
            )
            / n
        ),
        "mean_abs_margin": (
            sum(
                abs(r["margin"])
                for r in rows
            )
            / n
        ),
        "recall_pass": recall_pass,
        "recall_fail": recall_fail,
        "confusion": {
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        },
    }


def close_images(
    images: list[Any],
) -> None:

    seen = set()

    for image in images:

        if id(image) in seen:
            continue

        seen.add(id(image))

        fn = getattr(
            image,
            "close",
            None,
        )

        if callable(fn):
            fn()


def parse_args() -> argparse.Namespace:

    p = argparse.ArgumentParser()

    p.add_argument(
        "--model-id",
        required=True,
    )

    p.add_argument(
        "--eval-manifest",
        type=Path,
        required=True,
    )

    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    p.add_argument(
        "--checkpoints",
        type=Path,
        nargs="*",
        default=[],
    )

    p.add_argument(
        "--include-base",
        action="store_true",
    )

    p.add_argument(
        "--task",
        default="groundedness",
    )

    p.add_argument(
        "--max-examples",
        type=int,
    )

    p.add_argument(
        "--min-pixels",
        type=int,
        default=3136,
    )

    p.add_argument(
        "--max-pixels",
        type=int,
        default=262144,
    )

    p.add_argument(
        "--max-input-tokens",
        type=int,
        default=262144,
    )

    p.add_argument(
        "--image-context-target-fraction",
        type=float,
        default=0.85,
    )

    p.add_argument(
        "--image-text-token-reserve",
        type=int,
        default=8192,
    )

    p.add_argument(
        "--image-item-token-overhead",
        type=int,
        default=2,
    )

    p.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.92,
    )

    p.add_argument(
        "--max-num-seqs",
        type=int,
        default=4,
    )

    p.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=32768,
    )

    p.add_argument(
        "--mm-processor-cache-gb",
        type=float,
        default=16.0,
    )

    p.add_argument(
        "--max-lora-rank",
        type=int,
        default=16,
    )

    p.add_argument(
        "--attention-backend",
        default="auto",
    )

    return p.parse_args()


def main() -> int:

    args = parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = [
        r
        for r in read_jsonl(
            args.eval_manifest
        )
        if str(
            r.get("task")
        ) == args.task
    ]

    if args.max_examples is not None:
        rows = rows[
            : args.max_examples
        ]

    if not rows:
        raise RuntimeError(
            f"no {args.task!r} rows "
            f"in {args.eval_manifest}"
        )

    for row in rows:
        if not row.get(
            "frame_packet"
        ):
            raise RuntimeError(
                f"{row['example_id']}: "
                "missing frame_packet"
            )

    checkpoints = [
        p.expanduser().resolve()
        for p in args.checkpoints
    ]

    for p in checkpoints:
        if not (
            p / "adapter_config.json"
        ).is_file():
            raise FileNotFoundError(
                p / "adapter_config.json"
            )

    model_specs = []

    if args.include_base:
        model_specs.append(
            ("base", None)
        )

    for i, p in enumerate(
        checkpoints,
        1,
    ):
        model_specs.append(
            (
                p.name,
                LoRARequest(
                    p.name,
                    i,
                    str(p),
                ),
            )
        )

    if not model_specs:
        raise RuntimeError(
            "nothing to evaluate"
        )

    grouped = defaultdict(list)

    for row in rows:
        grouped[
            media_key(row)
        ].append(row)

    groups = list(
        grouped.values()
    )

    for group in groups:
        group.sort(
            key=lambda r:
            str(r["example_id"])
        )

    max_images = max(
        len(
            packet_frames(
                group[0]
            )[0]
        )
        for group in groups
    )

    processor = (
        AutoProcessor
        .from_pretrained(
            args.model_id,
            trust_remote_code=True,
        )
    )

    geometry = (
        qwen_vision_geometry(
            processor
        )
    )

    llm_kwargs = {
        "model": args.model_id,
        "dtype": "bfloat16",
        "trust_remote_code": True,
        "max_model_len":
            args.max_input_tokens,
        "gpu_memory_utilization":
            args.gpu_memory_utilization,
        "max_num_seqs":
            args.max_num_seqs,
        "max_num_batched_tokens":
            args.max_num_batched_tokens,
        "enable_chunked_prefill":
            True,
        "enable_prefix_caching":
            True,
        "limit_mm_per_prompt": {
            "image": max_images
        },
        "mm_processor_cache_gb":
            args.mm_processor_cache_gb,
        "logprobs_mode":
            "raw_logprobs",
    }

    if (
        args.attention_backend.lower()
        != "auto"
    ):
        llm_kwargs[
            "attention_backend"
        ] = args.attention_backend

    if checkpoints:

        llm_kwargs.update(
            enable_lora=True,
            max_lora_rank=
                args.max_lora_rank,

            # Only one giant-context
            # LoRA request active at once.
            max_loras=1,

            # Keep remaining adapters
            # in CPU cache.
            max_cpu_loras=max(
                1,
                len(checkpoints),
            ),
        )

    print(
        "=== vLLM engine ===",
        flush=True,
    )

    for k, v in llm_kwargs.items():
        print(
            f"{k}={v}",
            flush=True,
        )

    llm = LLM(
        **llm_kwargs
    )

    tokenizer = (
        llm.get_tokenizer()
    )

    pass_id = one_token_id(
        tokenizer,
        "pass",
    )

    fail_id = one_token_id(
        tokenizer,
        "fail",
    )

    print(
        f"pass_id={pass_id} "
        f"fail_id={fail_id}",
        flush=True,
    )

    try:
        sampling = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            detokenize=False,
            logprob_token_ids=[
                pass_id,
                fail_id,
            ],
        )

    except TypeError as e:

        raise RuntimeError(
            "This vLLM is too old "
            "for logprob_token_ids. "
            "Upgrade vLLM first."
        ) from e

    predictions = {
        name: []
        for name, _
        in model_specs
    }

    start = (
        time.perf_counter()
    )

    def consume(
        row,
        model_name,
        output,
    ):

        lp_pass = (
            requested_logprob(
                output,
                pass_id,
            )
        )

        lp_fail = (
            requested_logprob(
                output,
                fail_id,
            )
        )

        margin = (
            lp_pass
            - lp_fail
        )

        gold = int(
            str(
                row["verdict"]
            ).strip().lower()
            == "pass"
        )

        pred = int(
            margin > 0.0
        )

        predictions[
            model_name
        ].append(
            {
                "example_id":
                    str(
                        row[
                            "example_id"
                        ]
                    ),

                "group_id":
                    str(
                        row.get(
                            "group_id"
                        ) or ""
                    ),

                "task":
                    str(
                        row["task"]
                    ),

                "gold":
                    gold,

                "gold_verdict":
                    (
                        "pass"
                        if gold
                        else "fail"
                    ),

                "pred":
                    pred,

                "pred_verdict":
                    (
                        "pass"
                        if pred
                        else "fail"
                    ),

                "margin":
                    margin,

                "p_pass":
                    sigmoid(
                        margin
                    ),

                "logp_pass":
                    lp_pass,

                "logp_fail":
                    lp_fail,

                "bce":
                    bce(
                        margin,
                        gold,
                    ),
            }
        )

    total = (
        len(rows)
        * len(model_specs)
    )

    done = 0

    # IMPORTANT:
    #
    # packet-major:
    # decode the 1800 images once.
    #
    # model-major inside packet:
    # only one giant LoRA prefix
    # occupies KV cache at a time.

    for gi, group in enumerate(
        groups,
        1,
    ):

        frame_paths, blocks = (
            packet_frames(
                group[0]
            )
        )

        for row in group[1:]:

            p2, b2 = (
                packet_frames(row)
            )

            if (
                p2 != frame_paths
                or b2 != blocks
            ):
                raise RuntimeError(
                    "media mismatch "
                    "inside group: "
                    f"{row['example_id']}"
                )

        effective_max_pixels = (
            adaptive_image_max_pixels(
                image_count=
                    len(frame_paths),

                configured_max_pixels=
                    args.max_pixels,

                min_pixels=
                    args.min_pixels,

                max_input_tokens=
                    args.max_input_tokens,

                target_fraction=
                    args.image_context_target_fraction,

                text_token_reserve=
                    args.image_text_token_reserve,

                item_token_overhead=
                    args.image_item_token_overhead,

                vision_token_pixel_area=
                    geometry[
                        "merged_token_pixel_area"
                    ],
            )
        )

        rep_messages = (
            build_messages(
                group[0],
                frame_paths,
                blocks,
                args.min_pixels,
                effective_max_pixels,
            )
        )

        image_inputs, video_inputs = (
            process_vision_info(
                rep_messages,
                image_patch_size=
                    geometry[
                        "patch_size"
                    ],
            )
        )

        if (
            video_inputs is not None
            and len(video_inputs) > 0
        ):
            raise RuntimeError(
                "unexpected video inputs"
            )

        image_inputs = list(
            image_inputs or []
        )

        if (
            len(image_inputs)
            != len(frame_paths)
        ):
            raise RuntimeError(
                "image count mismatch: "
                f"{len(image_inputs)} "
                "!= "
                f"{len(frame_paths)}"
            )

        uuids = [
            media_uuid(
                p,
                args.min_pixels,
                effective_max_pixels,
            )
            for p in frame_paths
        ]

        prompts = {}

        for row in group:

            messages = (
                build_messages(
                    row,
                    frame_paths,
                    blocks,
                    args.min_pixels,
                    effective_max_pixels,
                )
            )

            prompts[
                str(
                    row[
                        "example_id"
                    ]
                )
            ] = render_prompt(
                processor,
                messages,
            )

        def inp(row):

            return {
                "prompt":
                    prompts[
                        str(
                            row[
                                "example_id"
                            ]
                        )
                    ],

                "multi_modal_data": {
                    "image":
                        image_inputs
                },

                "multi_modal_uuids": {
                    "image":
                        uuids
                },
            }

        for (
            model_name,
            lora,
        ) in model_specs:

            # Seed one request first.
            #
            # This materializes the huge
            # image/frame prefix for this
            # exact model/LoRA.

            seed = group[0]

            out = llm.generate(
                [inp(seed)],
                sampling_params=
                    sampling,
                lora_request=
                    lora,
                use_tqdm=False,
            )

            consume(
                seed,
                model_name,
                out[0],
            )

            done += 1

            # Remaining QA in same packet
            # now share the cached prefix.

            rest = group[1:]

            if rest:

                lora_arg = (
                    None
                    if lora is None
                    else [
                        lora
                    ] * len(rest)
                )

                outs = llm.generate(
                    [
                        inp(r)
                        for r in rest
                    ],
                    sampling_params=
                        sampling,
                    lora_request=
                        lora_arg,
                    use_tqdm=False,
                )

                for row, output in zip(
                    rest,
                    outs,
                    strict=True,
                ):
                    consume(
                        row,
                        model_name,
                        output,
                    )

                done += len(rest)

        close_images(
            image_inputs
        )

        print(
            f"group={gi}/{len(groups)} "
            f"qas={len(group)} "
            f"frames={len(frame_paths)} "
            f"max_pixels="
            f"{effective_max_pixels} "
            f"done={done}/{total} "
            f"elapsed_s="
            f"{time.perf_counter()-start:.1f}",
            flush=True,
        )

    wall = (
        time.perf_counter()
        - start
    )

    summaries = {}

    for (
        name,
        pred_rows,
    ) in predictions.items():

        pred_rows.sort(
            key=lambda r:
            r["example_id"]
        )

        write_jsonl(
            args.output_dir
            / (
                f"predictions_"
                f"{name}.jsonl"
            ),
            pred_rows,
        )

        summaries[name] = (
            summarize(
                pred_rows
            )
        )

    summary = {
        "schema_version":
            "egolife_vllm_"
            "judge_eval_v2",

        "eval_manifest":
            str(
                args.eval_manifest.resolve()
            ),

        "task":
            args.task,

        "examples":
            len(rows),

        "media_groups":
            len(groups),

        "models":
            summaries,

        "pass_token_id":
            pass_id,

        "fail_token_id":
            fail_id,

        "wall_seconds":
            wall,

        "engine": {
            "gpu_memory_utilization":
                args.gpu_memory_utilization,

            "max_num_seqs":
                args.max_num_seqs,

            "max_num_batched_tokens":
                args.max_num_batched_tokens,

            "mm_processor_cache_gb":
                args.mm_processor_cache_gb,

            "attention_backend":
                args.attention_backend,

            "chunked_prefill":
                True,

            "prefix_caching":
                True,
        },
    }

    (
        args.output_dir
        / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print(
        "\n=== SUMMARY ==="
    )

    for (
        name,
        m,
    ) in summaries.items():

        print(
            f"{name:>20} "
            f"n={m['n']:3d} "
            f"loss="
            f"{m['raw_bce']:.6f} "
            f"acc="
            f"{m['accuracy']:.4f} "
            f"bal_acc="
            f"{m['balanced_accuracy']:.4f} "
            f"pred_pass="
            f"{m['pred_pass_rate']:.4f} "
            f"margin="
            f"{m['mean_margin']:.4f}"
        )

    print(
        f"wall_seconds={wall:.1f}"
    )

    print(
        "output="
        f"{args.output_dir.resolve()}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
