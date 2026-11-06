"""使用正式处理器核对输入形状与上下文长度。此入口只在CPU上运行。"""
import argparse
import json
from pathlib import Path

from .collator import JudgeFrameCollator
from .contracts import JudgeTask
from .loss import BinaryClassWeights, resolve_verdict_token_ids
from .train_aria12 import load_splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    train, test = load_splits(args.input / "train.jsonl", args.input / "test.jsonl")
    from transformers import AutoConfig, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True, trust_remote_code=True)
    config = AutoConfig.from_pretrained(args.model, local_files_only=True, trust_remote_code=True)
    limit = min(262144, int(config.text_config.max_position_embeddings))
    token_ids = resolve_verdict_token_ids(processor.tokenizer)
    collator = JudgeFrameCollator(
        processor=processor, class_weights={JudgeTask.ANSWERABILITY: BinaryClassWeights(1, 1, 120, 40)},
        task_scales={JudgeTask.ANSWERABILITY: 1}, min_pixels=3136, max_pixels=119808,
        max_input_tokens=262144, decoded_image_cache_entries=0,
    )
    longest = {}
    for example in train + test:
        speaker = example.frame_sets[0].label
        length = len(processor.tokenizer.encode(example.prompt, add_special_tokens=False))
        if speaker not in longest or length > longest[speaker][0]:
            longest[speaker] = (length, example)
    results = []
    for speaker, (_, example) in sorted(longest.items()):
        batch = collator([example])
        images = batch["image_grid_thw"]
        tokens = batch["input_ids"].shape[-1]
        if len(images) != 1404 or tokens > limit:
            raise ValueError(f"{speaker}: images={len(images)}, tokens={tokens}, limit={limit}")
        results.append({"speaker": speaker, "example_id": example.example_id,
                        "images": len(images), "tokens": tokens,
                        "image_grids": sorted({tuple(x) for x in images.tolist()})})
        del batch
    summary = {"context_limit": limit, "verdict_token_ids": {k.value: v for k, v in token_ids.items()},
               "train_examples": len(train), "test_examples": len(test), "processed": results}
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
