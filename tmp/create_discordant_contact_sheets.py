from __future__ import annotations

import argparse
import json
import re
import textwrap
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps


def short_id(evidence_id: str) -> str:
    return evidence_id.replace("EGOLIFE2U_RANDOM_PAIR_CLIP_PRUNED_", "")


def frame_number(path: Path) -> int:
    match = re.search(r"frame_(\d+)_", path.name)
    return int(match.group(1)) if match else 9999


def fit_frame(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as source:
        image = source.convert("RGB")
    return ImageOps.fit(image, size, method=Image.Resampling.LANCZOS)


def write_sheet(
    item: dict[str, Any], media_root: Path, output_path: Path
) -> None:
    evidence_id = item["evidence_id"]
    packet_id = short_id(evidence_id)
    day_time = "_".join(packet_id.split("_")[:2])
    frame_root = media_root / day_time / "sampled_frames"
    user_dirs = sorted(path for path in frame_root.iterdir() if path.is_dir())
    if len(user_dirs) != 2:
        raise ValueError(f"expected two sampled-frame directories in {frame_root}")

    without = item["without_rationale"]
    with_rationale = item["with_rationale"]
    accepted_items = []
    if without["status"] == "accepted":
        accepted_items.append(("without rationale", without["accepted_attempt"]))
    if with_rationale["status"] == "accepted":
        accepted_items.append(("with rationale", with_rationale["accepted_attempt"]))
    if not accepted_items:
        raise ValueError(f"{packet_id} has no accepted item")

    thumb = (210, 118)
    columns = 10
    gap = 6
    caption_height = 19
    row_height = thumb[1] + caption_height + gap
    header_height = 35 + 125 * len(accepted_items)
    user_header = 24
    rows_per_user = 3
    width = columns * thumb[0] + (columns - 1) * gap + 20
    height = header_height + len(user_dirs) * (user_header + rows_per_user * row_height) + 15
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    y = 10
    draw.text((10, y), packet_id, fill="black", font=font)
    y += 22
    for arm, accepted in accepted_items:
        qa = accepted["qa"]
        draw.text((10, y), f"{arm} | attempt {accepted['attempt']}", fill="darkgreen", font=font)
        y += 17
        question = str(qa.get("question") or "")
        answer = str(qa.get("answer") or "")
        for line in textwrap.wrap("Q: " + question, width=150)[:4]:
            draw.text((10, y), line, fill="black", font=font)
            y += 17
        for line in textwrap.wrap("A: " + answer, width=150)[:2]:
            draw.text((10, y), line, fill="navy", font=font)
            y += 17
        y += 8
    y = header_height

    for user_dir in user_dirs:
        draw.rectangle((0, y, width, y + user_header), fill=(225, 232, 240))
        draw.text((10, y + 5), user_dir.name, fill="black", font=font)
        y += user_header
        frames = sorted(user_dir.glob("frame_*.png"), key=frame_number)[:30]
        for index, frame in enumerate(frames):
            row, column = divmod(index, columns)
            x = 10 + column * (thumb[0] + gap)
            frame_y = y + row * row_height
            sheet.paste(fit_frame(frame, thumb), (x, frame_y))
            draw.text((x + 2, frame_y + thumb[1] + 2), f"{frame_number(frame):02d}s", fill="black", font=font)
        y += rows_per_user * row_height

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--details", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--subset",
        choices=("discordant", "concordant-accepted", "all-accepted"),
        default="discordant",
    )
    args = parser.parse_args()
    details = json.loads(args.details.read_text(encoding="utf-8"))
    manifest = []
    if args.subset == "discordant":
        items = details["discordant"]
    elif args.subset == "concordant-accepted":
        items = [
            item
            for item in details["comparisons"]
            if item["without_rationale"]["status"] == "accepted"
            and item["with_rationale"]["status"] == "accepted"
        ]
    else:
        items = [
            item
            for item in details["comparisons"]
            if item["without_rationale"]["status"] == "accepted"
            or item["with_rationale"]["status"] == "accepted"
        ]
    for item in items:
        packet_id = short_id(item["evidence_id"])
        output_path = args.output_dir / f"{packet_id}.jpg"
        write_sheet(item, args.media_root, output_path)
        manifest.append(str(output_path.resolve()))
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
