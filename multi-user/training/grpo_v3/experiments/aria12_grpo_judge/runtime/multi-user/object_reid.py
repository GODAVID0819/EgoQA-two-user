"""EgoEverything-style object crop ReID helpers.

This is intentionally a sidecar utility for ``object_hints``. It crops detected
objects, embeds object crops and names with CLIP, and merges detections
whose visual and text similarities both exceed thresholds.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

DEFAULT_REID_MODEL_ID = "openai/clip-vit-base-patch32"


def safe_object_name(name: Any) -> str:
    text = str(name or "object").strip()
    safe = re.sub(r"[^\w\-_\.]", "_", text)
    return safe.strip("_") or "object"


def crop_object_images(
    objects: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    padding_ratio: float = 0.15,
) -> list[dict[str, Any]]:
    """Crop object boxes from their source frames, matching EgoEverything."""

    crop_dir = Path(output_dir)
    crop_dir.mkdir(parents=True, exist_ok=True)
    valid: list[dict[str, Any]] = []
    for index, obj in enumerate(objects):
        frame_path = obj.get("frame_path")
        if not frame_path:
            continue
        cv2 = None
        frame = None
        image = None
        try:
            import cv2 as cv2_module

            cv2 = cv2_module
            frame = cv2.imread(str(frame_path))
        except Exception:
            cv2 = None
        if frame is not None:
            img_h, img_w = frame.shape[:2]
        else:
            try:
                from PIL import Image

                image = Image.open(frame_path).convert("RGB")
                img_w, img_h = image.size
            except Exception:
                continue
        try:
            x0, y0, x1, y1 = [int(value) for value in obj["bbox"]]
        except (KeyError, TypeError, ValueError):
            continue
        if x0 >= x1 or y0 >= y1:
            continue

        bbox_w = x1 - x0
        bbox_h = y1 - y0
        pad_w = int(bbox_w * padding_ratio)
        pad_h = int(bbox_h * padding_ratio)

        x0_pad = max(0, x0 - pad_w)
        y0_pad = max(0, y0 - pad_h)
        x1_pad = min(img_w, x1 + pad_w)
        y1_pad = min(img_h, y1 + pad_h)
        if x0_pad >= x1_pad or y0_pad >= y1_pad:
            continue

        row = dict(obj)
        crop_path = crop_dir / f"object_{index:05d}_{safe_object_name(row.get('object_name') or row.get('name'))}.jpg"
        if frame is not None and cv2 is not None:
            crop_img = frame[y0_pad:y1_pad, x0_pad:x1_pad]
            if crop_img.size == 0:
                continue
            cv2.imwrite(str(crop_path), crop_img)
        elif image is not None:
            image.crop((x0_pad, y0_pad, x1_pad, y1_pad)).save(crop_path, quality=95)
            image.close()
        else:
            continue
        row["crop_path"] = str(crop_path)
        row["padded_bbox"] = [x0_pad, y0_pad, x1_pad, y1_pad]
        valid.append(row)
    return valid


class EgoEverythingClipReidentifier:
    """CLIP crop/name reidentifier matching EgoEverything's merge philosophy."""

    def __init__(self, *, model_id: str = DEFAULT_REID_MODEL_ID, device: str | None = None) -> None:
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except Exception as exc:
            raise RuntimeError(
                "Object ReID requires the same torch/transformers CLIP stack used by "
                "the existing CLIP mining pipeline."
            ) from exc

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = model_id
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.clip_model = CLIPModel.from_pretrained(model_id).to(self.device).eval()

    def preprocess_for_clip(self, image_path: str | Path, target_size: int = 224):
        """Preprocess a crop the same way EgoEverything does before CLIP."""

        from PIL import Image

        try:
            image = Image.open(image_path).convert("RGB")
            w, h = image.size
            if w > h:
                new_w = target_size
                new_h = int(h * target_size / w)
            else:
                new_h = target_size
                new_w = int(w * target_size / h)
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (target_size, target_size), (128, 128, 128))
            offset_x = (target_size - new_w) // 2
            offset_y = (target_size - new_h) // 2
            canvas.paste(image, (offset_x, offset_y))
            return canvas
        except Exception:
            return Image.new("RGB", (target_size, target_size), (128, 128, 128))

    def extract_clip_features(
        self,
        objects: list[dict[str, Any]],
        *,
        batch_size: int = 32,
    ) -> tuple[Any, Any]:
        all_visual_features = []
        all_text_features = []
        object_names = [str(obj.get("object_name") or obj.get("name") or "object") for obj in objects]

        for start in range(0, len(objects), batch_size):
            batch_objects = objects[start : start + batch_size]
            batch_images = [
                self.preprocess_for_clip(obj["crop_path"])
                for obj in batch_objects
                if obj.get("crop_path")
            ]
            if not batch_images:
                continue
            inputs = self.processor(images=batch_images, return_tensors="pt").to(self.device)
            with self.torch.inference_mode():
                visual_features = self.clip_model.get_image_features(**inputs)
                visual_features = visual_features / visual_features.norm(dim=-1, keepdim=True)
            all_visual_features.append(visual_features.cpu())

        for start in range(0, len(object_names), batch_size):
            batch_names = object_names[start : start + batch_size]
            inputs = self.processor(text=batch_names, padding=True, return_tensors="pt").to(self.device)
            with self.torch.inference_mode():
                text_features = self.clip_model.get_text_features(**inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            all_text_features.append(text_features.cpu())

        if all_visual_features:
            return (
                self.torch.cat(all_visual_features, dim=0),
                self.torch.cat(all_text_features, dim=0),
            )
        projection_dim = int(getattr(self.clip_model.config, "projection_dim", 512))
        return self.torch.zeros((len(objects), projection_dim)), self.torch.zeros((len(objects), projection_dim))


def _similarity_matrix(features: Any) -> list[list[float]]:
    if isinstance(features, list):
        return [
            [float(sum(float(a) * float(b) for a, b in zip(left, right))) for right in features]
            for left in features
        ]
    matrix = features @ features.T
    try:
        matrix = matrix.cpu().tolist()
    except AttributeError:
        matrix = matrix.tolist()
    return [[float(value) for value in row] for row in matrix]


def merge_similar_objects(
    objects: list[dict[str, Any]],
    visual_features: Any,
    text_features: Any,
    *,
    visual_threshold: float = 0.8,
    text_threshold: float = 0.85,
    object_id_offset: int = 0,
) -> list[dict[str, Any]]:
    """Merge objects using EgoEverything's visual AND text similarity rule."""

    if not objects:
        return []
    visual_similarity = _similarity_matrix(visual_features)
    text_similarity = _similarity_matrix(text_features)
    visited = [False for _ in objects]
    object_id = object_id_offset
    merged = [dict(obj) for obj in objects]

    for index in range(len(merged)):
        if visited[index]:
            continue
        stack = [index]
        members = []
        while stack:
            current = stack.pop()
            if visited[current]:
                continue
            visited[current] = True
            members.append(current)
            for other in range(len(merged)):
                if visited[other]:
                    continue
                if visual_similarity[current][other] > visual_threshold and text_similarity[current][other] > text_threshold:
                    stack.append(other)

        for member in members:
            merged[member]["object_id"] = object_id
            merged[member]["max_visual_similarity"] = round(max(visual_similarity[member]), 6)
            merged[member]["max_text_similarity"] = round(max(text_similarity[member]), 6)
        object_id += 1
    return merged


def reidentify_objects(
    objects: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    reidentifier: EgoEverythingClipReidentifier,
    visual_threshold: float = 0.8,
    text_threshold: float = 0.85,
    batch_size: int = 32,
    object_id_offset: int = 0,
) -> list[dict[str, Any]]:
    objects_with_crops = crop_object_images(objects, output_dir)
    if not objects_with_crops:
        return [dict(obj) for obj in objects]
    visual_features, text_features = reidentifier.extract_clip_features(objects_with_crops, batch_size=batch_size)
    return merge_similar_objects(
        objects_with_crops,
        visual_features,
        text_features,
        visual_threshold=visual_threshold,
        text_threshold=text_threshold,
        object_id_offset=object_id_offset,
    )
