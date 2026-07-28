from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .domain import Anchor, AnchorSet


def load_anchor_set(path: str | Path | None = None) -> AnchorSet:
    anchor_path = Path(path) if path is not None else Path(__file__).with_name("anchors.json")
    raw = anchor_path.read_bytes()
    data = json.loads(raw.decode("utf-8"))
    return AnchorSet(
        strong=Anchor(str(data["strong"]["anchor_id"]), dict(data["strong"])),
        weak=Anchor(str(data["weak"]["anchor_id"]), dict(data["weak"])),
        sha256=hashlib.sha256(raw).hexdigest(),
    )
