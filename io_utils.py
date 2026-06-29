"""Small IO helpers for the EgoLife two-user QA pipeline."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: Any, *, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
        f.write("\n")


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL row: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            yield value


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def hf_resolve_url(dataset: str, repo_path: str, revision: str = "main") -> str:
    clean = repo_path.lstrip("/")
    return f"https://huggingface.co/datasets/{dataset}/resolve/{revision}/{clean}"


def fetch_json(url: str, *, timeout: int = 60, retries: int = 3) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "egolife-two-user-qa/0.1"})
    token = os.getenv("HF_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if 400 <= exc.code < 500 and exc.code != 429:
                raise RuntimeError(f"GET {url} failed with HTTP {exc.code}: {body[:300]}") from exc
            last_error = exc
        except Exception as exc:
            last_error = exc
        if attempt < retries:
            time.sleep(2 ** attempt)
    raise RuntimeError(f"GET {url} failed after {retries + 1} attempts: {last_error}") from last_error


def download_file(url: str, output_path: str | Path, *, timeout: int = 120, retries: int = 3) -> Path:
    """Download a URL atomically, reusing an existing non-empty file."""

    output_path = Path(output_path)
    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "egolife-two-user-qa/0.1"})
    token = os.getenv("HF_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")

    fd, tmp_name = tempfile.mkstemp(prefix=output_path.name, suffix=".tmp", dir=output_path.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp, tmp_path.open("wb") as f:
                shutil.copyfileobj(resp, f)
            tmp_path.replace(output_path)
            return output_path
        except Exception as exc:
            last_error = exc
            tmp_path.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(2 ** attempt)
                fd, tmp_name = tempfile.mkstemp(prefix=output_path.name, suffix=".tmp", dir=output_path.parent)
                os.close(fd)
                tmp_path = Path(tmp_name)
    raise RuntimeError(f"download failed after {retries + 1} attempts: {url}") from last_error


def stable_id(*parts: Any) -> str:
    return "_".join(str(part).strip().replace("/", "-").replace(" ", "_") for part in parts if str(part).strip())
