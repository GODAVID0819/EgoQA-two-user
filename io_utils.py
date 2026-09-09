"""Small IO helpers for the EgoLife two-user QA pipeline."""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
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


def _hugging_face_token() -> str | None:
    """Resolve an HF token without assuming that ``HOME`` is the login home.

    HPC launchers redirect ``HOME`` and ``HF_HOME`` to job-local scratch.  Keep
    accepting the standard environment token, but also honor ``HF_TOKEN_PATH``
    so the launcher can point at the token created by ``huggingface-cli login``
    before it redirects those cache directories.
    """

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token.strip() or None

    token_path = os.getenv("HF_TOKEN_PATH")
    if not token_path:
        return None
    try:
        return Path(token_path).read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def _retry_delay_seconds(error: Exception, attempt: int) -> float:
    """Return bounded exponential backoff, honoring HTTP Retry-After."""

    exponential_delay = min(float(2**attempt), 300.0)
    if not isinstance(error, urllib.error.HTTPError):
        return exponential_delay

    retry_after = error.headers.get("Retry-After")
    if not retry_after:
        return exponential_delay
    try:
        server_delay = float(retry_after)
    except ValueError:
        try:
            retry_time = parsedate_to_datetime(retry_after)
            if retry_time.tzinfo is None:
                retry_time = retry_time.replace(tzinfo=timezone.utc)
            server_delay = (retry_time - datetime.now(timezone.utc)).total_seconds()
        except (TypeError, ValueError, OverflowError):
            return exponential_delay
    return min(max(exponential_delay, server_delay, 0.0), 300.0)


def _request(url: str) -> urllib.request.Request:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "egolife-two-user-qa/0.1"},
    )
    token = _hugging_face_token()
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    return request


def _report_retry(
    *,
    operation: str,
    url: str,
    error: Exception,
    attempt: int,
    retries: int,
    delay: float,
) -> None:
    status = error.code if isinstance(error, urllib.error.HTTPError) else type(error).__name__
    print(
        f"{operation}_retry attempt={attempt + 1}/{retries + 1} "
        f"status={status} wait_seconds={delay:g} url={url}",
        file=sys.stderr,
        flush=True,
    )


def fetch_json(url: str, *, timeout: int = 60, retries: int = 3) -> Any:
    req = _request(url)
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
            delay = _retry_delay_seconds(last_error, attempt)
            _report_retry(
                operation="fetch_json",
                url=url,
                error=last_error,
                attempt=attempt,
                retries=retries,
                delay=delay,
            )
            time.sleep(delay)
    raise RuntimeError(f"GET {url} failed after {retries + 1} attempts: {last_error}") from last_error


def download_file(url: str, output_path: str | Path, *, timeout: int = 120, retries: int = 8) -> Path:
    """Download a URL atomically, reusing an existing non-empty file."""

    output_path = Path(output_path)
    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    req = _request(url)

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
                delay = _retry_delay_seconds(exc, attempt)
                _report_retry(
                    operation="download",
                    url=url,
                    error=exc,
                    attempt=attempt,
                    retries=retries,
                    delay=delay,
                )
                time.sleep(delay)
                fd, tmp_name = tempfile.mkstemp(prefix=output_path.name, suffix=".tmp", dir=output_path.parent)
                os.close(fd)
                tmp_path = Path(tmp_name)
    raise RuntimeError(f"download failed after {retries + 1} attempts: {url}") from last_error


def stable_id(*parts: Any) -> str:
    return "_".join(str(part).strip().replace("/", "-").replace(" ", "_") for part in parts if str(part).strip())
