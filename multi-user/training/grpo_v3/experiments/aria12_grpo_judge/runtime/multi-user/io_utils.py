"""Small IO helpers for the EgoLife two-user QA pipeline."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Iterable


_DOWNLOAD_RATE_LIMIT_LOCK = threading.Lock()
_DOWNLOAD_NOT_BEFORE_MONOTONIC = 0.0
DEFAULT_DOWNLOAD_RETRIES = 8
DEFAULT_HTTP_429_BASE_DELAY_SECONDS = 15.0
DEFAULT_HTTP_MAX_RETRY_DELAY_SECONDS = 300.0


def _huggingface_token() -> str | None:
    token = os.getenv("HF_TOKEN", "").strip()
    if token:
        return token
    token_path_value = os.getenv("HF_TOKEN_PATH", "").strip()
    if not token_path_value:
        return None
    token_path = Path(token_path_value)
    try:
        token = token_path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return token or None


def _request(url: str) -> urllib.request.Request:
    request = urllib.request.Request(
        url, headers={"User-Agent": "egolife-two-user-qa/0.1"}
    )
    token = _huggingface_token()
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    return request


def _positive_environment_float(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


def _wait_for_download_slot() -> None:
    while True:
        with _DOWNLOAD_RATE_LIMIT_LOCK:
            remaining = _DOWNLOAD_NOT_BEFORE_MONOTONIC - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 1.0))


def _extend_download_cooldown(seconds: float) -> None:
    global _DOWNLOAD_NOT_BEFORE_MONOTONIC
    with _DOWNLOAD_RATE_LIMIT_LOCK:
        _DOWNLOAD_NOT_BEFORE_MONOTONIC = max(
            _DOWNLOAD_NOT_BEFORE_MONOTONIC,
            time.monotonic() + max(0.0, float(seconds)),
        )


def _retry_after_seconds(error: urllib.error.HTTPError) -> float | None:
    value = error.headers.get("Retry-After") if error.headers else None
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        pass
    try:
        retry_at = parsedate_to_datetime(str(value))
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
    except (TypeError, ValueError, OverflowError):
        return None


def _rate_limit_delay_seconds(
    error: urllib.error.HTTPError, attempt: int
) -> float:
    base = _positive_environment_float(
        "EGOLIFE_HTTP_429_BASE_DELAY_SECONDS",
        DEFAULT_HTTP_429_BASE_DELAY_SECONDS,
    )
    maximum = _positive_environment_float(
        "EGOLIFE_HTTP_MAX_RETRY_DELAY_SECONDS",
        DEFAULT_HTTP_MAX_RETRY_DELAY_SECONDS,
    )
    retry_after = _retry_after_seconds(error) or 0.0
    exponential = min(maximum, base * (2**attempt))
    # Stagger worker wakeups slightly after the shared cooldown expires.
    jitter = 0.1 * (1 + threading.get_ident() % 11)
    return min(maximum, max(retry_after, exponential) + jitter)


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
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        _wait_for_download_slot()
        try:
            with urllib.request.urlopen(_request(url), timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if 400 <= exc.code < 500 and exc.code != 429:
                raise RuntimeError(f"GET {url} failed with HTTP {exc.code}: {body[:300]}") from exc
            last_error = exc
            if exc.code == 429 and attempt < retries:
                delay = _rate_limit_delay_seconds(exc, attempt)
                _extend_download_cooldown(delay)
                print(
                    "http_request status=rate_limited "
                    f"attempt={attempt + 1}/{retries + 1} wait_seconds={delay:.1f}",
                    flush=True,
                )
        except Exception as exc:
            last_error = exc
        if attempt < retries:
            if not isinstance(last_error, urllib.error.HTTPError) or last_error.code != 429:
                time.sleep(2**attempt)
    raise RuntimeError(f"GET {url} failed after {retries + 1} attempts: {last_error}") from last_error


def download_file(
    url: str,
    output_path: str | Path,
    *,
    timeout: int = 120,
    retries: int | None = None,
) -> Path:
    """Download atomically with authenticated, coordinated 429 recovery."""

    output_path = Path(output_path)
    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path

    if retries is None:
        try:
            retries = int(os.getenv("EGOLIFE_DOWNLOAD_RETRIES", DEFAULT_DOWNLOAD_RETRIES))
        except ValueError:
            retries = DEFAULT_DOWNLOAD_RETRIES
    if retries < 0:
        raise ValueError("download retries must be non-negative")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        _wait_for_download_slot()
        fd, tmp_name = tempfile.mkstemp(
            prefix=output_path.name, suffix=".tmp", dir=output_path.parent
        )
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            with urllib.request.urlopen(
                _request(url), timeout=timeout
            ) as resp, tmp_path.open("wb") as f:
                shutil.copyfileobj(resp, f)
            tmp_path.replace(output_path)
            return output_path
        except urllib.error.HTTPError as exc:
            last_error = exc
            tmp_path.unlink(missing_ok=True)
            if 400 <= exc.code < 500 and exc.code != 429:
                body = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"download {url} failed with HTTP {exc.code}: {body[:300]}"
                ) from exc
            if exc.code == 429 and attempt < retries:
                delay = _rate_limit_delay_seconds(exc, attempt)
                _extend_download_cooldown(delay)
                print(
                    "media_download status=rate_limited "
                    f"attempt={attempt + 1}/{retries + 1} "
                    f"wait_seconds={delay:.1f} destination={output_path}",
                    flush=True,
                )
        except Exception as exc:
            last_error = exc
            tmp_path.unlink(missing_ok=True)
        if attempt < retries and (
            not isinstance(last_error, urllib.error.HTTPError)
            or last_error.code != 429
        ):
            time.sleep(min(60.0, float(2**attempt)))
    raise RuntimeError(f"download failed after {retries + 1} attempts: {url}") from last_error


def stable_id(*parts: Any) -> str:
    return "_".join(str(part).strip().replace("/", "-").replace(" ", "_") for part in parts if str(part).strip())
