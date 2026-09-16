"""Serve the six-user RLHF source-video labeler."""

from __future__ import annotations

import http.server
from pathlib import Path
import sys
import threading
import webbrowser

PACKAGE_ROOT = Path(__file__).resolve().parent


def main() -> int:
    html_path = PACKAGE_ROOT / "rlhf_labeling.html"
    if not html_path.is_file():
        print(f"Missing labeler: {html_path}")
        return 1
    handler = lambda *args, **kwargs: http.server.SimpleHTTPRequestHandler(
        *args, directory=str(PACKAGE_ROOT), **kwargs
    )
    with http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
        url = f"http://127.0.0.1:{server.server_address[1]}/rlhf_labeling.html"
        print(f"Serving labeler: {url}")
        print("Video segments load directly from Hugging Face when selected.")
        print("Keep this window open while labeling. Press Ctrl+C to stop.")
        if "--no-browser" not in sys.argv:
            threading.Timer(0.35, lambda: webbrowser.open(url)).start()
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nLabeling server stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
