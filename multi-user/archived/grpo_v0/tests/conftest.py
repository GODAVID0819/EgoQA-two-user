from __future__ import annotations

import sys
from pathlib import Path


ARCHIVE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ARCHIVE_ROOT))
