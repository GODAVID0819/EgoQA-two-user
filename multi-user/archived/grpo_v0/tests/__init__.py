"""Historical GRPO v0 tests."""

from __future__ import annotations

import sys
import types
from pathlib import Path


ARCHIVE_ROOT = Path(__file__).resolve().parents[1]
REPO_SOURCE_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ARCHIVE_ROOT))

if "egolife_two_user_qa" not in sys.modules:
    package = types.ModuleType("egolife_two_user_qa")
    package.__path__ = [str(REPO_SOURCE_ROOT)]
    package.__package__ = "egolife_two_user_qa"
    sys.modules["egolife_two_user_qa"] = package
