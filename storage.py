# storage.py
"""
Python Quest — storage layer (save/load)

Responsibilities:
- Persist player state to a JSON file (save.json)
- Load player state safely (handle missing/corrupt files)
- Keep storage logic isolated from UI and AI
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional


DEFAULT_SAVE_PATH = "save.json"


class StorageError(Exception):
    pass


def _safe_read_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
        return None
    except Exception:
        return None


def _safe_write_json(path: str, data: Dict[str, Any]) -> None:
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)  # atomic on most OS
    except Exception as e:
        # cleanup temp file if something went wrong
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        raise StorageError(f"Failed to save file: {e}") from e


def load_state_dict(path: str = DEFAULT_SAVE_PATH) -> Optional[Dict[str, Any]]:
    """
    Load state as a dict (JSON). Returns None if file doesn't exist or is unreadable.
    """
    return _safe_read_json(path)


def save_state_dict(state: Dict[str, Any], path: str = DEFAULT_SAVE_PATH) -> None:
    """
    Save state dict to JSON. Adds metadata for debugging.
    """
    payload = dict(state)
    payload["_meta"] = {
        "saved_at": time.time(),
        "app": "Python Quest",
        "version": 1,
    }
    _safe_write_json(path, payload)


def reset_save(path: str = DEFAULT_SAVE_PATH) -> None:
    """
    Delete save file (if exists).
    """
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception as e:
        raise StorageError(f"Failed to reset save: {e}") from e
