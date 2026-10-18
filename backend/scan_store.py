"""In-memory scan records. Uploaded media is not written to a durable store."""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional

_lock = threading.Lock()
_scans: Dict[str, Dict[str, Any]] = {}
_cancelled: set[str] = set()
_MAX = 24


def save_scan(scan: Dict[str, Any]) -> None:
    with _lock:
        _scans[scan["scan_id"]] = scan
        while len(_scans) > _MAX:
            oldest = next(iter(_scans))
            _scans.pop(oldest, None)


def get_scan(scan_id: str) -> Optional[Dict[str, Any]]:
    with _lock:
        record = _scans.get(scan_id)
        return record


def cancel_scan(scan_id: str) -> None:
    with _lock:
        _cancelled.add(scan_id)


def is_cancelled(scan_id: str) -> bool:
    with _lock:
        return scan_id in _cancelled


def clear_cancel(scan_id: str) -> None:
    with _lock:
        _cancelled.discard(scan_id)
