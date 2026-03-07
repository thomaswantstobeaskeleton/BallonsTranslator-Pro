"""
Batch run report: track skipped pages and reasons (comic-translate style).
Used by pipeline to record detection/OCR/translation failures; UI shows table and allows opening page.
"""
from __future__ import annotations

import re
import threading
from typing import Any, Dict, List, Optional
from datetime import datetime

# Module-level report state (one report per run)
_report: Optional[Dict[str, Any]] = None
_lock = threading.Lock()


def start_batch_report(page_keys: List[str]) -> None:
    """Initialize a new batch report for the given page keys."""
    global _report
    with _lock:
        _report = {
            "page_keys": list(page_keys),
            "skipped": {},  # page_key -> {"reason": str, "error": str, "action": str}
            "started_at": datetime.now().isoformat(),
            "finished_at": None,
            "cancelled": False,
        }


def register_batch_skip(page_key: str, skip_reason: str, error: str = "") -> None:
    """Record a skipped page with reason and optional error text."""
    global _report
    with _lock:
        if _report is None:
            return
        detail = _localize_batch_skip_detail(error)
        action = _localize_batch_skip_action(skip_reason, error)
        _report["skipped"][page_key] = {
            "reason": skip_reason,
            "error": error,
            "detail": detail,
            "action": action,
        }


def _localize_batch_skip_detail(error: str) -> str:
    """Map raw error text to a short user-facing description."""
    if not error:
        return ""
    err_lower = error.lower()
    if "content" in err_lower and ("flag" in err_lower or "policy" in err_lower or "filter" in err_lower):
        return "Content flagged"
    if "credit" in err_lower or "quota" in err_lower or "limit" in err_lower:
        return "Insufficient credits / quota"
    if "timeout" in err_lower or "timed out" in err_lower:
        return "Timeout"
    if "429" in error or "rate limit" in err_lower:
        return "Rate limit"
    if "auth" in err_lower or "api key" in err_lower or "401" in error or "403" in error:
        return "Authentication / API key"
    if "network" in err_lower or "connection" in err_lower or "unreachable" in err_lower:
        return "Network error"
    if "server" in err_lower or "500" in error or "502" in error or "503" in error:
        return "Server error"
    if "json" in err_lower or "parse" in err_lower:
        return "Invalid response (JSON)"
    if "detection" in err_lower or "detect" in err_lower:
        return "Detection failed"
    if "ocr" in err_lower:
        return "OCR failed"
    if "translat" in err_lower:
        return "Translation failed"
    if "inpaint" in err_lower:
        return "Inpainting failed"
    # Keep first line or first 80 chars
    first_line = error.split("\n")[0].strip()
    return first_line[:80] + ("..." if len(first_line) > 80 else "")


def _localize_batch_skip_action(skip_reason: str, error: str) -> str:
    """Suggest a user action based on skip reason and error."""
    err_lower = (error or "").lower()
    if "content" in err_lower and ("flag" in err_lower or "policy" in err_lower):
        return "Try another translator or adjust content."
    if "429" in error or "rate limit" in err_lower:
        return "Wait and retry, or add more API keys."
    if "auth" in err_lower or "api key" in err_lower or "401" in error or "403" in error:
        return "Check API key in Config."
    if "timeout" in err_lower:
        return "Check network; retry or increase timeout."
    if "credit" in err_lower or "quota" in err_lower:
        return "Check translator quota / credits."
    if "detection" in err_lower or "detect" in err_lower:
        return "Try another detector or check image."
    if "ocr" in err_lower:
        return "Try another OCR module or check image."
    if "translat" in err_lower:
        return "Try another translator or check API settings."
    return "Retry or check settings."


def finalize_batch_report(cancelled: bool = False) -> Optional[Dict[str, Any]]:
    """Finalize the current report and return it; clear internal state."""
    global _report
    with _lock:
        if _report is None:
            return None
        _report["finished_at"] = datetime.now().isoformat()
        _report["cancelled"] = cancelled
        result = dict(_report)
        _report = None
        return result


def get_current_report() -> Optional[Dict[str, Any]]:
    """Return the current report dict if any (for UI). Does not clear."""
    with _lock:
        return dict(_report) if _report else None


def has_report_with_skips() -> bool:
    """True if there is a current or last report with at least one skip (used to enable Show report)."""
    # Current report is cleared on finalize; we need to keep "last report" in UI layer
    return False  # Caller (mainwindow) holds last report and checks skipped count
