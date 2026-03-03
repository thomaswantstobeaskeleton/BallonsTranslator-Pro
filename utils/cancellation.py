"""
Explicit cancel flag abstraction (Section 7: Caching + memory / stability patterns).

Thread-safe cancellation object used across long operations (pipeline, batch, etc.).
"""
import threading


class CancelFlag:
    """
    Tiny thread-safe cancellation flag. Request cancel from any thread;
    long-running code checks is_set() periodically and exits cleanly.
    """

    __slots__ = ("_cancelled", "_lock")

    def __init__(self) -> None:
        self._cancelled = False
        self._lock = threading.Lock()

    def request(self) -> None:
        """Request cancellation (idempotent)."""
        with self._lock:
            self._cancelled = True

    def is_set(self) -> bool:
        """Return True if cancellation was requested."""
        with self._lock:
            return self._cancelled

    def reset(self) -> None:
        """Clear the flag for reuse (e.g. before starting a new run)."""
        with self._lock:
            self._cancelled = False
