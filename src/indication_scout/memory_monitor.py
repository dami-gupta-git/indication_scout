"""Periodic process-memory logging for diagnosing RSS growth in the API process.

A daemon thread logs one "memory sample" line at a fixed cadence: faster while an analysis job is
running, slower when idle. Each line carries the process RSS and glibc allocator totals, which
separate memory the program is using from memory freed by the program but still held by the
allocator. A daemon thread (not an asyncio task) is used so samples continue while the event loop
is blocked.
"""

import ctypes
import logging
import threading
from collections.abc import Callable

from indication_scout.constants import (
    MEMORY_SAMPLE_ACTIVE_SECONDS,
    MEMORY_SAMPLE_IDLE_SECONDS,
)
from indication_scout.services.job_store import job_store

logger = logging.getLogger(__name__)

_GB = 1e9


class _MallInfo2(ctypes.Structure):
    """glibc `struct mallinfo2` (glibc >= 2.33); all fields are size_t byte counts."""

    _fields_ = [
        ("arena", ctypes.c_size_t),
        ("ordblks", ctypes.c_size_t),
        ("smblks", ctypes.c_size_t),
        ("hblks", ctypes.c_size_t),
        ("hblkhd", ctypes.c_size_t),
        ("usmblks", ctypes.c_size_t),
        ("fsmblks", ctypes.c_size_t),
        ("uordblks", ctypes.c_size_t),
        ("fordblks", ctypes.c_size_t),
        ("keepcost", ctypes.c_size_t),
    ]


def _load_mallinfo2() -> Callable[[], _MallInfo2] | None:
    """Return glibc's mallinfo2, or None when not running on glibc >= 2.33 (e.g. macOS)."""
    try:
        libc = ctypes.CDLL("libc.so.6")
        fn = libc.mallinfo2
    except (OSError, AttributeError):
        return None
    fn.restype = _MallInfo2
    fn.argtypes = []
    mallinfo2: Callable[[], _MallInfo2] = fn
    return mallinfo2


def _rss_bytes() -> int | None:
    """Return the process resident set size from /proc, or None when /proc is unavailable."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def _fmt_gb(value: int | None) -> str:
    return "na" if value is None else f"{value / _GB:.3f}"


def _sample(mallinfo2: Callable[[], _MallInfo2] | None) -> None:
    """Log one memory sample line."""
    running = [j for j in job_store.all() if j.status in ("pending", "running")]
    heap_in_use: int | None = None
    heap_free_held: int | None = None
    mmapped: int | None = None
    if mallinfo2 is not None:
        info = mallinfo2()
        heap_in_use = info.uordblks
        heap_free_held = info.fordblks
        mmapped = info.hblkhd
    logger.info(
        "memory sample rss_gb=%s heap_in_use_gb=%s heap_free_held_gb=%s mmapped_gb=%s "
        "threads=%d active_jobs=%d drugs=%s",
        _fmt_gb(_rss_bytes()),
        _fmt_gb(heap_in_use),
        _fmt_gb(heap_free_held),
        _fmt_gb(mmapped),
        threading.active_count(),
        len(running),
        ",".join(j.drug_name for j in running) or "-",
    )


def _run(stop: threading.Event) -> None:
    mallinfo2 = _load_mallinfo2()
    while not stop.is_set():
        _sample(mallinfo2)
        active = any(j.status in ("pending", "running") for j in job_store.all())
        stop.wait(
            MEMORY_SAMPLE_ACTIVE_SECONDS if active else MEMORY_SAMPLE_IDLE_SECONDS
        )


_stop = threading.Event()
_thread: threading.Thread | None = None


def start_memory_monitor() -> None:
    """Start the sampling thread once per process."""
    global _thread
    if _thread is not None:
        return
    _thread = threading.Thread(
        target=_run, args=(_stop,), name="memory-monitor", daemon=True
    )
    _thread.start()


def stop_memory_monitor() -> None:
    """Signal the sampling thread to exit."""
    _stop.set()
