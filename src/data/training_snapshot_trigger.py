"""Helpers for automatic training snapshot compilation after data lands in S3/DB."""

import logging
import threading
from typing import Literal

SnapshotMode = Literal["initial", "incremental"]

logger = logging.getLogger(__name__)

_compile_lock = threading.Lock()
_queue_lock = threading.Lock()
_pending_modes: set[SnapshotMode] = set()
_pending_reasons: dict[SnapshotMode, str] = {}
_worker_thread: threading.Thread | None = None


def _compile_snapshot(mode: SnapshotMode, reason: str) -> str | None:
    from src.data.compile_training_data import compile_incremental, compile_initial

    compile_fn = compile_initial if mode == "initial" else compile_incremental

    logger.info("Starting %s training snapshot compile (%s)", mode, reason)
    with _compile_lock:
        version = compile_fn()

    if version:
        logger.info(
            "%s training snapshot compile finished with version %s (%s)",
            mode,
            version,
            reason,
        )
    else:
        logger.warning(
            "%s training snapshot compile produced no snapshot (%s)",
            mode,
            reason,
        )
    return version


def run_training_snapshot(mode: SnapshotMode, reason: str) -> str | None:
    """Run a training snapshot compile immediately and return the created version."""

    return _compile_snapshot(mode, reason)


def _pop_next_request() -> tuple[SnapshotMode, str] | None:
    with _queue_lock:
        if not _pending_modes:
            return None

        mode: SnapshotMode = "initial" if "initial" in _pending_modes else "incremental"
        _pending_modes.remove(mode)
        reason = _pending_reasons.pop(mode, f"{mode} snapshot requested")
        return mode, reason


def _drain_snapshot_queue() -> None:
    global _worker_thread

    while True:
        next_request = _pop_next_request()
        if next_request is None:
            with _queue_lock:
                _worker_thread = None
            return

        mode, reason = next_request
        try:
            _compile_snapshot(mode, reason)
        except Exception:
            logger.exception(
                "Automatic %s training snapshot compile failed (%s)",
                mode,
                reason,
            )


def queue_training_snapshot(mode: SnapshotMode, reason: str) -> dict[str, str]:
    """Queue a background training snapshot compile.

    Multiple requests for the same mode coalesce into one follow-up run.
    """

    global _worker_thread

    with _queue_lock:
        _pending_modes.add(mode)
        _pending_reasons[mode] = reason

        if _worker_thread is not None and _worker_thread.is_alive():
            status = "queued"
        else:
            _worker_thread = threading.Thread(
                target=_drain_snapshot_queue,
                name="training-snapshot-trigger",
                daemon=True,
            )
            _worker_thread.start()
            status = "started"

    logger.info("Automatic %s training snapshot %s (%s)", mode, status, reason)
    return {"status": status, "mode": mode, "reason": reason}
