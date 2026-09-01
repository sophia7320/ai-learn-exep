"""Sampling-based attribute logger for long-running training loops.

Unlike :mod:`src.util.deepvarlog.core`, this backend does not use
``sys.settrace``. A background thread reads a small set of instance attributes
(``self.epoch``, ``self.loss``, ...) every ``interval`` seconds and writes a
JSONL entry only when the summarized snapshot changes.

The target is expected to be an instance method that stores its metrics on the
instance while it runs, for example::

    @monitor("epoch", "loss", "accuracy", interval=0.2)
    def fit(self, train_data):
        for epoch in range(epochs):
            ...
            self.epoch = epoch
            self.loss = loss.item()
            self.accuracy = accuracy
"""

from __future__ import annotations

import sys
import threading
import uuid
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

from .core import (
    _default_path,
    _fingerprint,
    _settings,
    _summarize,
    _type_name,
    _write,
)

F = TypeVar("F", bound=Callable[..., Any])
_MISSING = object()


def monitor(
    *fields: str,
    interval: float = 0.5,
    path: Optional[str | Path] = None,
    console: Optional[bool] = None,
    compact: Optional[bool] = None,
) -> Callable[[F], F]:
    """Record instance attributes from a background thread while a method runs.

    ``fields`` are attribute names read from ``self`` (for example
    ``monitor("epoch", "loss", "accuracy")``). A sample is taken every
    ``interval`` seconds, and an entry is written only when at least one
    summarized value changes. One final ``return`` entry (or ``exception``
    entry when the method raises) is always written synchronously after the
    method finishes, so the final state is never lost.

    ``path``/``console``/``compact`` follow the same rules as
    :func:`src.util.deepvarlog.core.record`: ``None`` falls back to
    ``configure`` settings or ``<cwd>/log/<source-stem>.jsonl``.
    """
    if not fields or any(not isinstance(name, str) or not name for name in fields):
        raise ValueError("monitor() requires at least one non-empty field name")
    if len(set(fields)) != len(fields):
        raise ValueError("field names must be unique")
    if interval <= 0:
        raise ValueError("interval must be > 0")

    def decorate(function: F) -> F:
        @wraps(function)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            if path is not None:
                output_path = Path(path)
            elif _settings.path is not None:
                output_path = _settings.path
            else:
                output_path = _default_path(function)

            show_console = console if console is not None else _settings.console
            compact_output = compact if compact is not None else _settings.compact

            stop_event = threading.Event()
            run_id = uuid.uuid4().hex
            previous_fingerprint: Optional[str] = None
            thread_error: Optional[BaseException] = None

            def emit(event: str, error: Optional[BaseException] = None) -> None:
                nonlocal previous_fingerprint

                raw: dict[str, Any] = {}
                for name in fields:
                    value = getattr(self, name, _MISSING)
                    if value is not _MISSING:
                        raw[name] = value
                if not raw:
                    return

                available = {
                    name: _summarize(value, _settings.max_repr)
                    for name, value in raw.items()
                }
                fingerprint = _fingerprint(available)

                # Samples are deduplicated by fingerprint. The final
                # return/exception entry is forced so the end state is always
                # logged even when it has not changed since the last sample.
                if event == "sample" and fingerprint == previous_fingerprint:
                    return
                if event == "sample":
                    previous_fingerprint = fingerprint

                payload: dict[str, Any] = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "run_id": run_id,
                    "function": f"{function.__module__}.{function.__qualname__}",
                    "event": event,
                    "variables": available,
                }
                if compact_output:
                    payload["message"] = ", ".join(
                        f"{name}={available[name]}"
                        for name in fields
                        if name in available
                    )
                if error is not None:
                    payload["error"] = {
                        "type": _type_name(error),
                        "message": str(error),
                    }
                _write(payload, output_path, show_console, compact_output)

            def worker() -> None:
                nonlocal thread_error
                try:
                    while not stop_event.wait(interval):
                        emit("sample")
                except BaseException as exc:  # the observer must never break training
                    thread_error = exc

            thread = threading.Thread(
                target=worker,
                name=f"deepvarlog-{function.__qualname__}",
                daemon=True,
            )
            thread.start()

            error: Optional[BaseException] = None
            try:
                return function(self, *args, **kwargs)
            except BaseException as exc:
                error = exc
                raise
            finally:
                stop_event.set()
                thread.join(timeout=max(1.0, min(5.0, interval * 2)))
                if thread_error is not None:
                    print(
                        f"deepvarlog monitor thread error: {thread_error}",
                        file=sys.stderr,
                    )
                # Synchronous final entry: even if the function returned before
                # the first sample, the end state is guaranteed to be logged.
                emit("exception" if error is not None else "return", error)

        return wrapper  # type: ignore[return-value]

    return decorate
