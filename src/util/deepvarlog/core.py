"""Decorator-based local-variable logging."""

from __future__ import annotations

import functools
import inspect
import json
import math
import string
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import FrameType
from typing import Any, Callable, Dict, Optional, TextIO, Tuple, TypeVar


F = TypeVar("F", bound=Callable[..., Any])


class _Settings:
    def __init__(self) -> None:
        # None means derive a file from the decorated function's source name.
        self.path: Optional[Path] = None
        self.console = False
        self.compact = False
        self.max_repr = 240


_settings = _Settings()
_write_lock = threading.Lock()


def configure(
    path: Optional[str | Path] = None,
    *,
    console: bool = False,
    compact: bool = False,
    max_repr: int = 240,
) -> None:
    """Configure output for subsequently recorded calls.

    ``compact=True`` writes only the rendered ``message`` field to JSONL;
    ``console`` is opt-in and defaults to no terminal output.
    """
    if max_repr < 20:
        raise ValueError("max_repr must be at least 20")
    _settings.path = Path(path) if path is not None else None
    _settings.console = console
    _settings.compact = compact
    _settings.max_repr = max_repr


def _type_name(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _shape(value: Any) -> Optional[list[Any]]:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return [int(item) for item in shape]
    except (TypeError, ValueError):
        return [str(item) for item in shape]


def _summarize(value: Any, max_repr: int) -> Any:
    """Return a bounded, JSON-compatible representation of a local value."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Path):
        return str(value)

    shape = _shape(value)
    if shape is not None:
        summary: Dict[str, Any] = {"type": _type_name(value), "shape": shape}
        for attribute in ("dtype", "device"):
            item = getattr(value, attribute, None)
            if item is not None:
                summary[attribute] = str(item)
        size = 1
        for dimension in shape:
            if not isinstance(dimension, int):
                size = -1
                break
            size *= dimension
        if size == 1 and hasattr(value, "item"):
            try:
                summary["value"] = _summarize(value.item(), max_repr)
            except (RuntimeError, TypeError, ValueError):
                pass
        return summary

    if isinstance(value, (list, tuple)):
        if len(value) <= 12:
            return [_summarize(item, max_repr) for item in value]
        return {
            "type": _type_name(value),
            "length": len(value),
            "preview": [_summarize(item, max_repr) for item in value[:5]],
        }
    if isinstance(value, dict):
        if len(value) <= 12 and all(isinstance(key, str) for key in value):
            return {key: _summarize(item, max_repr) for key, item in value.items()}
        return {"type": _type_name(value), "length": len(value)}

    try:
        rendered = repr(value)
    except Exception:
        rendered = "<repr failed>"
    if len(rendered) > max_repr:
        rendered = rendered[: max_repr - 1] + "…"
    return {"type": _type_name(value), "repr": rendered}


def _fingerprint(values: Dict[str, Any]) -> str:
    return json.dumps(values, ensure_ascii=False, sort_keys=True, default=str)


def _template_fields(template: str) -> Tuple[str, ...]:
    """Extract root local names from a str.format-style template."""
    fields = []
    for _, field_name, _, _ in string.Formatter().parse(template):
        if field_name is None or not field_name:
            continue
        root = field_name.split(".", 1)[0].split("[", 1)[0]
        if not root.isidentifier():
            raise ValueError(
                f"format field {field_name!r} must start with a local variable name"
            )
        if root not in fields:
            fields.append(root)
    return tuple(fields)


class _MissingFormatValue(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _format_message(template: str, locals_: Dict[str, Any]) -> str:
    """Render a template without allowing formatting errors to stop training."""
    formatting_values = dict(locals_)
    for name, value in formatting_values.items():
        # Scalar tensors can be formatted with numeric specs such as ``:.4f``.
        if _shape(value) == [] and hasattr(value, "item"):
            try:
                formatting_values[name] = value.item()
            except (RuntimeError, TypeError, ValueError):
                pass
    try:
        return template.format_map(_MissingFormatValue(formatting_values))
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        return template


def _write(event: Dict[str, Any], path: Path, console: bool, compact: bool) -> None:
    output = {"message": event.get("message", "")} if compact else event
    line = json.dumps(output, ensure_ascii=False, separators=(",", ":"))
    with _write_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
        if console:
            print(line, file=sys.stderr)


def _default_path(function: Callable[..., Any]) -> Path:
    """Choose ``<cwd>/log/<source-stem>.jsonl`` when no path was specified."""
    source = Path(function.__code__.co_filename)
    stem = source.stem if source.stem and not source.stem.startswith("<") else "deepvarlog"
    return Path.cwd() / "log" / f"{stem}.jsonl"


def _make_tracer(
    function: Callable[..., Any],
    variable_names: Tuple[str, ...],
    every: int,
    on_change: bool,
    path: Path,
    console: bool,
    template: Optional[str],
    compact: bool,
) -> Callable[[FrameType, str, Any], Any]:
    target_frame: Optional[FrameType] = None
    previous_fingerprint: Optional[str] = None
    observation_count = 0
    run_id = uuid.uuid4().hex
    required_fields = _template_fields(template) if template is not None else ()

    def emit(frame: FrameType, event_name: str, error: Optional[BaseException] = None) -> None:
        nonlocal observation_count, previous_fingerprint
        raw_available = {
            name: frame.f_locals[name]
            for name in variable_names
            if name in frame.f_locals
        }
        if not raw_available and template is None:
            return
        # Do not emit half-rendered messages while a training loop is still
        # initializing its metrics (for example ``loss = None``).
        if template is not None and (
            any(name not in raw_available for name in required_fields)
            or any(raw_available[name] is None for name in required_fields)
        ):
            return

        available = {
            name: _summarize(value, _settings.max_repr)
            for name, value in raw_available.items()
        }

        fingerprint = _fingerprint(available)
        if event_name == "line" and on_change and fingerprint == previous_fingerprint:
            return
        if event_name == "line":
            previous_fingerprint = fingerprint
            observation_count += 1
            if (observation_count - 1) % every != 0:
                return

        payload: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
            "function": f"{function.__module__}.{function.__qualname__}",
            "event": event_name,
            "line": frame.f_lineno,
            "variables": available,
        }
        if template is not None:
            payload["message"] = _format_message(template, raw_available)
        elif compact:
            payload["message"] = ", ".join(
                f"{name}={available[name]}" for name in variable_names if name in available
            )
        if error is not None:
            payload["error"] = {"type": _type_name(error), "message": str(error)}
        _write(payload, path, console, compact)

    def tracer(frame: FrameType, event: str, arg: Any) -> Any:
        nonlocal target_frame
        if target_frame is None and event == "call" and frame.f_code is function.__code__:
            target_frame = frame
            return tracer
        if frame is not target_frame:
            # The global trace still sees future calls. Returning None here keeps
            # model/framework helper frames from receiving costly line events.
            return None
        if event == "line":
            emit(frame, "line")
        elif event == "return":
            emit(frame, "return")
        elif event == "exception":
            exception = arg[1] if isinstance(arg, tuple) and len(arg) > 1 else None
            emit(frame, "exception", exception)
        return tracer

    return tracer


def _run_traced(function: Callable[..., Any], tracer: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    previous_trace = sys.gettrace()
    sys.settrace(tracer)
    try:
        return function(*args, **kwargs)
    finally:
        sys.settrace(previous_trace)


async def _run_traced_async(
    function: Callable[..., Any], tracer: Callable[..., Any], *args: Any, **kwargs: Any
) -> Any:
    previous_trace = sys.gettrace()
    sys.settrace(tracer)
    try:
        return await function(*args, **kwargs)
    finally:
        sys.settrace(previous_trace)


def record(
    *variable_names: str,
    every: int = 1,
    on_change: bool = True,
    path: Optional[str | Path] = None,
    console: Optional[bool] = None,
    compact: Optional[bool] = None,
) -> Callable[[F], F]:
    """Record local variables from a function without explicit log calls.

    Arguments can be names (``record("loss", "accuracy")``) or one format
    template (``record("step={step}, loss={loss:.4f}")``). Values are inspected
    at Python line boundaries. By default an entry is written only when at least
    one requested variable changes.
    """
    if not variable_names or any(not name for name in variable_names):
        raise ValueError("record() requires at least one non-empty variable name")
    templates = [name for name in variable_names if "{" in name or "}" in name]
    if len(templates) > 1:
        raise ValueError("record() accepts at most one format template")
    template = templates[0] if templates else None
    if template is not None:
        explicit_names = tuple(name for name in variable_names if name != template)
        variable_names = tuple(dict.fromkeys(_template_fields(template) + explicit_names))
    if len(set(variable_names)) != len(variable_names):
        raise ValueError("variable names must be unique")
    if every < 1:
        raise ValueError("every must be at least 1")

    def decorate(function: F) -> F:
        if inspect.isgeneratorfunction(function):
            raise TypeError("generator functions are not supported")

        def new_tracer() -> Callable[..., Any]:
            if path is not None:
                output_path = Path(path)
            elif _settings.path is not None:
                output_path = _settings.path
            else:
                output_path = _default_path(function)
            show_console = console if console is not None else _settings.console
            compact_output = compact if compact is not None else _settings.compact
            return _make_tracer(
                function,
                tuple(variable_names),
                every,
                on_change,
                output_path,
                show_console,
                template,
                compact_output,
            )

        if inspect.iscoroutinefunction(function):
            @functools.wraps(function)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                return await _run_traced_async(function, new_tracer(), *args, **kwargs)

            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return _run_traced(function, new_tracer(), *args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorate
