from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from qwen_image_19.contracts import public_path

try:
    from rich.console import Console as RichConsole
    from rich.panel import Panel
    from rich.table import Table
except Exception:  # pragma: no cover - exercised indirectly in environments without rich
    RichConsole = None  # type: ignore[assignment]
    Panel = None  # type: ignore[assignment]
    Table = None  # type: ignore[assignment]


class _FallbackConsole:
    def print(self, message: object = "") -> None:
        print(message, file=sys.stderr)

    def print_json(self, *, data: str) -> None:
        print(data, file=sys.stderr)


console = RichConsole(stderr=True) if RichConsole is not None else _FallbackConsole()
RICH_ENABLED = RichConsole is not None


def log_stage_start(stage: str, **details: Any) -> None:
    if RICH_ENABLED:
        body = "\n".join(f"[bold]{key}[/bold]: {value}" for key, value in details.items() if value not in (None, "", [], {}))
        console.print(Panel(body or "starting", title=f"{stage} start", border_style="cyan"))  # type: ignore[arg-type]
        return
    payload = ", ".join(f"{key}={value}" for key, value in details.items() if value not in (None, "", [], {}))
    console.print(f"[{stage}] start {payload}".rstrip())


def log_stage_progress(stage: str, message: str, **details: Any) -> None:
    if RICH_ENABLED:
        console.print(f"[cyan][{stage}][/cyan] {message}")
        if details:
            table = Table(show_header=False, box=None, pad_edge=False)  # type: ignore[operator]
            table.add_column("key", style="bold")
            table.add_column("value")
            for key, value in details.items():
                if value in (None, "", [], {}):
                    continue
                table.add_row(str(key), str(value))
            if table.row_count:
                console.print(table)
        return
    payload = ", ".join(f"{key}={value}" for key, value in details.items() if value not in (None, "", [], {}))
    console.print(f"[{stage}] {message}" + (f" :: {payload}" if payload else ""))


def log_stage_complete(stage: str, result: dict[str, Any]) -> None:
    if RICH_ENABLED:
        table = Table(title=f"{stage} complete")  # type: ignore[operator]
        table.add_column("Field", style="bold green")
        table.add_column("Value")
        for key in ("run_id", "run_dir", "artifact_dir", "written", "run_count"):
            value = result.get(key)
            if value in (None, "", [], {}):
                continue
            table.add_row(key, _render_value(value))
        step_result = result.get("step_result")
        if isinstance(step_result, dict):
            table.add_row("step", str(step_result.get("step")))
            table.add_row("input_checkpoint", str(step_result.get("input_checkpoint")))
            table.add_row("output_checkpoint", str(step_result.get("output_checkpoint")))
        console.print(table)
        return
    summary = {
        "stage": stage,
        "run_id": result.get("run_id"),
        "run_dir": result.get("run_dir"),
        "artifact_dir": result.get("artifact_dir"),
        "written": result.get("written"),
        "run_count": result.get("run_count"),
        "step_result": result.get("step_result"),
    }
    console.print(f"[{stage}] complete {json.dumps(summary, default=str)}")


def _render_value(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(public_path(item) if isinstance(item, (str, Path)) else str(item) for item in value)
    if isinstance(value, (str, Path)):
        return public_path(value)
    return str(value)
