"""Abliteration — refusal-direction tensor removal.

Wraps the existing abliterate module with a clean plan/execute interface.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from qwen_image_19.contracts import public_path


def plan_abliteration(
    *,
    input_checkpoint: str,
    run_dir: Path,
    recipe_config: str | None = None,
    remote_config: str | None = None,
) -> dict[str, Any]:
    from qwen_image_19.remote import default_remote_context

    ctx = default_remote_context(remote_config)
    abl_dir = run_dir / "abliterate"
    # Use local path for execution; declared path for manifest references
    local_output_ckpt = str(abl_dir / "abliterated-checkpoint")
    declared_output_ckpt = f"{ctx['artifact_dir']}/runs/{run_dir.name}/abliterate/abliterated-checkpoint"
    log_path = abl_dir / "abliterate.log"

    return {
        "input_checkpoint": input_checkpoint,
        "output_checkpoint": local_output_ckpt,
        "declared_output_checkpoint": declared_output_ckpt,
        "log_path": str(log_path),
        "execution_manifest": str(abl_dir / "execution-manifest.json"),
        "recipe_config": recipe_config,
        "command": [
            str(ctx.get("python") or "python3"), "-m", "qwen_image_19.abliterate",
            "--execute-worker",
            "--input-checkpoint", input_checkpoint,
            "--output-checkpoint", local_output_ckpt,
            "--declared-output-checkpoint", declared_output_ckpt,
            "--execution-manifest", str(abl_dir / "execution-manifest.json"),
        ] + ([f"--recipe-config={recipe_config}"] if recipe_config else []),
        "remote_job": {
            "name": "abliterate",
            "workdir": ctx["workdir"],
            "artifact_dir": ctx["artifact_dir"],
            "status": "planned",
        },
        "metrics": {
            "recipe_config": recipe_config or "none",
            "input_checkpoint": input_checkpoint,
        },
    }


def execute_abliteration(plan: dict[str, Any]) -> dict[str, Any]:
    """Delegate to the real abliterate module if available."""
    from qwen_image_19.contracts import utc_now

    started_at = utc_now()
    try:
        from qwen_image_19.abliterate import execute_abliteration as _real_execute
        return _real_execute(plan)
    except ImportError:
        return {
            "status": "skipped",
            "reason": "abliterate module dependencies not available",
            "output_checkpoint": plan["output_checkpoint"],
            "log_path": plan["log_path"],
            "execution_manifest": plan["execution_manifest"],
            "started_at": started_at,
            "ended_at": utc_now(),
            "duration_seconds": 0,
        }
