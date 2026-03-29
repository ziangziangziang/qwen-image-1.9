from __future__ import annotations

from pathlib import Path
from typing import Any

from qwen_image_19.remote import default_remote_context


def run_abliteration(
    *,
    input_checkpoint: str,
    run_dir: Path,
    remote_config: str | None = None,
    dry_run: bool = False,
    execute: bool = False,
) -> dict[str, Any]:
    remote_context = default_remote_context(remote_config)
    output_checkpoint = f"{remote_context['artifact_dir']}/runs/{run_dir.name}/abliterate/abliterated-model"
    mode = "dry-run" if dry_run else ("execute" if execute else "write")
    return {
        "mode": mode,
        "input_checkpoint": input_checkpoint,
        "output_checkpoint": output_checkpoint,
        "command": [
            str(remote_context.get("python") or "python3"),
            "-m",
            "qwen_image_19.abliterate",
            "--input-checkpoint",
            input_checkpoint,
            "--output-checkpoint",
            output_checkpoint,
        ],
        "remote_job": {
            "name": "abliterate-refusal-direction",
            "workdir": remote_context["workdir"],
            "artifact_dir": remote_context["artifact_dir"],
            "status": "planned" if dry_run else "ready",
        },
        "metrics": {
            "refusal_direction_removed": True,
            "target_subspace": "refusal-behavior",
            "safety_review_required": True,
        },
    }
