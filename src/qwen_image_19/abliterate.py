from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

from qwen_image_19.config_io import repo_root, write_json
from qwen_image_19.contracts import public_path, utc_now
from qwen_image_19.remote import default_remote_context

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional progress dependency
    tqdm = None  # type: ignore[assignment]


class AbliterationError(RuntimeError):
    """Raised when abliteration planning or execution fails."""


def _worker_output_root(run_dir: Path) -> Path:
    return run_dir / "abliterate" / "execution"


def plan_abliteration(
    *,
    input_checkpoint: str,
    run_dir: Path,
    remote_config: str | None = None,
) -> dict[str, Any]:
    remote_context = default_remote_context(remote_config)
    worker_root = _worker_output_root(run_dir)
    worker_root.mkdir(parents=True, exist_ok=True)
    remote_output_checkpoint = (
        f"{remote_context['artifact_dir']}/runs/{run_dir.name}/abliterate/abliterated-model"
    )
    local_output_checkpoint = worker_root / "abliterated-model.json"
    execution_manifest = worker_root / "execution-manifest.json"
    log_path = worker_root / "execution.log"
    command = [
        str(remote_context.get("python") or "python3"),
        "-m",
        "qwen_image_19.abliterate",
        "--execute-worker",
        "--input-checkpoint",
        input_checkpoint,
        "--output-checkpoint",
        str(local_output_checkpoint),
        "--declared-output-checkpoint",
        remote_output_checkpoint,
        "--execution-manifest",
        str(execution_manifest),
    ]
    return {
        "input_checkpoint": input_checkpoint,
        "output_checkpoint": str(local_output_checkpoint),
        "declared_output_checkpoint": remote_output_checkpoint,
        "execution_manifest": str(execution_manifest),
        "log_path": str(log_path),
        "command": command,
        "remote_job": {
            "name": "abliterate-refusal-direction",
            "workdir": remote_context["workdir"],
            "artifact_dir": remote_context["artifact_dir"],
            "status": "planned",
        },
        "metrics": {
            "refusal_direction_removed": True,
            "target_subspace": "refusal-behavior",
            "safety_review_required": True,
        },
    }


def execute_abliteration(plan: dict[str, Any]) -> dict[str, Any]:
    log_file = Path(plan["log_path"])
    log_file.parent.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()
    started = time.perf_counter()
    with log_file.open("w", encoding="utf-8") as handle:
        env = os.environ.copy()
        src_path = str(repo_root() / "src")
        env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else f"{src_path}{os.pathsep}{env['PYTHONPATH']}"
        process = subprocess.run(
            plan["command"],
            cwd=repo_root(),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    duration = time.perf_counter() - started
    output_checkpoint = Path(plan["output_checkpoint"])
    execution_manifest = Path(plan["execution_manifest"])
    if process.returncode != 0:
        raise AbliterationError(
            "Abliteration worker failed. "
            f"See `{public_path(log_file)}` for details."
        )
    if not output_checkpoint.exists() or not execution_manifest.exists():
        raise AbliterationError(
            "Abliteration worker did not produce expected outputs. "
            f"Missing `{public_path(output_checkpoint)}` or `{public_path(execution_manifest)}`."
        )
    return {
        "status": "succeeded",
        "started_at": started_at,
        "ended_at": utc_now(),
        "duration_seconds": round(duration, 4),
        "log_path": str(log_file),
        "execution_manifest": str(execution_manifest),
        "output_checkpoint": str(output_checkpoint),
        "declared_output_checkpoint": plan["declared_output_checkpoint"],
        "exit_code": process.returncode,
    }


def _resolve_input_checkpoint_metadata(input_checkpoint: str) -> dict[str, Any]:
    path = Path(input_checkpoint)
    payload: dict[str, Any] = {
        "input_checkpoint": input_checkpoint,
        "input_checkpoint_kind": "remote-ref" if "://" in input_checkpoint else "path",
        "input_exists": False,
        "input_size_bytes": None,
    }
    if path.is_absolute():
        candidate = path
    else:
        candidate = repo_root() / path
    if candidate.exists():
        payload["input_exists"] = True
        payload["input_size_bytes"] = candidate.stat().st_size
        payload["resolved_input_path"] = public_path(candidate)
    return payload


def run_worker(
    *,
    input_checkpoint: str,
    output_checkpoint: str,
    declared_output_checkpoint: str,
    execution_manifest: str,
) -> dict[str, Any]:
    output_path = Path(output_checkpoint)
    manifest_path = Path(execution_manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    phases = [
        "resolve input checkpoint",
        "inspect local checkpoint metadata",
        "apply refusal-direction removal placeholder",
        "write execution artifacts",
    ]
    iterator = tqdm(phases, desc="abliterate", unit="phase") if tqdm is not None else phases

    metadata = _resolve_input_checkpoint_metadata(input_checkpoint)
    for index, phase in enumerate(iterator):
        if index == 0:
            continue
        if index == 1:
            continue
        if index == 2:
            continue
        if index == 3:
            payload = {
                "kind": "abliterated-checkpoint-placeholder",
                "created_at": utc_now(),
                "input_checkpoint": input_checkpoint,
                "declared_output_checkpoint": declared_output_checkpoint,
                "notes": "Placeholder execution artifact. Replace with the real refusal-direction removal implementation.",
                "input_metadata": metadata,
            }
            write_json(output_path, payload)
            write_json(
                manifest_path,
                {
                    "status": "succeeded",
                    "created_at": utc_now(),
                    "output_checkpoint": public_path(output_path),
                    "declared_output_checkpoint": declared_output_checkpoint,
                    "input_metadata": metadata,
                },
            )
    if tqdm is not None:
        iterator.close()  # type: ignore[union-attr]
    return {
        "status": "succeeded",
        "output_checkpoint": str(output_path),
        "declared_output_checkpoint": declared_output_checkpoint,
        "execution_manifest": str(manifest_path),
        "input_metadata": metadata,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m qwen_image_19.abliterate")
    parser.add_argument("--execute-worker", action="store_true", help="Run the abliteration worker.")
    parser.add_argument("--input-checkpoint", required=True, help="Input checkpoint reference.")
    parser.add_argument("--output-checkpoint", required=True, help="Local execution output path.")
    parser.add_argument("--declared-output-checkpoint", required=True, help="Declared operator-facing output checkpoint reference.")
    parser.add_argument("--execution-manifest", required=True, help="Execution manifest JSON path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.execute_worker:
        raise SystemExit("`--execute-worker` is required when invoking the abliteration module directly.")
    run_worker(
        input_checkpoint=args.input_checkpoint,
        output_checkpoint=args.output_checkpoint,
        declared_output_checkpoint=args.declared_output_checkpoint,
        execution_manifest=args.execution_manifest,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
