from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qwen_image_19.config_io import repo_root, write_json, write_text


# ── Pipeline definition ────────────────────────────────────────────
PIPELINE_STEPS = (
    "merge",
    "post_merge_train",
    "abliterate",
    "post_abliterate_train",
    "quantize",
    "post_quantize_eval",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# ── Path helpers ────────────────────────────────────────────────────
def _is_remote_uri(value: str) -> bool:
    prefixes = ("s3://", "gs://", "hf://", "ssh://", "http://", "https://")
    return value.startswith(prefixes)


def public_path(value: str | Path) -> str:
    if isinstance(value, str) and _is_remote_uri(value):
        return value
    path = Path(value)
    try:
        return path.relative_to(repo_root()).as_posix()
    except ValueError:
        return path.as_posix()


# ── Artifact reference ──────────────────────────────────────────────
def artifact_ref(
    *,
    kind: str,
    path_or_uri: str | Path,
    content_type: str,
    size_bytes: int | None = None,
    label: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "kind": kind,
        "path_or_uri": public_path(path_or_uri),
        "content_type": content_type,
        "size_bytes": size_bytes,
    }
    if label:
        payload["label"] = label
    return payload


# ── Step record ─────────────────────────────────────────────────────
def default_step_record(step: str) -> dict[str, Any]:
    return {
        "step": step,
        "status": "pending",
        "input_checkpoint": None,
        "output_checkpoint": None,
        "command": [],
        "remote_job": {},
        "artifacts": [],
        "metrics": {},
        "eval_summary": None,
        "step_result": None,
        "report": None,
        "updated_at": None,
    }


# ── Run manifest ────────────────────────────────────────────────────
def create_run_manifest(
    *,
    run_id: str,
    source_models: dict[str, Any],
    artifact_root: Path,
    tags: list[str] | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    timestamp = utc_now()
    return {
        "run_id": run_id,
        "created_at": timestamp,
        "updated_at": timestamp,
        "artifact_root": public_path(artifact_root),
        "source_models": source_models,
        "steps": {step: default_step_record(step) for step in PIPELINE_STEPS},
        "report_index": public_path(artifact_root / "report-index.json"),
        "tags": tags or [],
        "notes": notes or "",
    }


def ensure_run_manifest(
    *,
    run_id: str,
    runs_root: Path,
    source_models: dict[str, Any],
    tags: list[str] | None = None,
    notes: str | None = None,
) -> tuple[dict[str, Any], Path]:
    run_dir = runs_root / run_id
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        import json

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["updated_at"] = utc_now()
        # back-fill any steps added after the manifest was created
        for step in PIPELINE_STEPS:
            if step not in manifest["steps"]:
                manifest["steps"][step] = default_step_record(step)
        return manifest, run_dir
    manifest = create_run_manifest(
        run_id=run_id,
        source_models=source_models,
        artifact_root=run_dir,
        tags=tags,
        notes=notes,
    )
    return manifest, run_dir


def write_run_manifest(manifest: dict[str, Any], run_dir: Path) -> Path:
    manifest["updated_at"] = utc_now()
    return write_json(run_dir / "manifest.json", manifest)


# ── Step result builder ─────────────────────────────────────────────
def build_step_result(
    *,
    run_id: str,
    step: str,
    status: str = "completed",
    input_checkpoint: str | None,
    output_checkpoint: str | None,
    command: list[str],
    remote_job: dict[str, Any],
    artifacts: list[dict[str, Any]],
    metrics: dict[str, Any],
    eval_summary_path: str | None = None,
    report_path: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "run_id": run_id,
        "step": step,
        "status": status,
        "input_checkpoint": input_checkpoint,
        "output_checkpoint": output_checkpoint,
        "command": command,
        "remote_job": remote_job,
        "artifacts": artifacts,
        "metrics": metrics,
        "eval_summary": eval_summary_path,
        "report": report_path,
        "updated_at": utc_now(),
    }
    if extra:
        payload["extra"] = extra
    return payload


# ── Bundle writer ───────────────────────────────────────────────────
def write_step_bundle(
    *,
    run_dir: Path,
    step: str,
    step_result: dict[str, Any],
    eval_summary: dict[str, Any],
    report_markdown: str,
) -> dict[str, Path]:
    step_dir = run_dir / step
    eval_path = write_json(step_dir / "eval-summary.json", eval_summary)
    result_path = write_json(step_dir / "step-result.json", step_result)
    report_path = write_text(step_dir / "README.md", report_markdown)
    samples_dir = step_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    return {
        "step_dir": step_dir,
        "eval_summary": eval_path,
        "step_result": result_path,
        "report": report_path,
        "samples_dir": samples_dir,
    }


# ── Manifest updater ───────────────────────────────────────────────
def update_manifest_with_step(
    manifest: dict[str, Any],
    *,
    step: str,
    step_result: dict[str, Any],
    eval_summary_path: Path,
    step_result_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    record = manifest["steps"][step]
    record.update(
        {
            "status": step_result["status"],
            "input_checkpoint": step_result["input_checkpoint"],
            "output_checkpoint": step_result["output_checkpoint"],
            "command": step_result["command"],
            "remote_job": step_result["remote_job"],
            "artifacts": step_result["artifacts"],
            "metrics": step_result["metrics"],
            "eval_summary": public_path(eval_summary_path),
            "step_result": public_path(step_result_path),
            "report": public_path(report_path),
            "updated_at": utc_now(),
        }
    )
    return manifest
