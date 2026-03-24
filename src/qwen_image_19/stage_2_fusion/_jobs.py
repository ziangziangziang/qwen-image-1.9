from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any

from qwen_image_19.config_io import repo_root, write_json
from qwen_image_19.stage_2_fusion._manifest import (
    Stage2FusionError,
    _describe_job,
    _emit_progress,
    _fmt_duration,
    repo_relative_path,
    stage2_remote_path,
)


def ensure_outputs_exist(paths: list[str]) -> list[str]:
    missing = []
    for relative in paths:
        path = repo_root() / relative
        if not path.exists():
            missing.append(relative)
            continue
        if path.is_file() and path.stat().st_size == 0:
            missing.append(relative)
            continue
        if path.is_dir() and not any(path.iterdir()):
            missing.append(relative)
    return missing


def clear_target_path(relative_path: str) -> bool:
    path = repo_root() / relative_path
    if not path.exists():
        return False
    if path.is_dir():
        shutil.rmtree(path)
        return True
    path.unlink()
    return True


def clear_output_targets(paths: list[str]) -> int:
    removed = 0
    for relative_path in paths:
        if clear_target_path(relative_path):
            removed += 1
    return removed


def run_subprocess_job(
    command: list[str],
    log_path: str,
    workdir: str,
) -> tuple[int, float]:
    log_file = repo_root() / log_path
    log_file.parent.mkdir(parents=True, exist_ok=True)
    (repo_root() / workdir).mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_file.open("w", encoding="utf-8") as handle:
        process = subprocess.run(
            command,
            cwd=repo_root(),
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    duration = time.perf_counter() - started
    return process.returncode, duration


def diffusion_resource_args(manifest: dict[str, Any]) -> list[str]:
    profile = manifest.get("resource_profile", {})
    required_gpus = int(profile.get("num_gpus", 1))
    required_total_vram_gb = float(profile.get("vram_target_gb", 80))
    return [
        "--required-gpus",
        str(required_gpus),
        "--required-total-vram-gb",
        f"{required_total_vram_gb:g}",
    ]


def _write_edit_prompts_json(output_path: Path, dataset_manifest_path: Path, n_pairs: int) -> None:
    """Extract edit_teacher source/instruction pairs from the dataset manifest and write to JSON.

    Falls back to hardcoded defaults if the manifest is absent.
    """
    pairs: list[dict[str, str]] = []
    if dataset_manifest_path.exists():
        try:
            dm = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
            for record in dm.get("planned_records", []):
                if record.get("output_metadata", {}).get("task") == "generate-then-edit":
                    entry: dict[str, str] = {}
                    if "source_prompt" in record:
                        entry["source_prompt"] = record["source_prompt"]
                    if "edit_instruction" in record:
                        entry["edit_instruction"] = record["edit_instruction"]
                    if "source_prompt" in entry and "edit_instruction" in entry:
                        pairs.append(entry)
        except (json.JSONDecodeError, KeyError):
            pass
    if not pairs:
        pairs = [
            {
                "source_prompt": "street style portrait of a runner in a red jacket against a subway wall",
                "edit_instruction": "change the jacket to white while keeping pose, lighting, and camera angle",
            },
            {
                "source_prompt": "product shot of a ceramic mug on a wooden table in morning light",
                "edit_instruction": "replace the mug pattern with blue stripes and keep the same composition",
            },
            {
                "source_prompt": "retro poster of a rocket launch with bold orange typography",
                "edit_instruction": "update the poster palette to teal and cream while preserving layout",
            },
            {
                "source_prompt": "close portrait of a corgi in a yellow raincoat on wet pavement",
                "edit_instruction": "switch the raincoat to forest green and keep the dog expression unchanged",
            },
        ]
    pairs = pairs[:max(1, n_pairs)]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(pairs, indent=2) + "\n", encoding="utf-8")


def diffusion_sampling_args(manifest: dict[str, Any]) -> list[str]:
    limits = manifest.get("limits", {})
    true_cfg_scale = float(limits.get("poc_true_cfg_scale", 4.0))
    guidance_scale = float(limits.get("poc_guidance_scale", 1.0))
    negative_prompt = str(
        limits.get(
            "poc_negative_prompt",
            "low resolution, low quality, deformed limbs, deformed fingers, oversaturated image, waxy skin, over-smoothed face, artificial look, chaotic composition, blurry text, distorted text",
        )
    )
    return [
        "--true-cfg-scale",
        f"{true_cfg_scale:g}",
        "--guidance-scale",
        f"{guidance_scale:g}",
        "--negative-prompt",
        negative_prompt,
    ]


def build_job_command(
    job_name: str,
    job_payload: dict[str, Any],
    manifest: dict[str, Any],
    dataset_manifest: dict[str, Any],
    remote_context: dict[str, Any],
) -> list[str]:
    python_cmd = str(remote_context.get("python") or "python3")
    poc_steps = int(manifest["limits"].get("poc_steps", 6))
    poc_side = int(manifest["limits"].get("poc_side", 512))
    prompt = "studio product photo of a camera on a clean table"
    if job_name == "core_delta_sweep":
        candidate = manifest["selected_core_candidate"] if manifest["run_profile"] == "smoke" else manifest["core_delta_candidates"][0]
        return [
            python_cmd,
            str(repo_root() / "scripts" / "stage-2-build-edit-delta.py"),
            "--execute",
            "--candidate-id",
            candidate["candidate_id"],
            "--output-checkpoint",
            candidate["output_checkpoint"],
            "--foundation-model",
            manifest["core_delta_recipe"]["foundation_model"],
            "--edit-model",
            manifest["core_delta_recipe"]["delta_source_model"],
            "--blend-weight",
            str(candidate["blend_weight"]),
            "--prompt",
            prompt,
            "--steps",
            str(poc_steps),
            "--width",
            str(poc_side),
            "--height",
            str(poc_side),
            *diffusion_sampling_args(manifest),
            *diffusion_resource_args(manifest),
        ]
    if job_name == "core_smoke_eval":
        return [
            python_cmd,
            str(repo_root() / "scripts" / "stage-2-compose-bf16-checkpoint.py"),
            "--execute",
            "--task",
            "core-smoke",
            "--model-ref",
            manifest["selected_core_candidate"]["output_checkpoint"],
            "--model-id",
            manifest["core_delta_recipe"]["foundation_model"],
            "--output",
            manifest["selected_core_candidate"]["smoke_report"],
            "--num-prompts",
            str(int(manifest["limits"].get("eval_prompt_count", 6))),
            "--steps",
            str(poc_steps),
            "--width",
            str(poc_side),
            "--height",
            str(poc_side),
            *diffusion_sampling_args(manifest),
            *diffusion_resource_args(manifest),
        ]
    if job_name == "teacher_dataset_generation":
        return [
            python_cmd,
            str(repo_root() / "scripts" / "stage-2-generate-teacher-dataset.py"),
            "--execute",
            "--manifest",
            manifest["dataset"]["manifest_path"],
            "--max-steps",
            str(poc_steps),
            "--max-side",
            str(poc_side),
            *diffusion_sampling_args(manifest),
            *diffusion_resource_args(manifest),
        ]
    if job_name == "layered_bridge_train":
        recipe_limits = manifest["layered_bridge_recipe"].get("training_limits", {})
        profile_limits = manifest.get("limits", {})
        max_steps = int(recipe_limits.get("max_steps") or profile_limits.get("bridge_train_steps", 500))
        batch_size = int(recipe_limits.get("batch_size") or profile_limits.get("bridge_batch_size", 1))
        return [
            python_cmd,
            str(repo_root() / "scripts" / "stage-2-build-layered-bridge.py"),
            "--execute",
            "--output-adapter",
            manifest["layered_bridge_recipe"]["output_adapter"],
            "--output-checkpoint",
            manifest["layered_bridge_recipe"]["output_checkpoint"],
            "--metrics-output",
            stage2_remote_path("stage-2", "metrics", "layered-bridge-train.json"),
            "--dataset-root",
            manifest["dataset"]["output_root"],
            "--max-steps",
            str(max_steps),
            "--batch-size",
            str(batch_size),
        ]
    if job_name == "experimental_smoke_eval":
        return [
            python_cmd,
            str(repo_root() / "scripts" / "stage-2-compose-bf16-checkpoint.py"),
            "--execute",
            "--task",
            "experimental-smoke",
            "--model-ref",
            manifest["layered_bridge_recipe"]["output_checkpoint"],
            "--model-id",
            manifest["layered_bridge_recipe"]["foundation_model"],
            "--output",
            stage2_remote_path("stage-2", "evals", "experimental", "smoke-summary.json"),
            "--num-prompts",
            str(int(manifest["limits"].get("eval_prompt_count", 6))),
            "--steps",
            str(poc_steps),
            "--width",
            str(poc_side),
            "--height",
            str(poc_side),
            *diffusion_sampling_args(manifest),
            *diffusion_resource_args(manifest),
        ]
    if job_name == "core_edit_eval":
        n_pairs = int(manifest["limits"].get("eval_edit_prompt_count", 3))
        edit_prompts_path = stage2_remote_path("stage-2", "evals", "core-edit", "edit-prompts.json")
        # Write the edit prompts JSON from the dataset manifest's edit_teacher split
        _write_edit_prompts_json(
            repo_root() / edit_prompts_path,
            repo_root() / manifest["dataset"]["manifest_path"],
            n_pairs,
        )
        return [
            python_cmd,
            str(repo_root() / "scripts" / "stage-2-compose-bf16-checkpoint.py"),
            "--execute",
            "--eval-type", "edit",
            "--task", "core-edit",
            "--model-ref",
            manifest["selected_core_candidate"]["output_checkpoint"],
            "--model-id",
            manifest["core_delta_recipe"]["foundation_model"],
            "--foundation-model-id",
            manifest["core_delta_recipe"]["foundation_model"],
            "--edit-prompts-json",
            edit_prompts_path,
            "--output",
            stage2_remote_path("stage-2", "evals", "core-edit", "edit-summary.json"),
            "--num-prompts",
            str(n_pairs),
            "--steps",
            str(poc_steps),
            "--width",
            str(poc_side),
            "--height",
            str(poc_side),
            *diffusion_sampling_args(manifest),
            *diffusion_resource_args(manifest),
        ]
    if job_name == "consistency_eval":
        n_prompts = int(manifest["limits"].get("consistency_eval_prompt_count", 4))
        return [
            python_cmd,
            str(repo_root() / "scripts" / "stage-2-compose-bf16-checkpoint.py"),
            "--execute",
            "--eval-type", "consistency",
            "--task", "consistency",
            "--model-id",
            manifest["core_delta_recipe"]["foundation_model"],
            "--consistency-baseline-model-id",
            manifest["core_delta_recipe"]["foundation_model"],
            "--output",
            stage2_remote_path("stage-2", "evals", "consistency", "consistency-summary.json"),
            "--num-prompts",
            str(n_prompts),
            "--steps",
            str(poc_steps),
            "--width",
            str(poc_side),
            "--height",
            str(poc_side),
            *diffusion_sampling_args(manifest),
            *diffusion_resource_args(manifest),
        ]
    raise Stage2FusionError(f"Unsupported remote job `{job_name}` in Stage 2 executor.")


def run_stage2_jobs(
    manifest: dict[str, Any],
    dataset_manifest: dict[str, Any],
    artifact_paths: dict[str, Path],
    remote_context: dict[str, Any],
    resume: bool = False,
) -> dict[str, Any]:
    cleanup_performed = False
    status_payload = {
        "stage": "stage2",
        "run_profile": manifest["run_profile"],
        "execution_policy": "resume" if resume else "overwrite",
        "cleanup_performed": cleanup_performed,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "jobs": {},
        "summary": {
            "failed_job": None,
            "resume_hint": None,
        },
    }
    existing_status = artifact_paths["run_status_json"]
    if resume and existing_status.exists():
        try:
            status_payload = json.loads(existing_status.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
        status_payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        status_payload["run_profile"] = manifest["run_profile"]
        status_payload["execution_policy"] = "resume"
        status_payload.setdefault("jobs", {})
        status_payload.setdefault("summary", {"failed_job": None, "resume_hint": None})
    elif not resume and existing_status.exists():
        existing_status.unlink()
        cleanup_performed = True
        status_payload["cleanup_performed"] = cleanup_performed

    job_names = list(manifest["remote_jobs"].keys())
    total_jobs = len(job_names)
    _profile = manifest.get("run_profile", "?")
    _tag = f"[q19|stage2|{_profile}]"
    _run_started = time.perf_counter()
    _emit_progress(
        f"{_tag} Starting {total_jobs} jobs  "
        f"policy={status_payload['execution_policy']}"
    )
    _emit_progress(f"{_tag} Plan: {' -> '.join(job_names)}")
    for job_idx, job_name in enumerate(job_names):
        payload = manifest["remote_jobs"][job_name]
        existing = status_payload["jobs"].get(job_name, {})
        if resume and existing.get("status") == "succeeded":
            missing = ensure_outputs_exist(payload["outputs"])
            if not missing:
                _emit_progress(
                    f"{_tag} [{job_idx + 1}/{total_jobs}] SKIPPED   {job_name}"
                    f"  (already succeeded)"
                )
                status_payload["jobs"][job_name] = {
                    **existing,
                    "status": "skipped",
                    "skip_reason": "already_succeeded",
                }
                write_json(artifact_paths["run_status_json"], status_payload)
                continue

        if not resume:
            removed_count = clear_output_targets(payload["outputs"])
            if removed_count > 0:
                cleanup_performed = True
                status_payload["cleanup_performed"] = True

        _emit_progress(f"{_tag} [{job_idx + 1}/{total_jobs}] RUNNING   {job_name}")
        for _detail in _describe_job(job_name, manifest):
            _emit_progress(f"{_tag}{_detail}")

        if job_name == "core_delta_sweep":
            commands = []
            started_iso = datetime.now(timezone.utc).isoformat()
            status_payload["jobs"][job_name] = {
                "status": "running",
                "commands": [],
                "workdir": payload["workdir"],
                "stdout_stderr_log": payload["log_path"],
                "started_at": started_iso,
            }
            write_json(artifact_paths["run_status_json"], status_payload)
            started = time.perf_counter()
            for candidate in manifest["core_delta_candidates"]:
                poc_steps = int(manifest["limits"].get("poc_steps", 6))
                poc_side = int(manifest["limits"].get("poc_side", 512))
                command = [
                    str(remote_context.get("python") or "python3"),
                    str(repo_root() / "scripts" / "stage-2-build-edit-delta.py"),
                    "--execute",
                    "--candidate-id",
                    candidate["candidate_id"],
                    "--output-checkpoint",
                    candidate["output_checkpoint"],
                    "--foundation-model",
                    manifest["core_delta_recipe"]["foundation_model"],
                    "--edit-model",
                    manifest["core_delta_recipe"]["delta_source_model"],
                    "--blend-weight",
                    str(candidate["blend_weight"]),
                    "--prompt",
                    "studio product photo of a camera on a clean table",
                    "--steps",
                    str(poc_steps),
                    "--width",
                    str(poc_side),
                    "--height",
                    str(poc_side),
                    *diffusion_sampling_args(manifest),
                    *diffusion_resource_args(manifest),
                ]
                commands.append(command)
                exit_code, _ = run_subprocess_job(command, payload["log_path"], payload["workdir"])
                if exit_code != 0:
                    duration = time.perf_counter() - started
                    status_payload["jobs"][job_name] = {
                        "status": "failed",
                        "commands": commands,
                        "workdir": payload["workdir"],
                        "stdout_stderr_log": payload["log_path"],
                        "started_at": started_iso,
                        "ended_at": datetime.now(timezone.utc).isoformat(),
                        "duration_seconds": round(duration, 4),
                        "exit_code": exit_code,
                        "outputs": payload["outputs"],
                        "missing_outputs": ensure_outputs_exist(payload["outputs"]),
                        "failure_reason": f"core sweep candidate command failed with exit code {exit_code}",
                    }
                    status_payload["summary"] = {
                        "failed_job": job_name,
                        "resume_hint": f"Re-run `q19 stage2 fuse --run-profile {manifest['run_profile']} --execute --resume` after fixing `{job_name}`.",
                    }
                    _emit_progress(
                        f"{_tag} [{job_idx + 1}/{total_jobs}] FAILED    {job_name}"
                        f"  {round(duration, 1)}s  exit={exit_code}"
                    )
                    write_json(artifact_paths["run_status_json"], status_payload)
                    raise Stage2FusionError(
                        f"Stage 2 execution failed at `{job_name}`. "
                        f"Check `{repo_relative_path(artifact_paths['run_status_json'])}` for details."
                    )
            duration = time.perf_counter() - started
            missing_outputs = ensure_outputs_exist(payload["outputs"])
            failed = bool(missing_outputs)
            status_payload["jobs"][job_name] = {
                "status": "failed" if failed else "succeeded",
                "commands": commands,
                "workdir": payload["workdir"],
                "stdout_stderr_log": payload["log_path"],
                "started_at": started_iso,
                "ended_at": datetime.now(timezone.utc).isoformat(),
                "duration_seconds": round(duration, 4),
                "exit_code": 0,
                "outputs": payload["outputs"],
                "missing_outputs": missing_outputs,
                "failure_reason": f"Missing expected outputs: {', '.join(missing_outputs)}" if failed else None,
            }
            if failed:
                _emit_progress(
                    f"{_tag} [{job_idx + 1}/{total_jobs}] FAILED    {job_name}"
                    f"  {round(duration, 1)}s  missing outputs"
                )
                status_payload["summary"] = {
                    "failed_job": job_name,
                    "resume_hint": f"Re-run `q19 stage2 fuse --run-profile {manifest['run_profile']} --execute --resume` after fixing `{job_name}`.",
                }
                write_json(artifact_paths["run_status_json"], status_payload)
                raise Stage2FusionError(
                    f"Stage 2 execution failed at `{job_name}`. "
                    f"Check `{repo_relative_path(artifact_paths['run_status_json'])}` for details."
                )
            # Synthesize core-delta metrics file so the training report can render this workflow.
            _core_delta_metrics: dict[str, Any] = {
                "workflow": "core-delta",
                "run_started_at": started_iso,
                "run_ended_at": status_payload["jobs"][job_name]["ended_at"],
                "elapsed_seconds": round(duration, 3),
                "status": "succeeded",
                "training_method": {
                    "type": "coefficient-sweep",
                    "model": (
                        f"{manifest['core_delta_recipe']['foundation_model']}"
                        f" + {manifest['core_delta_recipe']['delta_source_model']}"
                    ),
                    "objective": "edit-delta blend sweep",
                    "optimizer": "n/a",
                    "notes": (
                        f"Image-level delta sweep over blend weights "
                        f"{manifest['core_delta_recipe']['coefficient_sweep']}. "
                        "No gradient-based training."
                    ),
                },
                "candidates": [
                    {
                        "candidate_id": c["candidate_id"],
                        "blend_weight": c["blend_weight"],
                        "output_checkpoint": c["output_checkpoint"],
                    }
                    for c in manifest["core_delta_candidates"]
                ],
            }
            _core_delta_metrics_path = repo_root() / stage2_remote_path(
                "stage-2", "metrics", "core-delta-train.json"
            )
            write_json(_core_delta_metrics_path, _core_delta_metrics)
            _emit_progress(
                f"{_tag} [{job_idx + 1}/{total_jobs}] DONE      {job_name}  {round(duration, 1)}s"
            )
            write_json(artifact_paths["run_status_json"], status_payload)
            continue

        command = build_job_command(job_name, payload, manifest, dataset_manifest, remote_context)
        started_iso = datetime.now(timezone.utc).isoformat()
        status_payload["jobs"][job_name] = {
            "status": "running",
            "command": command,
            "workdir": payload["workdir"],
            "stdout_stderr_log": payload["log_path"],
            "started_at": started_iso,
        }
        write_json(artifact_paths["run_status_json"], status_payload)
        exit_code, duration = run_subprocess_job(command, payload["log_path"], payload["workdir"])
        ended_iso = datetime.now(timezone.utc).isoformat()
        missing_outputs = ensure_outputs_exist(payload["outputs"])
        failed = exit_code != 0 or bool(missing_outputs)
        job_status = {
            "status": "failed" if failed else "succeeded",
            "command": command,
            "workdir": payload["workdir"],
            "stdout_stderr_log": payload["log_path"],
            "started_at": started_iso,
            "ended_at": ended_iso,
            "duration_seconds": round(duration, 4),
            "exit_code": exit_code,
            "outputs": payload["outputs"],
            "missing_outputs": missing_outputs,
        }
        if job_name == "layered_bridge_train":
            metrics_path = repo_root() / stage2_remote_path("stage-2", "metrics", "layered-bridge-train.json")
            if metrics_path.exists():
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                if any(not math.isfinite(float(value)) for value in metrics.get("loss_curve", [])):
                    failed = True
                    job_status["status"] = "failed"
                    job_status["failure_reason"] = "Detected non-finite training metric."
        if job_name == "experimental_smoke_eval" and not failed:
            # Synthesize experimental metrics file so the training report can render this workflow.
            _smoke_eval_path = repo_root() / stage2_remote_path(
                "stage-2", "evals", "experimental", "smoke-summary.json"
            )
            _smoke_data: dict[str, Any] = (
                json.loads(_smoke_eval_path.read_text(encoding="utf-8"))
                if _smoke_eval_path.exists()
                else {}
            )
            _exp_metrics: dict[str, Any] = {
                "workflow": "experimental",
                "run_started_at": started_iso,
                "run_ended_at": ended_iso,
                "elapsed_seconds": round(duration, 3),
                "status": "succeeded",
                "training_method": {
                    "type": "experimental-smoke-eval",
                    "model": manifest["layered_bridge_recipe"]["output_checkpoint"],
                    "objective": "smoke quality check on layered bridge checkpoint",
                    "optimizer": "n/a",
                    "notes": (
                        "Experimental eval: visual pass/fail on the layered bridge checkpoint "
                        "after MSE distillation training."
                    ),
                },
                "num_prompts": _smoke_data.get("num_prompts"),
                "generated_images": _smoke_data.get("generated_images"),
                "mean_luminance": _smoke_data.get("mean_luminance"),
                "eval_status": _smoke_data.get("status", "unknown"),
            }
            _exp_metrics_path = repo_root() / stage2_remote_path(
                "stage-2", "metrics", "experimental-train.json"
            )
            write_json(_exp_metrics_path, _exp_metrics)
        if failed and "failure_reason" not in job_status:
            job_status["failure_reason"] = (
                f"Job returned exit code {exit_code}."
                if exit_code != 0
                else f"Missing expected outputs: {', '.join(missing_outputs)}"
            )
        status_payload["jobs"][job_name] = job_status
        _emit_progress(
            f"{_tag} [{job_idx + 1}/{total_jobs}] "
            + ("DONE      " if not failed else "FAILED    ")
            + f"{job_name}  {round(duration, 1)}s"
            + (f"  !! {job_status.get('failure_reason', '')}" if failed else "")
        )
        if failed:
            status_payload["summary"] = {
                "failed_job": job_name,
                "resume_hint": f"Re-run `q19 stage2 fuse --run-profile {manifest['run_profile']} --execute --resume` after fixing `{job_name}`.",
            }
            write_json(artifact_paths["run_status_json"], status_payload)
            raise Stage2FusionError(
                f"Stage 2 execution failed at `{job_name}`. "
                f"Check `{repo_relative_path(artifact_paths['run_status_json'])}` for details."
            )
        write_json(artifact_paths["run_status_json"], status_payload)

    status_payload["cleanup_performed"] = cleanup_performed
    status_payload["summary"] = {
        "failed_job": None,
        "resume_hint": None,
    }
    write_json(artifact_paths["run_status_json"], status_payload)
    _emit_progress(
        f"{_tag} All {total_jobs} jobs complete."
        f"  total {_fmt_duration(time.perf_counter() - _run_started)}"
    )
    return status_payload


