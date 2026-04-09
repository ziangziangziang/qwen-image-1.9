from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qwen_image_19.abliterate import execute_abliteration, plan_abliteration
from qwen_image_19.stage_2b_train import execute_training, load_training_config, plan_training
from qwen_image_19.config_io import repo_root, write_text
from qwen_image_19.contracts import (
    artifact_ref,
    build_step_result,
    ensure_run_manifest,
    public_path,
    update_manifest_with_step,
    write_run_manifest,
    write_step_bundle,
)
from qwen_image_19.logging_utils import log_stage_complete, log_stage_progress, log_stage_start
from qwen_image_19.remote import default_remote_context
from qwen_image_19.reporting import generate_results_summary, write_report_index
from qwen_image_19.stage_1_analysis import analyze, load_model_inventory
from qwen_image_19.stage_2_fusion import fuse
from qwen_image_19.stage_3_eval import build_eval_summary, render_eval_report
from qwen_image_19.stage_3_eval.eval_worker import run_eval_for_step
from qwen_image_19.stage_4_quant import quantize as legacy_quantize
from qwen_image_19.webserver import serve_results


def default_run_id(prefix: str = "run") -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{prefix}-{timestamp}"


def runs_root(artifact_dir: str | None = None) -> Path:
    return Path(artifact_dir) if artifact_dir else repo_root() / "reports" / "runs"


def pipeline_source_models() -> dict[str, Any]:
    inventory = load_model_inventory()
    aliases = ("qwen-image-2512", "qwen-image-edit-2511", "qwen-image-layered")
    return {alias: inventory[alias] for alias in aliases if alias in inventory}


def render_step_report(
    *,
    step: str,
    step_result: dict[str, Any],
    eval_report: str,
    extra_sections: list[str] | None = None,
) -> str:
    artifacts = "\n".join(
        f"| `{artifact['kind']}` | `{artifact['path_or_uri']}` | `{artifact['content_type']}` |"
        for artifact in step_result["artifacts"]
    )
    metrics = "\n".join(f"- `{key}`: `{value}`" for key, value in step_result["metrics"].items())
    sections = extra_sections or []
    extra = "\n\n".join(section for section in sections if section)
    return f"""# {step.title()}

## Status
- Run: `{step_result['run_id']}`
- Step: `{step}`
- Status: `{step_result['status']}`
- Input checkpoint: `{step_result['input_checkpoint']}`
- Output checkpoint: `{step_result['output_checkpoint']}`

## Remote Job
- Name: `{step_result['remote_job'].get('name', 'n/a')}`
- Workdir: `{step_result['remote_job'].get('workdir', 'n/a')}`
- Artifact dir: `{step_result['remote_job'].get('artifact_dir', 'n/a')}`
- Status: `{step_result['remote_job'].get('status', 'n/a')}`

## Metrics
{metrics}

## Artifacts
| Kind | Path | Content Type |
| --- | --- | --- |
{artifacts}

{extra}

{eval_report}
"""


def _merge_extra_sections(merge_result: dict[str, Any]) -> list[str]:
    manifest = merge_result["manifest"]
    dataset_manifest = merge_result["dataset_manifest"]
    return [
        "## Fusion Plan\n"
        f"- Run profile: `{merge_result['run_profile']}`\n"
        f"- Execution enabled: `{merge_result['execution_enabled']}`\n"
        f"- Selected checkpoint: `{manifest['selected_core_candidate']['output_checkpoint']}`",
        "## Dataset\n"
        f"- Output root: `{dataset_manifest['output_root']}`\n"
        f"- Splits: `{', '.join(dataset_manifest['splits'].keys())}`",
    ]


def _preflight_progress(event: str, details: dict[str, Any]) -> None:
    messages = {
        "load_model_inventory": "loading source model metadata",
        "load_cache_alias_map": "loading Hugging Face cache alias map",
        "resolve_remote_context": "resolving remote execution context",
        "resolve_hf_home": "resolving Hugging Face cache roots",
        "collect_snapshot_inventory": "scanning cache snapshots already present under HF_HOME",
        "inspect_cache_models": "reading cached model manifests and safetensors headers",
        "inspect_model_snapshot": "inspecting cached model snapshot",
        "build_compatibility_matrix": "building structural compatibility matrix",
        "build_weight_pairwise_analysis": "running value-level tensor comparison across roadmap pairs",
        "skip_weight_analysis": "skipping value-level tensor comparison",
        "generate_stage1_figures": "rendering stage figures",
        "write_artifacts": "writing preflight artifacts",
        "dry_run_complete": "preflight dry-run complete",
    }
    log_stage_progress("preflight", messages.get(event, event), **details)


def run_preflight(
    *,
    artifact_dir: str | None = None,
    remote_config: str | None = None,
    cache_dir: str | None = None,
    hf_home: str | None = None,
    cache_map_config: str | None = None,
    dry_run: bool = False,
    smoke_run: bool = False,
    execute: bool = False,
) -> dict[str, Any]:
    log_stage_start(
        "preflight",
        artifact_dir=artifact_dir or public_path(repo_root() / "reports" / "stage-1"),
        remote_config=remote_config or "(default)",
        dry_run=dry_run,
        smoke_run=smoke_run,
        execute=execute,
        hf_home=hf_home or "(auto)",
    )
    result = analyze(
        artifact_dir=artifact_dir,
        remote_config=remote_config,
        cache_dir=cache_dir,
        dry_run=dry_run,
        smoke_run=smoke_run,
        execute=execute,
        hf_home=hf_home,
        cache_map_config=cache_map_config,
        progress_callback=_preflight_progress,
    )
    log_stage_complete("preflight", result)
    return result


def run_merge(
    *,
    run_id: str | None = None,
    artifact_dir: str | None = None,
    remote_config: str | None = None,
    cache_dir: str | None = None,
    run_profile: str | None = None,
    dry_run: bool = False,
    smoke_run: bool = False,
    execute: bool = False,
    resume: bool = False,
    tags: list[str] | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    selected_run_id = run_id or default_run_id("merge")
    root = runs_root(artifact_dir)
    log_stage_start(
        "merge",
        run_id=selected_run_id,
        runs_root=public_path(root),
        run_profile=run_profile or ("smoke" if smoke_run else "full"),
        dry_run=dry_run,
        execute=execute,
        resume=resume,
    )
    manifest, run_dir = ensure_run_manifest(
        run_id=selected_run_id,
        runs_root=root,
        source_models=pipeline_source_models(),
        tags=tags,
        notes=notes,
    )
    step_artifact_dir = run_dir / "merge" / "artifacts"
    merge_result = fuse(
        artifact_dir=step_artifact_dir,
        remote_config=remote_config,
        cache_dir=cache_dir,
        dry_run=dry_run,
        smoke_run=smoke_run,
        run_profile=run_profile,
        execute=execute,
        resume=resume,
    )
    log_stage_progress(
        "merge",
        "fusion artifacts prepared",
        run_dir=public_path(run_dir),
        selected_checkpoint=merge_result["manifest"]["selected_core_candidate"]["output_checkpoint"],
        dataset_root=merge_result["dataset_manifest"]["output_root"],
    )
    checkpoint_ref = merge_result["manifest"]["selected_core_candidate"]["output_checkpoint"]
    eval_worker_result = run_eval_for_step(
        step="merge",
        model_id=checkpoint_ref,
        sample_root=run_dir / "merge" / "samples",
        baseline_model_id=manifest["source_models"]["qwen-image-2512"]["model_id"],
        foundation_model_id=manifest["source_models"]["qwen-image-2512"]["model_id"],
        execute=execute,
        dry_run=dry_run,
    )
    eval_summary = build_eval_summary(
        run_id=selected_run_id,
        step="merge",
        checkpoint_ref=checkpoint_ref,
        sample_root=run_dir / "merge" / "samples",
        eval_worker_result=eval_worker_result,
    )
    eval_report = render_eval_report("merge", eval_summary)
    artifacts = [
        artifact_ref(kind="manifest", path_or_uri=step_artifact_dir / "merge-manifest.json", content_type="application/json"),
        artifact_ref(kind="dataset_manifest", path_or_uri=step_artifact_dir / "dataset-manifest.json", content_type="application/json"),
        artifact_ref(kind="report", path_or_uri=step_artifact_dir / "README.md", content_type="text/markdown"),
    ]
    step_result = build_step_result(
        run_id=selected_run_id,
        step="merge",
        input_checkpoint=manifest["source_models"]["qwen-image-2512"]["model_id"],
        output_checkpoint=checkpoint_ref,
        command=["q19", "merge"] + ([f"--run-profile={merge_result['run_profile']}"] if merge_result.get("run_profile") else []),
        remote_job={
            "name": "merge-fusion",
            "workdir": default_remote_context(remote_config)["workdir"],
            "artifact_dir": default_remote_context(remote_config)["artifact_dir"],
            "status": "planned" if dry_run else "ready",
        },
        artifacts=artifacts,
        metrics={
            "run_profile": merge_result["run_profile"],
            "execution_enabled": merge_result["execution_enabled"],
            "candidate_count": len(merge_result["manifest"]["core_delta_candidates"]),
        },
        eval_summary_path=public_path(run_dir / "merge" / "eval-summary.json"),
        report_path=public_path(run_dir / "merge" / "README.md"),
        extra={
            "legacy_stage_result": {
                "mode": merge_result["mode"],
                "artifact_dir": merge_result["artifact_dir"],
            }
        },
    )
    report_markdown = render_step_report(
        step="merge",
        step_result=step_result,
        eval_report=eval_report,
        extra_sections=_merge_extra_sections(merge_result),
    )
    if dry_run:
        result = {
            "run_id": selected_run_id,
            "run_dir": public_path(run_dir),
            "step_result": step_result,
            "eval_summary": eval_summary,
            "report_preview": report_markdown,
        }
        log_stage_complete("merge", result)
        return result
    bundle = write_step_bundle(
        run_dir=run_dir,
        step="merge",
        step_result=step_result,
        eval_summary=eval_summary,
        report_markdown=report_markdown,
    )
    manifest = update_manifest_with_step(
        manifest,
        step="merge",
        step_result=step_result,
        eval_summary_path=bundle["eval_summary"],
        step_result_path=bundle["step_result"],
        report_path=bundle["report"],
    )
    write_report_index(run_dir, manifest)
    write_run_manifest(manifest, run_dir)
    result = {
        "run_id": selected_run_id,
        "run_dir": public_path(run_dir),
        "step_result": step_result,
        "written": [public_path(bundle["step_result"]), public_path(bundle["eval_summary"]), public_path(bundle["report"])],
    }
    log_stage_complete("merge", result)
    return result


def _require_previous_checkpoint(manifest: dict[str, Any], step: str, fallback: str | None = None) -> str:
    if fallback:
        return fallback
    output = manifest["steps"][step]["output_checkpoint"]
    if not output:
        raise ValueError(f"Run `{manifest['run_id']}` has no completed `{step}` output checkpoint.")
    return output


def run_train_step(
    *,
    run_id: str,
    artifact_dir: str | None = None,
    remote_config: str | None = None,
    input_checkpoint: str | None = None,
    training_config_path: str | None = None,
    dataset_dir: str | None = None,
    dry_run: bool = False,
    execute: bool = False,
    resume: bool = False,
    tags: list[str] | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    root = runs_root(artifact_dir)
    log_stage_start(
        "train",
        run_id=run_id,
        runs_root=public_path(root),
        dry_run=dry_run,
        execute=execute,
        resume=resume,
        input_checkpoint=input_checkpoint or "(from merge output)",
        training_config=training_config_path or "(defaults)",
    )
    manifest, run_dir = ensure_run_manifest(
        run_id=run_id,
        runs_root=root,
        source_models=pipeline_source_models(),
        tags=tags,
        notes=notes,
    )
    merged_checkpoint = _require_previous_checkpoint(manifest, "merge", input_checkpoint)
    training_config = load_training_config(training_config_path)
    plan = plan_training(
        input_checkpoint=merged_checkpoint,
        run_dir=run_dir,
        training_config=training_config,
        remote_config=remote_config,
        dataset_dir=dataset_dir,
    )
    if execute and not dry_run:
        log_stage_progress(
            "train",
            "launching training worker",
            log_path=public_path(plan["log_path"]),
            method=training_config["method"],
        )
        plan["remote_job"]["status"] = "running"
        execution_result = execute_training(plan)
        plan["remote_job"].update(
            {
                "status": execution_result["status"],
                "started_at": execution_result.get("started_at"),
                "ended_at": execution_result.get("ended_at"),
                "duration_seconds": execution_result.get("duration_seconds", 0),
            }
        )
        plan["output_checkpoint"] = execution_result["output_checkpoint"]
        plan["metrics"]["final_loss"] = execution_result.get("final_loss")
        plan["metrics"]["duration_seconds"] = execution_result.get("duration_seconds", 0)
        plan["metrics"]["training_status"] = execution_result["status"]
    log_stage_progress(
        "train",
        "resolved post-merge training job",
        run_dir=public_path(run_dir),
        merged_checkpoint=merged_checkpoint,
        output_checkpoint=plan["output_checkpoint"],
    )
    trained_checkpoint = plan["output_checkpoint"]
    eval_worker_result = run_eval_for_step(
        step="train",
        model_id=trained_checkpoint,
        sample_root=run_dir / "train" / "samples",
        baseline_model_id=merged_checkpoint,
        foundation_model_id=manifest["source_models"].get("qwen-image-2512", {}).get("model_id"),
        execute=execute,
        dry_run=dry_run,
    )
    eval_summary = build_eval_summary(
        run_id=run_id,
        step="train",
        checkpoint_ref=trained_checkpoint,
        sample_root=run_dir / "train" / "samples",
        eval_worker_result=eval_worker_result,
    )
    eval_report = render_eval_report("train", eval_summary)
    artifacts = [
        artifact_ref(kind="checkpoint_ref", path_or_uri=plan["output_checkpoint"], content_type="application/octet-stream"),
        artifact_ref(kind="execution_log", path_or_uri=plan["log_path"], content_type="text/plain"),
    ]
    if training_config_path:
        artifacts.append(artifact_ref(kind="training_config", path_or_uri=training_config_path, content_type="application/yaml"))
    step_result = build_step_result(
        run_id=run_id,
        step="train",
        status="planned" if (dry_run or not execute) else plan["remote_job"].get("status", "completed"),
        input_checkpoint=merged_checkpoint,
        output_checkpoint=trained_checkpoint,
        command=plan["command"],
        remote_job=plan["remote_job"],
        artifacts=artifacts,
        metrics=plan["metrics"],
        eval_summary_path=public_path(run_dir / "train" / "eval-summary.json"),
        report_path=public_path(run_dir / "train" / "README.md"),
        extra={
            "legacy_stage_result": {
                "mode": "train",
                "artifact_dir": plan["train_dir"],
            },
            "training_config": training_config,
        },
    )
    report_markdown = render_step_report(
        step="train",
        step_result=step_result,
        eval_report=eval_report,
        extra_sections=[
            "## Training Configuration\n"
            f"- Method: `{training_config['method']}`\n"
            f"- Epochs: `{training_config['epochs']}`\n"
            f"- Max steps: `{training_config['max_steps']}`\n"
            f"- Learning rate: `{training_config['learning_rate']}`\n"
            f"- LoRA rank: `{training_config.get('lora_rank', 'n/a')}`\n"
            f"- Resolution: `{training_config['resolution']}`\n"
            f"- Mixed precision: `{training_config['mixed_precision']}`",
        ],
    )
    if dry_run:
        result = {
            "run_id": run_id,
            "run_dir": public_path(run_dir),
            "step_result": step_result,
            "eval_summary": eval_summary,
            "report_preview": report_markdown,
        }
        log_stage_complete("train", result)
        return result
    bundle = write_step_bundle(
        run_dir=run_dir,
        step="train",
        step_result=step_result,
        eval_summary=eval_summary,
        report_markdown=report_markdown,
    )
    manifest = update_manifest_with_step(
        manifest,
        step="train",
        step_result=step_result,
        eval_summary_path=bundle["eval_summary"],
        step_result_path=bundle["step_result"],
        report_path=bundle["report"],
    )
    write_report_index(run_dir, manifest)
    write_run_manifest(manifest, run_dir)
    result = {
        "run_id": run_id,
        "run_dir": public_path(run_dir),
        "step_result": step_result,
        "written": [public_path(bundle["step_result"]), public_path(bundle["eval_summary"]), public_path(bundle["report"])],
    }
    log_stage_complete("train", result)
    return result
    return output


def run_abliterate_step(
    *,
    run_id: str,
    artifact_dir: str | None = None,
    remote_config: str | None = None,
    input_checkpoint: str | None = None,
    recipe_config: str | None = None,
    dry_run: bool = False,
    execute: bool = False,
) -> dict[str, Any]:
    root = runs_root(artifact_dir)
    log_stage_start(
        "abliterate",
        run_id=run_id,
        runs_root=public_path(root),
        dry_run=dry_run,
        execute=execute,
        input_checkpoint=input_checkpoint or "(from merge output)",
        recipe_config=recipe_config or "(required for execute)",
    )
    manifest, run_dir = ensure_run_manifest(
        run_id=run_id,
        runs_root=root,
        source_models=pipeline_source_models(),
    )
    merged_checkpoint = _require_previous_checkpoint(manifest, "train", input_checkpoint)
    if not merged_checkpoint:
        merged_checkpoint = _require_previous_checkpoint(manifest, "merge", input_checkpoint)
    abliteration = plan_abliteration(
        input_checkpoint=merged_checkpoint,
        run_dir=run_dir,
        recipe_config=recipe_config,
        remote_config=remote_config,
        require_local_input=execute and not dry_run,
    )
    if execute and not dry_run:
        if not recipe_config:
            raise ValueError("`q19 abliterate --execute` requires `--recipe-config`.")
        log_stage_progress(
            "abliterate",
            "launching abliteration worker",
            log_path=public_path(abliteration["log_path"]),
            recipe_config=recipe_config,
        )
        abliteration["remote_job"]["status"] = "running"
        execution_result = execute_abliteration(abliteration)
        abliteration["remote_job"].update(
            {
                "status": execution_result["status"],
                "log_path": public_path(execution_result["log_path"]),
                "started_at": execution_result["started_at"],
                "ended_at": execution_result["ended_at"],
                "duration_seconds": execution_result["duration_seconds"],
                "execution_manifest": public_path(execution_result["execution_manifest"]),
            }
        )
        abliteration["output_checkpoint"] = execution_result["output_checkpoint"]
    log_stage_progress(
        "abliterate",
        "resolved refusal-direction removal job",
        run_dir=public_path(run_dir),
        merged_checkpoint=merged_checkpoint,
        output_checkpoint=abliteration["output_checkpoint"],
    )
    abliteration_checkpoint = abliteration["declared_output_checkpoint"] if execute and not dry_run else abliteration["output_checkpoint"]
    eval_worker_result = run_eval_for_step(
        step="abliterate",
        model_id=abliteration_checkpoint,
        sample_root=run_dir / "abliterate" / "samples",
        baseline_model_id=merged_checkpoint,
        execute=execute,
        dry_run=dry_run,
    )
    eval_summary = build_eval_summary(
        run_id=run_id,
        step="abliterate",
        checkpoint_ref=abliteration_checkpoint,
        sample_root=run_dir / "abliterate" / "samples",
        eval_worker_result=eval_worker_result,
    )
    eval_report = render_eval_report("abliterate", eval_summary)
    artifacts = [
        artifact_ref(kind="checkpoint_ref", path_or_uri=abliteration["output_checkpoint"], content_type="application/octet-stream"),
        artifact_ref(kind="declared_checkpoint_ref", path_or_uri=abliteration["declared_output_checkpoint"], content_type="application/octet-stream"),
        artifact_ref(kind="execution_log", path_or_uri=abliteration["log_path"], content_type="text/plain"),
        artifact_ref(kind="execution_manifest", path_or_uri=abliteration["execution_manifest"], content_type="application/json"),
    ]
    if recipe_config:
        artifacts.append(artifact_ref(kind="recipe_config", path_or_uri=recipe_config, content_type="application/yaml"))
    step_result = build_step_result(
        run_id=run_id,
        step="abliterate",
        status="planned" if (dry_run or not execute) else "succeeded",
        input_checkpoint=merged_checkpoint,
        output_checkpoint=abliteration["output_checkpoint"],
        command=abliteration["command"],
        remote_job=abliteration["remote_job"],
        artifacts=artifacts,
        metrics={
            **abliteration["metrics"],
            "execution_requested": execute,
            "declared_output_checkpoint": abliteration["declared_output_checkpoint"],
            "recipe_config": public_path(recipe_config) if recipe_config else None,
        },
        eval_summary_path=public_path(run_dir / "abliterate" / "eval-summary.json"),
        report_path=public_path(run_dir / "abliterate" / "README.md"),
    )
    report_markdown = render_step_report(
        step="abliterate",
        step_result=step_result,
        eval_report=eval_report,
        extra_sections=[
            "## Policy\n- This run treats refusal-direction removal as an internal-only experiment.\n- Eval and review are required before the run can be considered complete."
        ],
    )
    if dry_run:
        result = {
            "run_id": run_id,
            "run_dir": public_path(run_dir),
            "step_result": step_result,
            "eval_summary": eval_summary,
            "report_preview": report_markdown,
        }
        log_stage_complete("abliterate", result)
        return result
    bundle = write_step_bundle(
        run_dir=run_dir,
        step="abliterate",
        step_result=step_result,
        eval_summary=eval_summary,
        report_markdown=report_markdown,
    )
    manifest = update_manifest_with_step(
        manifest,
        step="abliterate",
        step_result=step_result,
        eval_summary_path=bundle["eval_summary"],
        step_result_path=bundle["step_result"],
        report_path=bundle["report"],
    )
    write_report_index(run_dir, manifest)
    write_run_manifest(manifest, run_dir)
    result = {
        "run_id": run_id,
        "run_dir": public_path(run_dir),
        "step_result": step_result,
        "written": [public_path(bundle["step_result"]), public_path(bundle["eval_summary"]), public_path(bundle["report"])],
    }
    log_stage_complete("abliterate", result)
    return result


def run_quantize_step(
    *,
    run_id: str,
    artifact_dir: str | None = None,
    remote_config: str | None = None,
    cache_dir: str | None = None,
    input_checkpoint: str | None = None,
    dry_run: bool = False,
    smoke_run: bool = False,
    execute: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    root = runs_root(artifact_dir)
    log_stage_start(
        "quantize",
        run_id=run_id,
        runs_root=public_path(root),
        dry_run=dry_run,
        smoke_run=smoke_run,
        execute=execute,
        resume=resume,
        input_checkpoint=input_checkpoint or "(from abliterate output)",
    )
    manifest, run_dir = ensure_run_manifest(
        run_id=run_id,
        runs_root=root,
        source_models=pipeline_source_models(),
    )
    abliterated_checkpoint = _require_previous_checkpoint(manifest, "abliterate", input_checkpoint)
    quant_dir = run_dir / "quantize" / "artifacts"
    quant_result = legacy_quantize(
        artifact_dir=quant_dir,
        remote_config=remote_config,
        cache_dir=cache_dir,
        dry_run=dry_run,
        smoke_run=smoke_run,
        execute=execute,
        resume=resume,
    )
    log_stage_progress(
        "quantize",
        "quantization profiles resolved",
        run_dir=public_path(run_dir),
        source_checkpoint=abliterated_checkpoint,
        quant_artifact_dir=quant_result["artifact_dir"],
    )
    remote_context = default_remote_context(remote_config)
    quantized_ref = f"{remote_context['artifact_dir']}/runs/{run_id}/quantize/qwen-image-1.9-q4"
    eval_worker_result = run_eval_for_step(
        step="quantize",
        model_id=quantized_ref,
        sample_root=run_dir / "quantize" / "samples",
        baseline_model_id=abliterated_checkpoint,
        execute=execute,
        dry_run=dry_run,
    )
    eval_summary = build_eval_summary(
        run_id=run_id,
        step="quantize",
        checkpoint_ref=quantized_ref,
        sample_root=run_dir / "quantize" / "samples",
        eval_worker_result=eval_worker_result,
    )
    eval_report = render_eval_report("quantize", eval_summary)
    step_result = build_step_result(
        run_id=run_id,
        step="quantize",
        input_checkpoint=abliterated_checkpoint,
        output_checkpoint=quantized_ref,
        command=["q19", "quantize"],
        remote_job={
            "name": "quantize-checkpoint",
            "workdir": remote_context["workdir"],
            "artifact_dir": remote_context["artifact_dir"],
            "status": "planned" if dry_run else "ready",
        },
        artifacts=[
            artifact_ref(kind="quant_report", path_or_uri=quant_dir / "README.md", content_type="text/markdown"),
            artifact_ref(kind="gguf_profile", path_or_uri=repo_root() / "configs" / "quant" / "stage-4-gguf-imatrix.yaml", content_type="application/json"),
            artifact_ref(kind="gptq_profile", path_or_uri=repo_root() / "configs" / "quant" / "stage-4-exl2-gptq.yaml", content_type="application/json"),
        ],
        metrics=eval_summary["aggregate_metrics"],
        eval_summary_path=public_path(run_dir / "quantize" / "eval-summary.json"),
        report_path=public_path(run_dir / "quantize" / "README.md"),
        extra={"legacy_stage_result": quant_result},
    )
    report_markdown = render_step_report(
        step="quantize",
        step_result=step_result,
        eval_report=eval_report,
        extra_sections=[
            "## Quantization Inputs\n"
            f"- Source checkpoint: `{abliterated_checkpoint}`\n"
            f"- Legacy quant artifact dir: `{quant_result['artifact_dir']}`"
        ],
    )
    if dry_run:
        result = {
            "run_id": run_id,
            "run_dir": public_path(run_dir),
            "step_result": step_result,
            "eval_summary": eval_summary,
            "report_preview": report_markdown,
        }
        log_stage_complete("quantize", result)
        return result
    bundle = write_step_bundle(
        run_dir=run_dir,
        step="quantize",
        step_result=step_result,
        eval_summary=eval_summary,
        report_markdown=report_markdown,
    )
    manifest = update_manifest_with_step(
        manifest,
        step="quantize",
        step_result=step_result,
        eval_summary_path=bundle["eval_summary"],
        step_result_path=bundle["step_result"],
        report_path=bundle["report"],
    )
    write_report_index(run_dir, manifest)
    write_run_manifest(manifest, run_dir)
    result = {
        "run_id": run_id,
        "run_dir": public_path(run_dir),
        "step_result": step_result,
        "written": [public_path(bundle["step_result"]), public_path(bundle["eval_summary"]), public_path(bundle["report"])],
    }
    log_stage_complete("quantize", result)
    return result


def run_report(
    *,
    artifact_dir: str | None = None,
    run_id: str | None = None,
    serve: bool = False,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> dict[str, Any]:
    root = runs_root(artifact_dir)
    log_stage_start(
        "report",
        runs_root=public_path(root),
        run_id=run_id or "(all runs)",
        serve=serve,
        host=host,
        port=port,
    )
    summary = generate_results_summary(root)
    if run_id:
        report_index = root / run_id / "report-index.json"
        if not report_index.exists():
            raise ValueError(f"Run `{run_id}` does not exist under `{public_path(root)}`.")
        summary["run_report_index"] = public_path(report_index)
        log_stage_progress("report", "validated run report index", run_id=run_id, report_index=summary["run_report_index"])
    if serve:
        log_stage_progress("report", "starting results server", host=host, port=port)
        serve_results(host=host, port=port, runs_root=root)
    log_stage_complete("report", summary)
    return summary
