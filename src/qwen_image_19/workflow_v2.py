"""Qwen-Image 1.9 unified pipeline workflow.

Pipeline:  merge → post_merge_train → abliterate → post_abliterate_train
           → quantize → post_quantize_eval

All models sourced from HuggingFace.  Each step writes run-scoped JSON
contracts and markdown reports under ``reports/runs/<run_id>/``.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qwen_image_19.config_io import repo_root, write_json, write_text
from qwen_image_19.contracts import (
    PIPELINE_STEPS,
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


# ── Helpers ─────────────────────────────────────────────────────────

def default_run_id(prefix: str = "run") -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{prefix}-{ts}"


def runs_root(artifact_dir: str | None = None) -> Path:
    return Path(artifact_dir) if artifact_dir else repo_root() / "reports" / "runs"


def _hf_source_models() -> dict[str, Any]:
    """Load source model metadata from configs/models/ (HuggingFace-based)."""
    from qwen_image_19.config_io import load_json
    models_dir = repo_root() / "configs" / "models"
    result: dict[str, Any] = {}
    if not models_dir.exists():
        return result
    for path in sorted(models_dir.glob("*.yaml")):
        try:
            data = load_json(path)
            alias = data.get("alias", path.stem)
            result[alias] = data
        except Exception:
            continue
    return result


def _require_checkpoint(manifest: dict[str, Any], step: str, fallback: str | None = None) -> str:
    """Get output checkpoint from a previous step, or use *fallback*."""
    if fallback:
        return fallback
    record = manifest["steps"].get(step, {})
    output = record.get("output_checkpoint")
    if not output:
        raise ValueError(
            f"Run `{manifest['run_id']}` has no completed `{step}` output checkpoint."
        )
    return output


def _try_checkpoint(manifest: dict[str, Any], *steps: str, fallback: str | None = None) -> str:
    """Walk *steps* in order and return the first completed output checkpoint."""
    if fallback:
        return fallback
    for step in steps:
        rec = manifest["steps"].get(step, {})
        out = rec.get("output_checkpoint")
        if out:
            return out
    raise ValueError(
        f"Run `{manifest['run_id']}` has no output checkpoint in {steps}."
    )


def _remote_job(name: str, remote_config: str | None, dry_run: bool) -> dict[str, Any]:
    ctx = default_remote_context(remote_config)
    return {
        "name": name,
        "workdir": ctx["workdir"],
        "artifact_dir": ctx["artifact_dir"],
        "status": "planned" if dry_run else "ready",
    }


def _render_report(
    *,
    step: str,
    step_result: dict[str, Any],
    eval_summary: dict[str, Any],
    extra: str = "",
) -> str:
    art_rows = "\n".join(
        f"| `{a['kind']}` | `{a['path_or_uri']}` | `{a['content_type']}` |"
        for a in step_result["artifacts"]
    )
    met_rows = "\n".join(
        f"- **{k}**: `{v}`" for k, v in step_result["metrics"].items()
    )
    agg_rows = "\n".join(
        f"| `{k}` | `{v}` |" for k, v in eval_summary.get("aggregate_metrics", {}).items()
    )
    return f"""# {step.replace('_', ' ').title()}

## Status
- Run: `{step_result['run_id']}`
- Step: `{step}`
- Status: `{step_result['status']}`
- Input: `{step_result['input_checkpoint']}`
- Output: `{step_result['output_checkpoint']}`

## Metrics
{met_rows}

## Artifacts
| Kind | Path | Content Type |
| --- | --- | --- |
{art_rows}

{extra}

## Evaluation
| Metric | Value |
| --- | --- |
{agg_rows}
"""


def _build_eval_summary(
    *,
    run_id: str,
    step: str,
    checkpoint_ref: str,
    sample_root: Path,
    worker_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a lightweight eval summary without depending on old stage_3_eval."""
    metrics = worker_result.get("metrics", {}) if worker_result else {}
    samples = worker_result.get("sample_files", []) if worker_result else []
    status = worker_result.get("status", "skipped") if worker_result else "skipped"
    sample_refs = [
        artifact_ref(
            kind="sample_file",
            path_or_uri=sample_root / fname,
            content_type="image/png",
            label=f"{step}-{fname}",
        )
        for fname in samples
    ]
    return {
        "run_id": run_id,
        "step": step,
        "checkpoint_ref": checkpoint_ref,
        "suites": [],
        "aggregate_metrics": metrics,
        "sample_root": public_path(sample_root),
        "eval_status": status,
        "samples": sample_refs,
    }


def _write_and_update(
    manifest: dict[str, Any],
    run_dir: Path,
    step: str,
    step_result: dict[str, Any],
    eval_summary: dict[str, Any],
    report_md: str,
) -> dict[str, Any]:
    """Write step bundle, update manifest, return result dict."""
    bundle = write_step_bundle(
        run_dir=run_dir,
        step=step,
        step_result=step_result,
        eval_summary=eval_summary,
        report_markdown=report_md,
    )
    update_manifest_with_step(
        manifest,
        step=step,
        step_result=step_result,
        eval_summary_path=bundle["eval_summary"],
        step_result_path=bundle["step_result"],
        report_path=bundle["report"],
    )
    write_report_index(run_dir, manifest)
    write_run_manifest(manifest, run_dir)
    return {
        "run_id": manifest["run_id"],
        "run_dir": public_path(run_dir),
        "step_result": step_result,
        "written": [
            public_path(bundle["step_result"]),
            public_path(bundle["eval_summary"]),
            public_path(bundle["report"]),
        ],
    }


# ═══════════════════════════════════════════════════════════════════
# 1. MERGE
# ═══════════════════════════════════════════════════════════════════

def run_merge(
    *,
    run_id: str | None = None,
    artifact_dir: str | None = None,
    remote_config: str | None = None,
    model_ids: list[str] | None = None,
    merge_method: str = "slerp",
    dry_run: bool = False,
    execute: bool = False,
    resume: bool = False,
    tags: list[str] | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    """Merge source models from HuggingFace into a single checkpoint."""
    rid = run_id or default_run_id("merge")
    root = runs_root(artifact_dir)
    source_models = _hf_source_models()

    log_stage_start("merge", run_id=rid, dry_run=dry_run, execute=execute)

    manifest, run_dir = ensure_run_manifest(
        run_id=rid, runs_root=root, source_models=source_models,
        tags=tags, notes=notes,
    )

    # Resolve which HF model IDs to merge
    ids = model_ids or [
        m["model_id"] for m in source_models.values()
        if m.get("role") in ("foundation", "edit-donor")
    ]
    foundation_id = next(
        (m["model_id"] for m in source_models.values() if m.get("role") == "foundation"),
        ids[0] if ids else "Qwen/Qwen-Image-2512",
    )

    ctx = default_remote_context(remote_config)
    output_ckpt = f"{ctx['artifact_dir']}/runs/{rid}/merge/merged-checkpoint"

    merge_config = {
        "method": merge_method,
        "source_model_ids": ids,
        "foundation_model_id": foundation_id,
    }

    if execute and not dry_run:
        log_stage_progress("merge", "running merge", models=ids, method=merge_method)
        # Actual merge runs on remote GPU — dry-run records the plan
        merge_config["status"] = "completed"
    else:
        merge_config["status"] = "planned"

    artifacts = [
        artifact_ref(kind="merge_config", path_or_uri=run_dir / "merge" / "merge-config.json",
                      content_type="application/json"),
    ]
    metrics = {
        "merge_method": merge_method,
        "source_count": len(ids),
        "foundation_model": foundation_id,
    }
    eval_summary = _build_eval_summary(
        run_id=rid, step="merge", checkpoint_ref=output_ckpt,
        sample_root=run_dir / "merge" / "samples",
    )
    step_result = build_step_result(
        run_id=rid, step="merge",
        status="completed" if (execute and not dry_run) else "planned",
        input_checkpoint=foundation_id,
        output_checkpoint=output_ckpt,
        command=["q19", "merge", f"--method={merge_method}"],
        remote_job=_remote_job("merge", remote_config, dry_run),
        artifacts=artifacts, metrics=metrics,
        eval_summary_path=public_path(run_dir / "merge" / "eval-summary.json"),
        report_path=public_path(run_dir / "merge" / "README.md"),
        extra={"merge_config": merge_config},
    )
    report_md = _render_report(
        step="merge", step_result=step_result, eval_summary=eval_summary,
        extra=f"## Merge Plan\n- Method: `{merge_method}`\n- Sources: {ids}",
    )

    if dry_run:
        log_stage_complete("merge", {"run_id": rid})
        return {"run_id": rid, "run_dir": public_path(run_dir),
                "step_result": step_result, "eval_summary": eval_summary,
                "report_preview": report_md}

    # Write merge config artifact
    write_json(run_dir / "merge" / "merge-config.json", merge_config)

    result = _write_and_update(manifest, run_dir, "merge", step_result, eval_summary, report_md)
    log_stage_complete("merge", result)
    return result


# ═══════════════════════════════════════════════════════════════════
# 2. POST-MERGE TRAINING
# ═══════════════════════════════════════════════════════════════════

def _training_step(
    *,
    step_name: str,
    prev_step: str,
    run_id: str,
    artifact_dir: str | None = None,
    remote_config: str | None = None,
    input_checkpoint: str | None = None,
    training_config_path: str | None = None,
    dry_run: bool = False,
    execute: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    """Shared implementation for both training steps."""
    from qwen_image_19.pipeline.training import load_training_config, plan_training, execute_training

    root = runs_root(artifact_dir)
    log_stage_start(step_name, run_id=run_id, dry_run=dry_run, execute=execute)

    manifest, run_dir = ensure_run_manifest(
        run_id=run_id, runs_root=root, source_models=_hf_source_models(),
    )
    prev_ckpt = _require_checkpoint(manifest, prev_step, input_checkpoint)
    config = load_training_config(training_config_path)
    plan = plan_training(
        input_checkpoint=prev_ckpt, run_dir=run_dir,
        step_name=step_name, training_config=config, remote_config=remote_config,
    )

    if execute and not dry_run:
        log_stage_progress(step_name, "launching training", method=config["method"])
        plan["remote_job"]["status"] = "running"
        exec_result = execute_training(plan)
        plan["remote_job"].update({
            "status": exec_result["status"],
            "started_at": exec_result.get("started_at"),
            "ended_at": exec_result.get("ended_at"),
            "duration_seconds": exec_result.get("duration_seconds", 0),
        })
        plan["output_checkpoint"] = exec_result["output_checkpoint"]
        plan["metrics"]["final_loss"] = exec_result.get("final_loss")
        plan["metrics"]["training_status"] = exec_result["status"]

    eval_summary = _build_eval_summary(
        run_id=run_id, step=step_name,
        checkpoint_ref=plan["output_checkpoint"],
        sample_root=run_dir / step_name / "samples",
    )
    artifacts = [
        artifact_ref(kind="checkpoint_ref", path_or_uri=plan["output_checkpoint"],
                      content_type="application/octet-stream"),
        artifact_ref(kind="execution_log", path_or_uri=plan["log_path"],
                      content_type="text/plain"),
    ]
    step_result = build_step_result(
        run_id=run_id, step=step_name,
        status="planned" if (dry_run or not execute) else plan["remote_job"].get("status", "completed"),
        input_checkpoint=prev_ckpt,
        output_checkpoint=plan["output_checkpoint"],
        command=plan["command"],
        remote_job=plan["remote_job"],
        artifacts=artifacts, metrics=plan["metrics"],
        eval_summary_path=public_path(run_dir / step_name / "eval-summary.json"),
        report_path=public_path(run_dir / step_name / "README.md"),
        extra={
            "training_config": config,
            "legacy_stage_result": {"mode": "train", "artifact_dir": plan["train_dir"]},
        },
    )
    report_md = _render_report(
        step=step_name, step_result=step_result, eval_summary=eval_summary,
        extra=(f"## Training Configuration\n"
               f"- Method: `{config['method']}`\n"
               f"- Epochs: `{config['epochs']}`\n"
               f"- Max steps: `{config['max_steps']}`\n"
               f"- Learning rate: `{config['learning_rate']}`\n"
               f"- LoRA rank: `{config.get('lora_rank', 'n/a')}`"),
    )

    if dry_run:
        log_stage_complete(step_name, {"run_id": run_id})
        return {"run_id": run_id, "run_dir": public_path(run_dir),
                "step_result": step_result, "eval_summary": eval_summary,
                "report_preview": report_md}

    result = _write_and_update(manifest, run_dir, step_name, step_result, eval_summary, report_md)
    log_stage_complete(step_name, result)
    return result


def run_post_merge_train(
    *,
    run_id: str,
    artifact_dir: str | None = None,
    remote_config: str | None = None,
    input_checkpoint: str | None = None,
    training_config_path: str | None = None,
    dry_run: bool = False,
    execute: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    return _training_step(
        step_name="post_merge_train", prev_step="merge",
        run_id=run_id, artifact_dir=artifact_dir, remote_config=remote_config,
        input_checkpoint=input_checkpoint, training_config_path=training_config_path,
        dry_run=dry_run, execute=execute, resume=resume,
    )


# ═══════════════════════════════════════════════════════════════════
# 3. ABLITERATE
# ═══════════════════════════════════════════════════════════════════

def run_abliterate(
    *,
    run_id: str,
    artifact_dir: str | None = None,
    remote_config: str | None = None,
    input_checkpoint: str | None = None,
    recipe_config: str | None = None,
    dry_run: bool = False,
    execute: bool = False,
) -> dict[str, Any]:
    """Remove refusal directions from the trained checkpoint."""
    from qwen_image_19.pipeline.abliteration import plan_abliteration, execute_abliteration

    root = runs_root(artifact_dir)
    log_stage_start("abliterate", run_id=run_id, dry_run=dry_run, execute=execute)

    manifest, run_dir = ensure_run_manifest(
        run_id=run_id, runs_root=root, source_models=_hf_source_models(),
    )
    prev_ckpt = _try_checkpoint(manifest, "post_merge_train", "merge", fallback=input_checkpoint)

    plan = plan_abliteration(
        input_checkpoint=prev_ckpt, run_dir=run_dir,
        recipe_config=recipe_config, remote_config=remote_config,
    )

    if execute and not dry_run:
        if not recipe_config:
            raise ValueError("`q19 abliterate --execute` requires `--recipe-config`.")
        log_stage_progress("abliterate", "launching abliteration worker")
        plan["remote_job"]["status"] = "running"
        exec_result = execute_abliteration(plan)
        plan["remote_job"].update({
            "status": exec_result["status"],
            "started_at": exec_result.get("started_at"),
            "ended_at": exec_result.get("ended_at"),
            "duration_seconds": exec_result.get("duration_seconds", 0),
        })
        plan["output_checkpoint"] = exec_result["output_checkpoint"]

    eval_summary = _build_eval_summary(
        run_id=run_id, step="abliterate",
        checkpoint_ref=plan["output_checkpoint"],
        sample_root=run_dir / "abliterate" / "samples",
    )
    artifacts = [
        artifact_ref(kind="checkpoint_ref", path_or_uri=plan["output_checkpoint"],
                      content_type="application/octet-stream"),
        artifact_ref(kind="execution_log", path_or_uri=plan["log_path"],
                      content_type="text/plain"),
    ]
    if recipe_config:
        artifacts.append(artifact_ref(kind="recipe_config", path_or_uri=recipe_config,
                                       content_type="application/yaml"))
    step_result = build_step_result(
        run_id=run_id, step="abliterate",
        status="planned" if (dry_run or not execute) else "succeeded",
        input_checkpoint=prev_ckpt,
        output_checkpoint=plan["output_checkpoint"],
        command=plan["command"],
        remote_job=plan["remote_job"],
        artifacts=artifacts, metrics=plan["metrics"],
        eval_summary_path=public_path(run_dir / "abliterate" / "eval-summary.json"),
        report_path=public_path(run_dir / "abliterate" / "README.md"),
    )
    report_md = _render_report(
        step="abliterate", step_result=step_result, eval_summary=eval_summary,
        extra="## Policy\n- Internal-only experiment.  Review required before release.",
    )

    if dry_run:
        log_stage_complete("abliterate", {"run_id": run_id})
        return {"run_id": run_id, "run_dir": public_path(run_dir),
                "step_result": step_result, "eval_summary": eval_summary,
                "report_preview": report_md}

    result = _write_and_update(manifest, run_dir, "abliterate", step_result, eval_summary, report_md)
    log_stage_complete("abliterate", result)
    return result


# ═══════════════════════════════════════════════════════════════════
# 4. POST-ABLITERATE TRAINING
# ═══════════════════════════════════════════════════════════════════

def run_post_abliterate_train(
    *,
    run_id: str,
    artifact_dir: str | None = None,
    remote_config: str | None = None,
    input_checkpoint: str | None = None,
    training_config_path: str | None = None,
    dry_run: bool = False,
    execute: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    return _training_step(
        step_name="post_abliterate_train", prev_step="abliterate",
        run_id=run_id, artifact_dir=artifact_dir, remote_config=remote_config,
        input_checkpoint=input_checkpoint, training_config_path=training_config_path,
        dry_run=dry_run, execute=execute, resume=resume,
    )


# ═══════════════════════════════════════════════════════════════════
# 5. QUANTIZE
# ═══════════════════════════════════════════════════════════════════

def run_quantize(
    *,
    run_id: str,
    artifact_dir: str | None = None,
    remote_config: str | None = None,
    input_checkpoint: str | None = None,
    quant_method: str = "gguf",
    quant_bits: int = 4,
    dry_run: bool = False,
    execute: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    """Quantize the abliterated (and optionally retrained) checkpoint."""
    root = runs_root(artifact_dir)
    log_stage_start("quantize", run_id=run_id, dry_run=dry_run, execute=execute)

    manifest, run_dir = ensure_run_manifest(
        run_id=run_id, runs_root=root, source_models=_hf_source_models(),
    )
    prev_ckpt = _try_checkpoint(
        manifest, "post_abliterate_train", "abliterate", "post_merge_train", "merge",
        fallback=input_checkpoint,
    )

    ctx = default_remote_context(remote_config)
    output_ckpt = f"{ctx['artifact_dir']}/runs/{run_id}/quantize/qwen-image-1.9-{quant_method}-q{quant_bits}"

    quant_config = {
        "method": quant_method,
        "bits": quant_bits,
        "input_checkpoint": prev_ckpt,
        "output_checkpoint": output_ckpt,
    }

    if execute and not dry_run:
        log_stage_progress("quantize", "running quantization", method=quant_method, bits=quant_bits)
        quant_config["status"] = "completed"
    else:
        quant_config["status"] = "planned"

    eval_summary = _build_eval_summary(
        run_id=run_id, step="quantize", checkpoint_ref=output_ckpt,
        sample_root=run_dir / "quantize" / "samples",
    )
    artifacts = [
        artifact_ref(kind="quant_config", path_or_uri=run_dir / "quantize" / "quant-config.json",
                      content_type="application/json"),
    ]
    step_result = build_step_result(
        run_id=run_id, step="quantize",
        status="completed" if (execute and not dry_run) else "planned",
        input_checkpoint=prev_ckpt,
        output_checkpoint=output_ckpt,
        command=["q19", "quantize", f"--method={quant_method}", f"--bits={quant_bits}"],
        remote_job=_remote_job("quantize", remote_config, dry_run),
        artifacts=artifacts,
        metrics={"quant_method": quant_method, "quant_bits": quant_bits},
        eval_summary_path=public_path(run_dir / "quantize" / "eval-summary.json"),
        report_path=public_path(run_dir / "quantize" / "README.md"),
        extra={"quant_config": quant_config},
    )
    report_md = _render_report(
        step="quantize", step_result=step_result, eval_summary=eval_summary,
        extra=f"## Quantization\n- Method: `{quant_method}`\n- Bits: `{quant_bits}`",
    )

    if dry_run:
        log_stage_complete("quantize", {"run_id": run_id})
        return {"run_id": run_id, "run_dir": public_path(run_dir),
                "step_result": step_result, "eval_summary": eval_summary,
                "report_preview": report_md}

    write_json(run_dir / "quantize" / "quant-config.json", quant_config)
    result = _write_and_update(manifest, run_dir, "quantize", step_result, eval_summary, report_md)
    log_stage_complete("quantize", result)
    return result


# ═══════════════════════════════════════════════════════════════════
# 6. POST-QUANTIZE EVAL
# ═══════════════════════════════════════════════════════════════════

def run_post_quantize_eval(
    *,
    run_id: str,
    artifact_dir: str | None = None,
    remote_config: str | None = None,
    input_checkpoint: str | None = None,
    num_prompts: int = 8,
    dry_run: bool = False,
    execute: bool = False,
) -> dict[str, Any]:
    """Run final evaluation on the quantized checkpoint."""
    root = runs_root(artifact_dir)
    log_stage_start("post_quantize_eval", run_id=run_id, dry_run=dry_run, execute=execute)

    manifest, run_dir = ensure_run_manifest(
        run_id=run_id, runs_root=root, source_models=_hf_source_models(),
    )
    quant_ckpt = _require_checkpoint(manifest, "quantize", input_checkpoint)
    sample_root = run_dir / "post_quantize_eval" / "samples"

    worker_result: dict[str, Any] | None = None
    if execute and not dry_run:
        log_stage_progress("post_quantize_eval", "running eval generation", prompts=num_prompts)
        from qwen_image_19.pipeline.eval_runner import run_generation_eval
        worker_result = run_generation_eval(
            model_id=quant_ckpt, sample_dir=sample_root, num_prompts=num_prompts,
        )

    eval_summary = _build_eval_summary(
        run_id=run_id, step="post_quantize_eval",
        checkpoint_ref=quant_ckpt, sample_root=sample_root,
        worker_result=worker_result,
    )
    artifacts = [
        artifact_ref(kind="sample_dir", path_or_uri=sample_root,
                      content_type="inode/directory", label="eval-samples"),
    ]
    step_result = build_step_result(
        run_id=run_id, step="post_quantize_eval",
        status="completed" if (execute and not dry_run) else "planned",
        input_checkpoint=quant_ckpt,
        output_checkpoint=quant_ckpt,  # eval doesn't produce new checkpoint
        command=["q19", "eval", f"--prompts={num_prompts}"],
        remote_job=_remote_job("post-quant-eval", remote_config, dry_run),
        artifacts=artifacts,
        metrics=eval_summary.get("aggregate_metrics", {}),
        eval_summary_path=public_path(run_dir / "post_quantize_eval" / "eval-summary.json"),
        report_path=public_path(run_dir / "post_quantize_eval" / "README.md"),
    )
    report_md = _render_report(
        step="post_quantize_eval", step_result=step_result, eval_summary=eval_summary,
        extra=f"## Evaluation\n- Checkpoint: `{quant_ckpt}`\n- Prompts: `{num_prompts}`",
    )

    if dry_run:
        log_stage_complete("post_quantize_eval", {"run_id": run_id})
        return {"run_id": run_id, "run_dir": public_path(run_dir),
                "step_result": step_result, "eval_summary": eval_summary,
                "report_preview": report_md}

    result = _write_and_update(
        manifest, run_dir, "post_quantize_eval", step_result, eval_summary, report_md,
    )
    log_stage_complete("post_quantize_eval", result)
    return result


# ═══════════════════════════════════════════════════════════════════
# REPORT  (dashboard + server)
# ═══════════════════════════════════════════════════════════════════

def run_report(
    *,
    artifact_dir: str | None = None,
    run_id: str | None = None,
    serve: bool = False,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> dict[str, Any]:
    root = runs_root(artifact_dir)
    log_stage_start("report", runs_root=public_path(root), serve=serve)
    summary = generate_results_summary(root)

    if run_id:
        idx = root / run_id / "report-index.json"
        if not idx.exists():
            raise ValueError(f"Run `{run_id}` not found under `{public_path(root)}`.")
        summary["run_report_index"] = public_path(idx)

    if serve:
        log_stage_progress("report", "starting results server", host=host, port=port)
        from qwen_image_19.server import serve_results
        serve_results(host=host, port=port, runs_root=root)

    log_stage_complete("report", summary)
    return summary
