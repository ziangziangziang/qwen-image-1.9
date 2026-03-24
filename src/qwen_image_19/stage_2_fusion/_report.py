from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from qwen_image_19.config_io import repo_root
from qwen_image_19.stage_2_fusion._manifest import repo_relative_path


def render_candidate_rows(candidates: list[dict[str, Any]], selected_candidate_id: str) -> str:
    rows = []
    for candidate in candidates:
        selected = "selected" if candidate["candidate_id"] == selected_candidate_id else "candidate"
        rows.append(
            f"| `{candidate['candidate_id']}` | `{candidate['blend_weight']}` | `{selected}` | `{candidate['output_checkpoint']}` | `{candidate['smoke_report']}` |"
        )
    return "\n".join(rows)


def render_dataset_rows(dataset_manifest: dict[str, Any]) -> str:
    rows = []
    for split_name, payload in dataset_manifest["splits"].items():
        rows.append(
            f"| `{split_name}` | `{payload['teacher_model']}` | `{payload['task']}` | `{payload['planned_sample_count']}` | `{payload['asset_root']}` |"
        )
    return "\n".join(rows)


def render_job_rows(remote_jobs: dict[str, Any]) -> str:
    rows = []
    for name, payload in remote_jobs.items():
        rows.append(
            f"| `{name}` | `{payload['status']}` | `{payload['entrypoint']}` | `{payload['workdir']}` | `{payload['log_path']}` |"
        )
    return "\n".join(rows)


def render_artifact_rows(artifacts: dict[str, str]) -> str:
    return "\n".join(
        f"| `{name}` | `{path}` |" for name, path in artifacts.items()
    )


def _render_eval_gallery(
    eval_dir: Path,
    task_label: str,
    sample_subdir: str = "samples",
    readme_dir: Path | None = None,
) -> str:
    """Return a Markdown image gallery block for up to 3 sample images in an eval directory."""
    samples_path = eval_dir / sample_subdir
    if not samples_path.is_dir():
        return f"_No samples found at `{samples_path.as_posix()}`._"
    pngs = sorted(samples_path.glob("*.png"))[:3]
    if not pngs:
        return f"_No `.png` samples found in `{samples_path.as_posix()}`._"
    base = readme_dir if readme_dir is not None else repo_root()
    lines = []
    for png in pngs:
        try:
            rel = Path(os.path.relpath(png, base)).as_posix()
        except ValueError:
            rel = png.as_posix()
        lines.append(f"![{task_label} sample]({rel})")
    return "  ".join(lines)


def _render_edit_gallery(eval_dir: Path, readme_dir: Path | None = None) -> str:
    """Return a Markdown before/after table for edit eval pairs (up to 3 pairs)."""
    sample_dir = eval_dir / "edit-samples"
    if not sample_dir.is_dir():
        return f"_No edit samples found at `{sample_dir.as_posix()}`._"
    before_images = sorted(sample_dir.glob("edit-*-before.png"))[:3]
    if not before_images:
        return f"_No before/after pairs found in `{sample_dir.as_posix()}`._"
    base = readme_dir if readme_dir is not None else repo_root()
    rows = ["| Before | After |", "|--------|-------|"]
    for before_png in before_images:
        after_png = Path(str(before_png).replace("-before.png", "-after.png"))
        try:
            before_rel = Path(os.path.relpath(before_png, base)).as_posix()
        except ValueError:
            before_rel = before_png.as_posix()
        after_cell = ""
        if after_png.exists():
            try:
                after_rel = Path(os.path.relpath(after_png, base)).as_posix()
            except ValueError:
                after_rel = after_png.as_posix()
            after_cell = f"![after]({after_rel})"
        rows.append(f"| ![before]({before_rel}) | {after_cell} |")
    # Append structure-preservation SSIM summary if available
    summary_path = eval_dir / "edit-summary.json"
    footer = ""
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            ssim = summary.get("mean_ssim_score")
            if ssim is not None:
                footer = (
                    f"\n_Structure-preservation SSIM: `{ssim:.4f}` "
                    f"(1.0 = scene/character/pose identical; lower = structural change)_"
                )
        except (json.JSONDecodeError, KeyError):
            pass
    return "\n".join(rows) + footer


def _render_consistency_summary(consistency_summary_path: Path) -> str:
    if not consistency_summary_path.exists():
        return "_Consistency eval not yet run._"
    try:
        data = json.loads(consistency_summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "_Consistency eval summary is malformed._"
    c = data.get("consistency", {})
    mean_drift = c.get("mean_pixel_l2_drift", "n/a")
    max_drift = c.get("max_pixel_l2_drift", "n/a")
    min_drift = c.get("min_pixel_l2_drift", "n/a")
    n = data.get("num_prompts", "?")
    status = data.get("status", "?")
    return (
        f"- Prompts evaluated: `{n}`\n"
        f"- Mean pixel-L2 drift: `{mean_drift}`  _(lower = merged model stays close to foundation)_\n"
        f"- Min drift: `{min_drift}` / Max drift: `{max_drift}`\n"
        f"- Status: `{status}`"
    )


def _render_run_results_section(
    manifest: dict[str, Any],
    run_status: dict[str, Any] | None,
) -> str:
    """Render the post-execution Results section for the README hub.

    Only included after a real execution (run_status is not None and not empty).
    """
    if not run_status or not run_status.get("jobs"):
        return ""

    jobs: dict[str, Any] = run_status.get("jobs", {})

    # Job execution summary table
    job_rows = []
    total_duration = 0.0
    for name, job in jobs.items():
        if not isinstance(job, dict):
            continue
        status = job.get("status", "?")
        dur = job.get("duration_seconds", "—")
        exit_code = job.get("exit_code", "—")
        log = job.get("stdout_stderr_log", job.get("log_path", "—"))
        emoji = "✓" if status == "succeeded" else ("↩" if status == "skipped" else "✗")
        job_rows.append(f"| `{name}` | {emoji} `{status}` | `{dur}` | `{exit_code}` | `{log}` |")
        try:
            total_duration += float(dur)
        except (TypeError, ValueError):
            pass

    h, rem = divmod(int(total_duration), 3600)
    m, s = divmod(rem, 60)
    total_str = f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"
    job_table = "\n".join(job_rows)

    # --- eval galleries ---
    evals_root = repo_root() / "stage-2" / "evals"
    readme_dir = repo_root() / "reports" / "stage-2"

    generation_gallery = _render_eval_gallery(
        evals_root / "core-candidates" / manifest.get("selected_core_candidate", {}).get("candidate_id", "core-delta-w035"),
        "generation",
        readme_dir=readme_dir,
    )
    edit_gallery = _render_edit_gallery(evals_root / "core-edit", readme_dir=readme_dir)
    experimental_gallery = _render_eval_gallery(evals_root / "experimental", "experimental", "samples", readme_dir=readme_dir)
    consistency_block = _render_consistency_summary(
        evals_root / "consistency" / "consistency-summary.json"
    )

    # links to sub-reports
    training_report_path = repo_root() / manifest.get("artifacts", {}).get("training_report", "reports/stage-2/training-report.md")
    training_report_link = (
        f"[training-report.md]({repo_relative_path(training_report_path)})"
        if training_report_path.exists()
        else "_training-report.md not yet generated_"
    )

    return f"""
---

## Run Results

> Profile: `{run_status.get('run_profile', '?')}` · Policy: `{run_status.get('execution_policy', '?')}` · Total: `{total_str}`

### Job Execution

| Job | Status | Duration (s) | Exit | Log |
| --- | --- | --- | --- | --- |
{job_table}

### Training Report

{training_report_link}

### Visual Evaluation

#### Generation (core-delta merged model — text-to-image)

{generation_gallery}

#### Edit (before → after pairs)

{edit_gallery}

#### Experimental Layered Bridge (smoke eval)

{experimental_gallery}

### Consistency Eval

{consistency_block}
"""


def render_fusion_report(
    manifest: dict[str, Any],
    dataset_manifest: dict[str, Any],
    run_status: dict[str, Any] | None = None,
) -> str:
    core = manifest["core_delta_recipe"]
    selected = manifest["selected_core_candidate"]
    layered = manifest["layered_bridge_recipe"]
    evidence = manifest["stage1_evidence"]
    exclusions = manifest["exclusions"]
    return f"""# Stage 2 Fusion

## Mission
Stage 2 now builds two tracks from the Stage 1 evidence: a stable BF16 core based on `2512 + transformer-only edit delta`, and an experimental Layered bridge branch that learns RGB behavior from synthetic teacher data instead of pretending `vae` and `rope` mismatches will disappear.

## Run Mode
- Run mode: `{manifest['run_mode']}`
- Run profile: `{manifest['run_profile']}`
- Execution enabled: `{manifest['execution_enabled']}`
- Execution policy: `{manifest['execution_policy']}`
- Cleanup performed: `{manifest['cleanup_performed']}`
- Resource profile: `num_gpus={manifest['resource_profile'].get('num_gpus')}`, `vram_target_gb={manifest['resource_profile'].get('vram_target_gb')}`
- Limits: `{json.dumps(manifest['limits'], sort_keys=True)}`

## Stage 1 Evidence
- Foundation vs Edit transformer path is the one clean merge lane: `shared={evidence['foundation_vs_edit']['shared_key_count']}`, `exact={evidence['foundation_vs_edit']['exact_equal_tensor_ratio']}`, `strategy={evidence['foundation_vs_edit']['transformer_merge_strategy']}`.
- Edit deltas cluster in late MMDiT blocks: `{', '.join(str(value) for value in evidence['foundation_vs_edit']['observed_hot_blocks']) or 'none captured'}`.
- Layered text encoder is a no-op donor in practice: `exact={evidence['layered_vs_core']['text_encoder_exact_match']}`.
- Layered conflicts remain real: VAE `{evidence['layered_conflicts']['vae']['base_label']} -> {evidence['layered_conflicts']['vae']['layered_label']}`, rope `{evidence['layered_conflicts']['rope']['foundation_label']} -> {evidence['layered_conflicts']['rope']['layered_label']}`.

## No-Go List
- `text_encoder`: {exclusions['text_encoder']['reason']}
- `vae`: {exclusions['vae']['reason']}
- `rope`: {exclusions['rope']['reason']}

## Stable Core Track
- Foundation: `{core['foundation_model']}`
- Delta source: `{core['delta_source_model']}`
- Delta base candidate: `{core['delta_base_candidate_model']}`
- Target scope: `{', '.join(core['target_components'])}` / `{', '.join(core['target_subsystems'])}`
- Selection rule: `{core['selection_rule']}`

| Candidate | Blend weight | Status | Planned checkpoint | Planned smoke report |
| --- | --- | --- | --- | --- |
{render_candidate_rows(manifest['core_delta_candidates'], selected['candidate_id'])}

## Experimental Layered Bridge Track
- Donor: `{layered['donor_model']}`
- Strategy: `{layered['strategy']}`
- Base core candidate: `{layered['base_core_candidate_id']}`
- Bridge scope: `transformer_blocks.{layered['bridge_block_window']['start']}:{layered['bridge_block_window']['end']}`
- Extra parameter paths: {', '.join(layered['extra_parameter_paths'])}
- Trainable modules: {', '.join(layered['trainable_modules'])}
- Freeze policy: {', '.join(layered['freeze_policy'])}
- Distillation target: `{layered['distillation_target']}`
- Output adapter: `{layered['output_adapter']}`
- Output checkpoint: `{layered['output_checkpoint']}`

## Teacher Dataset
- Dataset manifest: `{manifest['dataset']['manifest_path']}`
- Output root: `{dataset_manifest['output_root']}`
- Layered flattening: `{dataset_manifest['flatten_layered_rgba']}`

| Split | Teacher model | Task | Planned samples | Asset root |
| --- | --- | --- | --- | --- |
{render_dataset_rows(dataset_manifest)}

## Remote Jobs
| Job | Status | Entry point | Workdir | Log |
| --- | --- | --- | --- | --- |
{render_job_rows(manifest['remote_jobs'])}

## Artifacts
| Artifact | Path |
| --- | --- |
{render_artifact_rows(manifest['artifacts'])}

## Limitations
- Stage 2 does not attempt true RGBA decomposition support. Layered supervision is flattened back into RGB composites.
- The stable core winner is provisional until the remote coefficient sweep and smoke suite complete.
- The Layered branch is experimental and should be treated as a bridge adapter, not a drop-in replacement for the core checkpoint.
{_render_run_results_section(manifest, run_status)}"""


