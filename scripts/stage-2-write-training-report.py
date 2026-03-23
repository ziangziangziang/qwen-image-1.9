#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional at runtime
    plt = None

_WORKFLOW_DEFAULTS = "core-delta,layered-bridge,experimental"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a curated Stage 2 training report (real runs only)."
    )
    parser.add_argument("--run-status", default="stage-2/run-status.json")
    parser.add_argument(
        "--merge-manifest",
        default="reports/stage-2-merge-manifest.json",
        help="Path to merge manifest JSON (flat reports/ path).",
    )
    parser.add_argument(
        "--dataset-manifest",
        default="reports/stage-2/dataset-manifest.json",
    )
    parser.add_argument("--output-md", default="reports/stage-2/training-report.md")
    parser.add_argument("--figures-dir", default="reports/stage-2/figures")
    parser.add_argument(
        "--workflows",
        default=_WORKFLOW_DEFAULTS,
        help="Comma-separated workflow names. Each resolves to "
             "stage-2/metrics/{name}-train.json and stage-2/evals/{name}/.",
    )
    parser.add_argument(
        "--metrics",
        default=None,
        help="Explicit path to a metrics JSON file. "
             "When provided, also resolved via --workflows; this flag is accepted "
             "for compatibility with the stage-2 launcher.",
    )
    parser.add_argument(
        "--mode",
        choices=["smoke", "real", "auto"],
        default="auto",
        help=(
            "smoke: skip all file output and exit 0. "
            "real: always generate the full report. "
            "auto (default): detect from run_profile in run-status.json."
        ),
    )
    return parser.parse_args()


def read_json(path: Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    if not path.exists():
        return default or {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default or {}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def is_smoke_run(run_status: dict[str, Any], mode: str) -> bool:
    if mode == "smoke":
        return True
    if mode == "real":
        return False
    # auto: inspect run_profile value
    profile = str(run_status.get("run_profile", "")).lower()
    return "smoke" in profile


def render_loss_curve(
    metrics: dict[str, Any], output_path: Path, workflow: str
) -> str | None:
    loss_curve = metrics.get("loss_curve") or []
    if not loss_curve or plt is None:
        return None
    ensure_dir(output_path.parent)
    plt.figure(figsize=(8, 4.5))
    plt.plot(list(range(1, len(loss_curve) + 1)), loss_curve, linewidth=1.8)
    plt.title(f"{workflow} — Training Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path.as_posix()


def render_loss_table(metrics: dict[str, Any]) -> str:
    loss_curve = metrics.get("loss_curve") or []
    if not loss_curve:
        return "_No loss curve data available._"

    def fmt(v: Any) -> str:
        try:
            return f"{float(v):.6f}"
        except (TypeError, ValueError):
            return str(v)

    rows = ["| Step | Loss |", "| ---: | ---: |"]
    total = len(loss_curve)
    if total <= 10:
        for i, v in enumerate(loss_curve, 1):
            rows.append(f"| {i} | {fmt(v)} |")
    else:
        for i, v in enumerate(loss_curve[:5], 1):
            rows.append(f"| {i} | {fmt(v)} |")
        rows.append(f"| … | _(steps 6–{total - 5} omitted)_ |")
        for i, v in zip(range(total - 4, total + 1), loss_curve[-5:]):
            rows.append(f"| {i} | {fmt(v)} |")
    return "\n".join(rows)


def copy_if_exists(src: Path, dst: Path) -> str | None:
    if not src.exists():
        return None
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)
    return dst.as_posix()


def first_file_in(directory: Path, suffix: str = ".png") -> Path | None:
    if not directory.is_dir():
        return None
    candidates = sorted(p for p in directory.iterdir() if p.suffix == suffix)
    return candidates[0] if candidates else None


def format_job_row(name: str, payload: dict[str, Any]) -> str:
    log = payload.get("stdout_stderr_log", payload.get("log_path", "n/a"))
    return (
        f"| `{name}` | `{payload.get('status', 'unknown')}` "
        f"| `{payload.get('duration_seconds', 'n/a')}` "
        f"| `{payload.get('exit_code', 'n/a')}` | `{log}` |"
    )


def render_hardware_block(hw_data: dict[str, Any]) -> str:
    cuda = hw_data.get("cuda", {}) if isinstance(hw_data, dict) else {}
    devices = cuda.get("devices", []) if isinstance(cuda, dict) else []
    lines = [
        f"- Hostname: `{hw_data.get('hostname', 'unknown')}`",
        f"- Platform: `{hw_data.get('platform', 'unknown')}`",
        f"- Python: `{hw_data.get('python', 'unknown')}`",
        f"- CPU: `{hw_data.get('cpu', 'unknown')}`",
        f"- Logical cores: `{hw_data.get('logical_cores', 'unknown')}`",
        f"- CUDA available: `{cuda.get('available', False)}`",
        f"- GPU count: `{cuda.get('device_count', 0)}`",
        f"- Selected device: `{cuda.get('selected_device', 'n/a')}`",
    ]
    for dev in devices:
        lines.append(
            f"- GPU `cuda:{dev.get('index')}`: `{dev.get('name')}` "
            f"({dev.get('total_memory_gb_decimal', 'n/a')} GB, "
            f"SMs={dev.get('multiprocessors', 'n/a')}, "
            f"cc={dev.get('compute_capability', 'n/a')})"
        )
    return "\n".join(lines)


def render_structure_mermaid(metrics: dict[str, Any], workflow: str) -> str:
    structure = metrics.get("structure", {})
    layers = structure.get("layers", []) if isinstance(structure, dict) else []
    if not layers:
        wf_lower = workflow.lower()
        if "layered" in wf_lower or "bridge" in wf_lower:
            layers = [
                "Input RGBA",
                "Channel Splitter",
                "Bridge Adapter",
                "RGB Projection",
                "Output RGB",
            ]
        elif "edit" in wf_lower or "delta" in wf_lower:
            layers = ["Base Model (2512)", "Edit Delta", "Merged Output"]
        else:
            layers = ["Input", "Merge Layer", "Output"]
    nodes = []
    edges = []
    for idx, layer in enumerate(layers):
        node_id = f"L{idx}"
        nodes.append(f'    {node_id}["{layer}"]')
        if idx > 0:
            edges.append(f"    L{idx - 1} --> {node_id}")
    return "\n".join(["```mermaid", "flowchart LR", *nodes, *edges, "```"])


def _job_entry_for_workflow(workflow: str, jobs: dict[str, Any]) -> dict[str, Any]:
    """Look up the job dict entry for a workflow using several key-name conventions."""
    candidates = [
        workflow.replace("-", "_"),
        workflow.replace("-", "_") + "_train",
        workflow,
    ]
    for key in candidates:
        entry = jobs.get(key, {})
        if isinstance(entry, dict) and entry:
            return entry
    return {}


def render_workflow_section(
    workflow: str,
    metrics: dict[str, Any],
    jobs: dict[str, Any],
    figures_dir: Path,
) -> tuple[str, str | None]:
    """Return (markdown_section, loss_png_path_or_None)."""
    safe = workflow.replace("/", "-").replace(" ", "-")

    loss_png = render_loss_curve(metrics, figures_dir / f"{safe}-loss.png", workflow)

    # Before: baseline eval sample; After: real-samples subdir (dedicated post-merge run)
    before_src = first_file_in(Path(f"stage-2/evals/{workflow}/samples"))
    after_src = first_file_in(Path(f"stage-2/evals/{workflow}/real-samples"))

    before_dst = (
        copy_if_exists(before_src, figures_dir / f"{safe}-before-001.png")
        if before_src
        else None
    )
    after_dst = (
        copy_if_exists(after_src, figures_dir / f"{safe}-after-001.png")
        if after_src
        else None
    )

    method = metrics.get("training_method", {}) if isinstance(metrics, dict) else {}
    job_entry = _job_entry_for_workflow(workflow, jobs)
    job_status = job_entry.get("status", metrics.get("status", "unknown"))
    elapsed_raw = metrics.get("elapsed_seconds")

    def _fmt_elapsed(v: Any) -> str:
        try:
            secs = float(v)
            h, rem = divmod(int(secs), 3600)
            m, s = divmod(rem, 60)
            return f"{secs:.1f}s ({h:02d}h {m:02d}m {s:02d}s)"
        except (TypeError, ValueError):
            return "n/a"

    lines = [
        f"## Workflow: {workflow}",
        "",
        "### Training Method",
        f"- Type: `{method.get('type', 'unknown')}`",
        f"- Model: `{method.get('model', 'unknown')}`",
        f"- Objective: `{method.get('objective', 'unknown')}`",
        f"- Optimizer: `{method.get('optimizer', 'unknown')}`",
        f"- Notes: {method.get('notes', 'n/a')}",
        "",
        "### Hyperparameters",
        f"- Max steps: `{metrics.get('max_steps', 'n/a')}`",
        f"- Batch size: `{metrics.get('batch_size', 'n/a')}`",
        f"- Learning rate: `{metrics.get('learning_rate', 'n/a')}`",
        f"- Seed: `{metrics.get('seed', 'n/a')}`",
        "",
        "### Timing",
        f"- Start: `{metrics.get('run_started_at', 'n/a')}`",
        f"- End: `{metrics.get('run_ended_at', 'n/a')}`",
        f"- Elapsed: `{_fmt_elapsed(elapsed_raw)}`",
        f"- Job status: `{job_status}`",
        "",
        "### Loss",
        f"- Final: `{metrics.get('final_loss', 'n/a')}`",
        f"- Min: `{metrics.get('min_loss', 'n/a')}`",
        f"- Max: `{metrics.get('max_loss', 'n/a')}`",
        "",
        render_loss_table(metrics),
        "",
    ]

    if loss_png:
        lines.append(f"![{workflow} training loss](figures/{safe}-loss.png)")
    else:
        lines.append(
            "_Loss curve figure unavailable (matplotlib not installed or `loss_curve` "
            "absent in metrics)._"
        )

    lines += [
        "",
        "### Structure Visualization",
        "",
        render_structure_mermaid(metrics, workflow),
        "",
        "### Visual Outcomes (Before / After Merge)",
        "",
    ]

    if before_dst:
        lines.append(
            f"**Before** (baseline eval sample — `stage-2/evals/{workflow}/samples/`):"
        )
        lines.append(f"![{workflow} before merge](figures/{safe}-before-001.png)")
    else:
        lines.append(
            f"_Before sample not available at `stage-2/evals/{workflow}/samples/`._"
        )

    lines.append("")

    if after_dst:
        lines.append(
            f"**After** (real eval sample — `stage-2/evals/{workflow}/real-samples/`):"
        )
        lines.append(f"![{workflow} after merge](figures/{safe}-after-001.png)")
    else:
        lines.append(
            f"_After sample not yet available at `stage-2/evals/{workflow}/real-samples/`._"
        )

    lines.append("")
    return "\n".join(lines), loss_png


def main() -> int:
    args = parse_args()
    run_status_path = Path(args.run_status)
    merge_manifest_path = Path(args.merge_manifest)
    dataset_manifest_path = Path(args.dataset_manifest)
    output_md_path = Path(args.output_md)
    figures_dir = Path(args.figures_dir)
    workflows = [w.strip() for w in args.workflows.split(",") if w.strip()]

    run_status = read_json(run_status_path)

    if is_smoke_run(run_status, args.mode):
        print(
            json.dumps(
                {
                    "status": "skipped",
                    "reason": "smoke mode — no files written to reports/",
                    "run_profile": run_status.get("run_profile", "unknown"),
                }
            )
        )
        return 0

    merge_manifest = read_json(merge_manifest_path)
    dataset_manifest = read_json(dataset_manifest_path)
    _ = dataset_manifest  # consumed for completeness; referenced in Artifact References

    ensure_dir(figures_dir)

    run_profile = run_status.get(
        "run_profile", merge_manifest.get("run_profile", "unknown")
    )
    git_commit = run_status.get("git_commit", "n/a")
    jobs: dict[str, Any] = (
        run_status.get("jobs", {}) if isinstance(run_status, dict) else {}
    )

    # Per-workflow rendering
    workflow_sections: list[str] = []
    workflow_coverage_rows: list[str] = []
    all_hw: list[dict[str, Any]] = []
    total_elapsed = 0.0
    has_elapsed = False

    for wf in workflows:
        metrics_path = Path(f"stage-2/metrics/{wf}-train.json")
        metrics = read_json(metrics_path)

        if not metrics and not metrics_path.exists():
            workflow_sections.append(
                f"## Workflow: {wf}\n\n"
                f"_Metrics file not found at `{metrics_path.as_posix()}`. "
                "This workflow has not been executed yet._\n"
            )
            workflow_coverage_rows.append(
                f"| `{wf}` | `not executed` "
                f"| `{metrics_path.as_posix()}` | — |"
            )
            continue

        hw = metrics.get("hardware", {})
        if isinstance(hw, dict) and hw:
            all_hw.append(hw)

        try:
            total_elapsed += float(metrics.get("elapsed_seconds", 0) or 0)
            has_elapsed = True
        except (TypeError, ValueError):
            pass

        section, _ = render_workflow_section(wf, metrics, jobs, figures_dir)
        workflow_sections.append(section)

        job_entry = _job_entry_for_workflow(wf, jobs)
        status_val = job_entry.get("status", metrics.get("status", "unknown"))
        duration_val = job_entry.get(
            "duration_seconds", metrics.get("elapsed_seconds", "n/a")
        )
        workflow_coverage_rows.append(
            f"| `{wf}` | `{status_val}` "
            f"| `{metrics_path.as_posix()}` | `{duration_val}` |"
        )

    # Aggregate hardware: first available hw block (all workflows share the same node)
    hw_data: dict[str, Any] = all_hw[0] if all_hw else {}

    # Job summary table
    job_rows = (
        "\n".join(
            format_job_row(name, payload)
            for name, payload in jobs.items()
            if isinstance(payload, dict)
        )
        or "| `n/a` | `n/a` | `n/a` | `n/a` | `n/a` |"
    )

    if has_elapsed:
        h, rem = divmod(int(total_elapsed), 3600)
        m, s = divmod(rem, 60)
        elapsed_summary = f"{total_elapsed:.1f}s ({h:02d}h {m:02d}m {s:02d}s)"
    else:
        elapsed_summary = "n/a"

    coverage_str = (
        "\n".join(workflow_coverage_rows)
        if workflow_coverage_rows
        else "| — | — | — | — |"
    )

    sections_str = "\n".join(workflow_sections)

    output_md = f"""# Stage 2 Real Training Report

## Run Profile

- Run profile: `{run_profile}`
- Git commit: `{git_commit}`
- Stage: `stage-2`
- Workflows documented: {", ".join(f"`{w}`" for w in workflows)}
- Report generated from: `{run_status_path.as_posix()}`, `{merge_manifest_path.as_posix()}`

## Hardware

{render_hardware_block(hw_data) if hw_data else "_Hardware metadata not present in any workflow metrics file._"}

## Aggregate Timing

- Total elapsed across all workflows: `{elapsed_summary}`

## Runtime / Job Summary

| Job | Status | Duration (s) | Exit code | Log |
| --- | --- | --- | --- | --- |
{job_rows}

---

{sections_str}
---

## Workflow Coverage

| Workflow | Status | Metrics source | Duration (s) |
| --- | --- | --- | --- |
{coverage_str}

## Artifact References

- Merge manifest: `{merge_manifest_path.as_posix()}`
- Dataset manifest: `{dataset_manifest_path.as_posix()}`
- Run status: `{run_status_path.as_posix()}`
- Figures: `{figures_dir.as_posix()}/`
"""

    ensure_dir(output_md_path.parent)
    output_md_path.write_text(output_md.rstrip() + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "ok",
                "output_md": output_md_path.as_posix(),
                "figures_dir": figures_dir.as_posix(),
                "workflows": workflows,
                "total_elapsed_seconds": total_elapsed if has_elapsed else None,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
