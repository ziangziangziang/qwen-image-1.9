from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from qwen_image_19.config_io import repo_root, write_json, write_text
from qwen_image_19.contracts import PIPELINE_STEPS, public_path


def build_report_index(run_manifest: dict[str, Any]) -> dict[str, Any]:
    steps = {}
    for step in PIPELINE_STEPS:
        record = run_manifest["steps"].get(step)
        if record is None:
            continue
        steps[step] = {
            "status": record["status"],
            "step_result": record.get("step_result"),
            "eval_summary": record.get("eval_summary"),
            "report": record.get("report"),
            "output_checkpoint": record.get("output_checkpoint"),
            "metrics": record.get("metrics", {}),
        }
    return {
        "run_id": run_manifest["run_id"],
        "manifest": public_path(Path(run_manifest["artifact_root"]) / "manifest.json"),
        "steps": steps,
        "updated_at": run_manifest["updated_at"],
        "tags": run_manifest.get("tags", []),
        "notes": run_manifest.get("notes", ""),
    }


def write_report_index(run_dir: Path, run_manifest: dict[str, Any]) -> Path:
    return write_json(run_dir / "report-index.json", build_report_index(run_manifest))


def collect_runs(runs_root: Path | None = None) -> list[dict[str, Any]]:
    root = runs_root or repo_root() / "reports" / "runs"
    if not root.exists():
        return []
    runs = []
    for manifest_path in sorted(root.glob("*/manifest.json")):
        try:
            runs.append(json.loads(manifest_path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return sorted(runs, key=lambda item: item.get("updated_at", ""), reverse=True)


def write_dashboard_fixture(runs_root: Path | None = None) -> Path:
    root = runs_root or repo_root() / "reports" / "runs"
    payload = {
        "runs": [
            {
                "run_id": manifest["run_id"],
                "updated_at": manifest["updated_at"],
                "steps": {
                    step: manifest["steps"][step]["status"]
                    for step in PIPELINE_STEPS
                    if step in manifest.get("steps", {})
                },
                "report_index": manifest["report_index"],
            }
            for manifest in collect_runs(root)
        ]
    }
    return write_json(root / "dashboard-index.json", payload)


def render_report_overview(runs: list[dict[str, Any]]) -> str:
    if not runs:
        return "# Results Dashboard\n\n_No runs found._\n"
    rows = []
    for run in runs:
        status = ", ".join(
            f"{step}:{run['steps'].get(step, {}).get('status', 'n/a')}"
            for step in PIPELINE_STEPS
        )
        rows.append(f"| `{run['run_id']}` | `{run['updated_at']}` | `{status}` | `{run['report_index']}` |")
    body = "\n".join(rows)
    return f"""# Results Dashboard

| Run | Updated | Step Status | Report Index |
| --- | --- | --- | --- |
{body}
"""


def generate_results_summary(runs_root: Path | None = None) -> dict[str, Any]:
    root = runs_root or repo_root() / "reports" / "runs"
    runs = collect_runs(root)
    dashboard_index = write_dashboard_fixture(root)
    dashboard_md = write_text(root / "README.md", render_report_overview(runs))
    return {
        "runs_root": public_path(root),
        "dashboard_index": public_path(dashboard_index),
        "dashboard_readme": public_path(dashboard_md),
        "run_count": len(runs),
    }
