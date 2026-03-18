from __future__ import annotations

from pathlib import Path
from typing import Any

from qwen_image_19.config_io import repo_root, write_text
from qwen_image_19.remote import default_remote_context


def build_eval_registry() -> dict[str, Any]:
    return {
        "capability_suites": [
            "text-to-image fidelity",
            "image editing consistency",
            "multi-subject composition",
            "layered decomposition reliability",
        ],
        "safety_suites": [
            "policy boundary prompts",
            "misuse-risk sampling",
            "failure-case review",
            "release-note gating review",
        ],
    }


def render_eval_report(registry: dict[str, Any], remote_context: dict[str, Any]) -> str:
    caps = "\n".join(f"- {item}" for item in registry["capability_suites"])
    safety = "\n".join(f"- {item}" for item in registry["safety_suites"])
    return f"""# Stage 3 Evaluation Report

## Scope
This stage replaces safeguard-bypass ambitions with capability measurement, misuse-risk documentation, and release gating.

## Remote context
- Artifact dir: `{remote_context['artifact_dir']}`
- Workdir: `{remote_context['workdir']}`

## Capability suites
{caps}

## Safety suites
{safety}

## Decision gate
Do not cut a public release candidate unless the failure cases and operating constraints are written down like adults.
"""


def evaluate(
    artifact_dir: str | Path | None = None,
    remote_config: str | None = None,
    cache_dir: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    registry = build_eval_registry()
    remote_context = default_remote_context(remote_config)
    if cache_dir:
        remote_context["cache_dir"] = cache_dir
    report = render_eval_report(registry, remote_context)
    target_dir = Path(artifact_dir) if artifact_dir else repo_root() / "reports"
    result = {
        "stage": "stage3",
        "mode": "dry-run" if dry_run else "write",
        "registry": registry,
        "artifact_dir": str(target_dir),
    }
    if dry_run:
        result["report_preview"] = report
        return result
    write_text(target_dir / "stage-3-eval-report.md", report)
    result["written"] = [str(target_dir / "stage-3-eval-report.md")]
    return result
