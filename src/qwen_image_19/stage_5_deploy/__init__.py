from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from qwen_image_19.config_io import load_json_yaml, repo_root, write_text
from qwen_image_19.remote import default_remote_context


REQUIRED_STAGE_CONFIG_KEYS = {"version", "runtime", "stages", "outputs"}


def load_stage_template(template_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(template_path) if template_path else repo_root() / "configs" / "deploy" / "stage-5-vllm-omni-stage-config.yaml"
    return load_json_yaml(path)


def generate_stage_config(template: dict[str, Any], remote_context: dict[str, Any]) -> dict[str, Any]:
    generated = dict(template)
    generated["remote"] = {
        "workdir": remote_context["workdir"],
        "artifact_dir": remote_context["artifact_dir"],
        "cache_dir": remote_context["cache_dir"],
    }
    generated["outputs"] = dict(template["outputs"])
    generated["outputs"]["artifact_manifest"] = f"{remote_context['artifact_dir']}/stage-5/generated-stage-config.yaml"
    return generated


def validate_stage_config(config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = REQUIRED_STAGE_CONFIG_KEYS.difference(config.keys())
    if missing:
        errors.append(f"missing keys: {sorted(missing)}")
    if not isinstance(config.get("stages", []), list) or not config.get("stages"):
        errors.append("stages must be a non-empty list")
    return errors


def render_deployment_report(config: dict[str, Any]) -> str:
    stages = "\n".join(f"- `{stage['name']}`: {stage['kind']} -> {stage['artifact']}" for stage in config["stages"])
    return f"""# Stage 5 Deployment Report

## Runtime
- Backend: `{config['runtime']['backend']}`
- Accelerator: `{config['runtime']['accelerator']}`

## Stage graph
{stages}

## Remote output
- Generated config: `{config['outputs']['artifact_manifest']}`

## Known limitations
- This scaffold validates structure and orchestration only.
- Benchmark fields should be appended after remote execution.
"""


def deploy(
    artifact_dir: str | Path | None = None,
    remote_config: str | None = None,
    cache_dir: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    template = load_stage_template()
    remote_context = default_remote_context(remote_config)
    if cache_dir:
        remote_context["cache_dir"] = cache_dir
    config = generate_stage_config(template, remote_context)
    errors = validate_stage_config(config)
    if errors:
        raise ValueError("; ".join(errors))
    report = render_deployment_report(config)
    target_dir = Path(artifact_dir) if artifact_dir else repo_root() / "reports"
    result = {
        "stage": "stage5",
        "mode": "dry-run" if dry_run else "write",
        "config": config,
        "artifact_dir": str(target_dir),
    }
    if dry_run:
        result["report_preview"] = report
        return result
    write_text(target_dir / "stage-5-generated-stage-config.yaml", json.dumps(config, indent=2))
    write_text(target_dir / "stage-5-deployment-report.md", report)
    result["written"] = [
        str(target_dir / "stage-5-generated-stage-config.yaml"),
        str(target_dir / "stage-5-deployment-report.md"),
    ]
    return result
