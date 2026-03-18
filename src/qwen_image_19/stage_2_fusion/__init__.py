from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from qwen_image_19.config_io import load_json_yaml, repo_root, write_json, write_text
from qwen_image_19.remote import default_remote_context
from qwen_image_19.stage_1_analysis import load_model_inventory


def build_edit_delta_recipe(models: dict[str, dict[str, Any]], config_dir: Path) -> dict[str, Any]:
    recipe = load_json_yaml(config_dir / "stage-2-delta-edit.yaml")
    return {
        "name": recipe["recipe_name"],
        "foundation": models[recipe["foundation_model"]]["model_id"],
        "delta_source": models[recipe["delta_source"]]["model_id"],
        "delta_base_hint": recipe["delta_base_hint"],
        "blend_weight": recipe["default_blend_weight"],
        "target_subsystems": recipe["target_subsystems"],
    }


def build_text_encoder_merge_recipe(models: dict[str, dict[str, Any]], config_dir: Path) -> dict[str, Any]:
    recipe = load_json_yaml(config_dir / "stage-2-ties-text-encoder.yaml")
    return {
        "name": recipe["recipe_name"],
        "foundation": models[recipe["foundation_model"]]["model_id"],
        "donor": models[recipe["donor_model"]]["model_id"],
        "strategy": recipe["strategy"],
        "conflict_policy": recipe["conflict_policy"],
        "target_subsystems": recipe["target_subsystems"],
    }


def build_fusion_manifest(
    matrix: dict[str, Any],
    models: dict[str, dict[str, Any]],
    remote_context: dict[str, Any],
    config_dir: Path,
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stage": "stage2",
        "foundation_model": models["qwen-image-2512"]["model_id"],
        "source_models": [payload["model_id"] for payload in models.values() if payload["alias"] != "qwen-image-base"],
        "compatibility_summary": matrix["summary"],
        "recipes": {
            "edit_delta": build_edit_delta_recipe(models, config_dir),
            "text_encoder_merge": build_text_encoder_merge_recipe(models, config_dir),
        },
        "remote": {
            "workdir": remote_context["workdir"],
            "artifact_dir": remote_context["artifact_dir"],
            "output_checkpoint": f"{remote_context['artifact_dir']}/stage-2/qwen-image-1.9-bf16.safetensors",
        },
    }


def render_fusion_report(manifest: dict[str, Any]) -> str:
    edit = manifest["recipes"]["edit_delta"]
    text = manifest["recipes"]["text_encoder_merge"]
    return f"""# Stage 2 Fusion Report

## Mission
Produce a BF16 research artifact that keeps 2512 as the realism anchor, injects edit behavior from 2511, and borrows layered prompt logic without pretending RGBA is just a mood.

## Edit delta recipe
- Foundation: `{edit['foundation']}`
- Delta source: `{edit['delta_source']}`
- Delta base hint: `{edit['delta_base_hint']}`
- Blend weight: `{edit['blend_weight']}`
- Target subsystems: {', '.join(edit['target_subsystems'])}

## Text encoder merge recipe
- Foundation: `{text['foundation']}`
- Donor: `{text['donor']}`
- Strategy: `{text['strategy']}`
- Conflict policy: `{text['conflict_policy']}`

## Remote artifact target
- Output checkpoint: `{manifest['remote']['output_checkpoint']}`

## Unresolved conflicts
- RGBA-VAE semantics remain out of line with the RGB foundation.
- Layer-specific RoPE must stay behind a compatibility gate until Stage 1 evidence improves.
"""


def fuse(
    artifact_dir: str | Path | None = None,
    remote_config: str | None = None,
    cache_dir: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    models = load_model_inventory()
    stage1_report = repo_root() / "reports" / "stage-1-compatibility-matrix.json"
    if stage1_report.exists():
        matrix = json.loads(stage1_report.read_text(encoding="utf-8"))
    else:
        matrix = {
            "summary": {
                "direct_merge": 0,
                "delta_merge": 1,
                "adapter_only": 2,
                "incompatible": 1,
            }
        }
    remote_context = default_remote_context(remote_config)
    if cache_dir:
        remote_context["cache_dir"] = cache_dir
    config_dir = repo_root() / "configs" / "merge"
    manifest = build_fusion_manifest(matrix, models, remote_context, config_dir)
    report = render_fusion_report(manifest)
    target_dir = Path(artifact_dir) if artifact_dir else repo_root() / "reports"
    result = {
        "stage": "stage2",
        "mode": "dry-run" if dry_run else "write",
        "manifest": manifest,
        "report_preview": report if dry_run else report.splitlines()[:8],
        "artifact_dir": str(target_dir),
    }
    if dry_run:
        return result
    write_json(target_dir / "stage-2-merge-manifest.json", manifest)
    write_text(target_dir / "stage-2-fusion-report.md", report)
    result["written"] = [
        str(target_dir / "stage-2-merge-manifest.json"),
        str(target_dir / "stage-2-fusion-report.md"),
    ]
    return result
