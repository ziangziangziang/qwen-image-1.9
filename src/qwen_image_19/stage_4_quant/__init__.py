from __future__ import annotations

from pathlib import Path
from typing import Any

from qwen_image_19.config_io import load_json_yaml, repo_root, write_text
from qwen_image_19.remote import default_remote_context


def load_quant_profiles(config_dir: str | Path | None = None) -> dict[str, dict[str, Any]]:
    root = Path(config_dir) if config_dir else repo_root() / "configs" / "quant"
    return {
        "gguf": load_json_yaml(root / "stage-4-gguf-imatrix.yaml"),
        "exl2-gptq": load_json_yaml(root / "stage-4-exl2-gptq.yaml"),
    }


def validate_quant_profiles(profiles: dict[str, dict[str, Any]]) -> None:
    gguf = profiles["gguf"]
    if "imatrix" not in gguf or not gguf["imatrix"].get("required"):
        raise ValueError("GGUF profile must declare required imatrix metadata.")
    if gguf["imatrix"].get("minimum_images", 0) < 1:
        raise ValueError("GGUF profile must specify a positive imatrix image count.")


def render_quantization_report(profiles: dict[str, dict[str, Any]], remote_context: dict[str, Any]) -> str:
    gguf = profiles["gguf"]
    exl2 = profiles["exl2-gptq"]
    return f"""# Stage 4 Quantization Report

## Purpose
Compress the BF16 artifact without casually deleting vision quality.

## GGUF path
- Targets: {', '.join(gguf['targets'])}
- IMatrix dataset: `{gguf['imatrix']['dataset_name']}`
- Minimum images: `{gguf['imatrix']['minimum_images']}`

## EXL2/GPTQ path
- Targets: {', '.join(exl2['targets'])}
- Runtime: `{exl2['runtime']['primary']}` on `{exl2['runtime']['hardware']}`

## Remote execution
- Artifact dir: `{remote_context['artifact_dir']}`
- Cache dir: `{remote_context['cache_dir']}`
"""


def quantize(
    artifact_dir: str | Path | None = None,
    remote_config: str | None = None,
    cache_dir: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    profiles = load_quant_profiles()
    validate_quant_profiles(profiles)
    remote_context = default_remote_context(remote_config)
    if cache_dir:
        remote_context["cache_dir"] = cache_dir
    report = render_quantization_report(profiles, remote_context)
    target_dir = Path(artifact_dir) if artifact_dir else repo_root() / "reports"
    result = {
        "stage": "stage4",
        "mode": "dry-run" if dry_run else "write",
        "profiles": profiles,
        "artifact_dir": str(target_dir),
    }
    if dry_run:
        result["report_preview"] = report
        return result
    write_text(target_dir / "stage-4-quantization-report.md", report)
    result["written"] = [str(target_dir / "stage-4-quantization-report.md")]
    return result
