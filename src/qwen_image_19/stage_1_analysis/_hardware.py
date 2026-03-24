from __future__ import annotations

import os
import platform
import shutil
import socket
from pathlib import Path
from typing import Any


MODEL_ORDER = (
    "qwen-image-base",
    "qwen-image-2512",
    "qwen-image-edit-2511",
    "qwen-image-layered",
)

SUBSYSTEM_ORDER = (
    "mmdit_backbone",
    "text_encoder",
    "vae",
    "rope",
)

WEIGHT_ANALYSIS_PAIRWISE = {
    "foundation_vs_edit": ("qwen-image-2512", "qwen-image-edit-2511"),
    "base_vs_layered": ("qwen-image-base", "qwen-image-layered"),
    "foundation_vs_layered": ("qwen-image-2512", "qwen-image-layered"),
}

ESTIMATE_READ_GBPS = {"low": 0.9, "typical": 1.8, "high": 3.2}
ESTIMATE_REDUCE_GBPS = {"low": 7.0, "typical": 14.0, "high": 28.0}


class Stage1AnalysisError(RuntimeError):
    """Raised when Stage 1 cannot inspect the remote cache layout."""


def _safe_round(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def bytes_to_gib(value: int | float) -> float:
    return round(float(value) / (1024**3), 4)


def format_bytes(value: int | float | None) -> str:
    if value is None:
        return "unknown"
    value = float(value)
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    unit_idx = 0
    while value >= 1024.0 and unit_idx < len(units) - 1:
        value /= 1024.0
        unit_idx += 1
    return f"{value:.2f} {units[unit_idx]}"


def detect_gpu_presence() -> dict[str, Any]:
    nvidia_smi = shutil.which("nvidia-smi")
    rocm_smi = shutil.which("rocm-smi")
    if nvidia_smi:
        return {
            "detected": True,
            "vendor_hint": "nvidia",
            "probe": "nvidia-smi",
            "probe_path": nvidia_smi,
        }
    if rocm_smi:
        return {
            "detected": True,
            "vendor_hint": "amd",
            "probe": "rocm-smi",
            "probe_path": rocm_smi,
        }
    return {
        "detected": False,
        "vendor_hint": "none",
        "probe": "path-scan",
        "probe_path": None,
    }


def detect_cpu_model() -> str:
    processor = platform.processor().strip()
    if processor:
        return processor
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        try:
            for line in cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
                if "model name" in line:
                    _, value = line.split(":", 1)
                    return value.strip()
        except OSError:
            pass
    return "unknown"


def detect_total_ram_bytes() -> int | None:
    if hasattr(os, "sysconf"):
        try:
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            pages = int(os.sysconf("SC_PHYS_PAGES"))
            total = page_size * pages
            if total > 0:
                return total
        except (TypeError, ValueError, OSError):
            return None
    return None


def build_hardware_snapshot(
    hf_home: Path,
    artifact_dir: Path,
    snapshot_inventory: dict[str, dict[str, str]],
) -> dict[str, Any]:
    total_ram = detect_total_ram_bytes()
    return {
        "hostname": socket.gethostname(),
        "os": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_model": detect_cpu_model(),
        "logical_cores": os.cpu_count(),
        "total_ram_bytes": total_ram,
        "total_ram_gib": bytes_to_gib(total_ram) if total_ram is not None else None,
        "gpu_probe": detect_gpu_presence(),
        "gpu_used": False,
        "hf_home": str(hf_home),
        "artifact_dir": str(artifact_dir),
        "snapshot_paths": {
            alias: item["snapshot_path"] for alias, item in snapshot_inventory.items()
        },
    }


def summarize_workload(
    manifests: dict[str, dict[str, Any]],
    weight_pairwise: dict[str, Any],
) -> dict[str, Any]:
    model_tensor_bytes = {
        alias: int(manifests[alias]["total_tensor_bytes"]) for alias in MODEL_ORDER
    }
    roadmap_pairs: dict[str, Any] = {}
    for pair_name, payload in weight_pairwise.items():
        roadmap_pairs[pair_name] = {
            "models": list(payload["models"]),
            "comparable_tensor_count": int(payload["comparable_tensor_count"]),
            "comparable_left_bytes": int(payload.get("comparable_left_bytes", 0)),
            "comparable_right_bytes": int(payload.get("comparable_right_bytes", 0)),
            "comparable_total_bytes": int(payload.get("comparable_total_bytes", 0)),
            "excluded_counts": {
                "missing": int(payload["exclusion_accounting"]["missing_keys"]),
                "shape_mismatch": int(payload["exclusion_accounting"]["shape_mismatch"]),
                "dtype_mismatch": int(payload["exclusion_accounting"]["dtype_mismatch"]),
            },
        }
    total_value_bytes = sum(item["comparable_total_bytes"] for item in roadmap_pairs.values())
    return {
        "model_tensor_bytes": model_tensor_bytes,
        "model_tensor_gib": {alias: bytes_to_gib(value) for alias, value in model_tensor_bytes.items()},
        "roadmap_pairs": roadmap_pairs,
        "value_analysis_total_bytes": int(total_value_bytes),
        "value_analysis_total_gib": bytes_to_gib(total_value_bytes),
    }


def estimate_runtime_seconds_for_bytes(byte_count: int, read_gbps: float, reduce_gbps: float) -> float:
    read_seconds = float(byte_count) / max(read_gbps * 1_000_000_000, 1e-9)
    reduce_seconds = float(byte_count) / max(reduce_gbps * 1_000_000_000, 1e-9)
    return read_seconds + reduce_seconds


def build_runtime_estimate(workload: dict[str, Any], observed_total_seconds: float) -> dict[str, Any]:
    pair_estimates: dict[str, Any] = {}
    totals = {"low": 0.0, "typical": 0.0, "high": 0.0}
    for pair_name, payload in workload["roadmap_pairs"].items():
        byte_count = int(payload["comparable_total_bytes"])
        low_seconds = estimate_runtime_seconds_for_bytes(
            byte_count,
            read_gbps=ESTIMATE_READ_GBPS["high"],
            reduce_gbps=ESTIMATE_REDUCE_GBPS["high"],
        )
        typical_seconds = estimate_runtime_seconds_for_bytes(
            byte_count,
            read_gbps=ESTIMATE_READ_GBPS["typical"],
            reduce_gbps=ESTIMATE_REDUCE_GBPS["typical"],
        )
        high_seconds = estimate_runtime_seconds_for_bytes(
            byte_count,
            read_gbps=ESTIMATE_READ_GBPS["low"],
            reduce_gbps=ESTIMATE_REDUCE_GBPS["low"],
        )
        pair_estimates[pair_name] = {
            "low_seconds": _safe_round(low_seconds),
            "typical_seconds": _safe_round(typical_seconds),
            "high_seconds": _safe_round(high_seconds),
        }
        totals["low"] += low_seconds
        totals["typical"] += typical_seconds
        totals["high"] += high_seconds
    return {
        "assumptions_gbps": {
            "read": dict(ESTIMATE_READ_GBPS),
            "reduction": dict(ESTIMATE_REDUCE_GBPS),
        },
        "pair_seconds": pair_estimates,
        "total_seconds": {
            "low": _safe_round(totals["low"]),
            "typical": _safe_round(totals["typical"]),
            "high": _safe_round(totals["high"]),
        },
        "observed_total_seconds": _safe_round(observed_total_seconds),
    }


def build_phase_timing(phase_seconds: dict[str, float]) -> dict[str, Any]:
    total_wall = sum(float(value) for key, value in phase_seconds.items() if key != "total_wall")
    if "total_wall" in phase_seconds:
        total_wall = float(phase_seconds["total_wall"])
    phase_percentages = {}
    for name, seconds in phase_seconds.items():
        if name == "total_wall":
            continue
        ratio = float(seconds) / total_wall if total_wall > 0 else 0.0
        phase_percentages[name] = _safe_round(ratio * 100.0)
    return {
        "phases_seconds": {name: _safe_round(value) for name, value in phase_seconds.items() if name != "total_wall"},
        "phase_percentages": phase_percentages,
        "total_wall_seconds": _safe_round(total_wall),
    }


def build_resource_accounting(
    hardware_snapshot: dict[str, Any],
    phase_seconds: dict[str, float],
    manifests: dict[str, dict[str, Any]],
    weight_pairwise: dict[str, Any],
    value_runtime_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    timing = build_phase_timing(phase_seconds)
    if value_runtime_profile:
        timing["value_pair_seconds"] = value_runtime_profile.get("pair_value_pass_seconds", {})
    workload = summarize_workload(manifests, weight_pairwise)
    estimate = build_runtime_estimate(workload, timing["total_wall_seconds"])
    return {
        "hardware": hardware_snapshot,
        "timing": timing,
        "workload": workload,
        "estimate": estimate,
    }
