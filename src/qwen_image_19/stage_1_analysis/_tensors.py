from __future__ import annotations

import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

from qwen_image_19.stage_1_analysis._hardware import (
    MODEL_ORDER,
    SUBSYSTEM_ORDER,
    WEIGHT_ANALYSIS_PAIRWISE,
    Stage1AnalysisError,
    _safe_round,
)
from qwen_image_19.stage_1_analysis._cache import (
    analyze_vae_compatibility,
    build_layer_pairwise,
    build_pairwise_comparisons,
    build_subsystem_result,
    compare_tensor_sets,
    normalize_layer_descriptor,
    probe_rope_compatibility,
    summarize_layer_inventory,
    summarize_manifest,
    subsystem_for_tensor,
)


WEIGHT_DELTA_THRESHOLD = 1e-6


def require_weight_analysis_runtime() -> tuple[Any, Any]:
    try:
        import torch
        from safetensors import safe_open
    except ModuleNotFoundError as exc:
        raise Stage1AnalysisError(
            "Stage 1 value-level weight analysis requires both `torch` and `safetensors`. "
            "Install project dependencies on the remote machine before running q19 stage1 analyze."
        ) from exc
    return torch, safe_open


def compute_tensor_value_metrics(left_tensor: Any, right_tensor: Any, torch_module: Any) -> dict[str, Any]:
    left_fp = left_tensor.detach().to(dtype=torch_module.float64).reshape(-1).cpu()
    right_fp = right_tensor.detach().to(dtype=torch_module.float64).reshape(-1).cpu()
    delta = left_fp - right_fp
    left_norm = float(torch_module.linalg.vector_norm(left_fp).item())
    right_norm = float(torch_module.linalg.vector_norm(right_fp).item())
    l2_norm_delta = float(torch_module.linalg.vector_norm(delta).item())
    abs_delta = delta.abs()
    mean_absolute_delta = float(abs_delta.mean().item()) if abs_delta.numel() else 0.0
    max_absolute_delta = float(abs_delta.max().item()) if abs_delta.numel() else 0.0
    denominator = max(left_norm, right_norm, 1e-12)
    relative_l2_delta = l2_norm_delta / denominator
    return {
        "exact_equal": bool(torch_module.equal(left_tensor, right_tensor)),
        "l2_norm_delta": round(l2_norm_delta, 10),
        "mean_absolute_delta": round(mean_absolute_delta, 10),
        "max_absolute_delta": round(max_absolute_delta, 10),
        "relative_l2_delta": round(relative_l2_delta, 10),
        "left_l2_norm": round(left_norm, 10),
        "right_l2_norm": round(right_norm, 10),
    }


def load_pair_weight_tensor_metrics(
    left_manifest: dict[str, Any],
    right_manifest: dict[str, Any],
    torch_module: Any,
    safe_open_fn: Any,
) -> dict[str, Any]:
    left_tensors = left_manifest["tensors"]
    right_tensors = right_manifest["tensors"]
    left_keys = set(left_tensors)
    right_keys = set(right_tensors)
    shared_keys = sorted(left_keys & right_keys)
    comparable_keys = [
        key
        for key in shared_keys
        if left_tensors[key]["shape"] == right_tensors[key]["shape"]
        and left_tensors[key]["dtype"] == right_tensors[key]["dtype"]
    ]
    shape_mismatch_excluded = [
        key for key in shared_keys if left_tensors[key]["shape"] != right_tensors[key]["shape"]
    ]
    dtype_mismatch_excluded = [
        key
        for key in shared_keys
        if left_tensors[key]["shape"] == right_tensors[key]["shape"]
        and left_tensors[key]["dtype"] != right_tensors[key]["dtype"]
    ]
    comparable_left_bytes = sum(int(left_tensors[key]["payload_nbytes"]) for key in comparable_keys)
    comparable_right_bytes = sum(int(right_tensors[key]["payload_nbytes"]) for key in comparable_keys)

    grouped_keys: dict[tuple[str, str], list[str]] = {}
    for key in comparable_keys:
        left_relative = str(left_tensors[key]["relative_path"])
        right_relative = str(right_tensors[key]["relative_path"])
        grouped_keys.setdefault((left_relative, right_relative), []).append(key)

    tensor_metrics: dict[str, Any] = {}
    for (left_relative, right_relative), keys in grouped_keys.items():
        left_path = Path(left_manifest["snapshot_path"]) / left_relative
        right_path = Path(right_manifest["snapshot_path"]) / right_relative
        with safe_open_fn(str(left_path), framework="pt", device="cpu") as left_handle:
            with safe_open_fn(str(right_path), framework="pt", device="cpu") as right_handle:
                for key in keys:
                    left_tensor = left_handle.get_tensor(str(left_tensors[key]["raw_name"]))
                    right_tensor = right_handle.get_tensor(str(right_tensors[key]["raw_name"]))
                    descriptor = normalize_layer_descriptor(key, left_tensors[key])
                    tensor_metrics[key] = {
                        "tensor_key": key,
                        "layer_id": descriptor["layer_id"],
                        "subsystem": descriptor["subsystem"],
                        "family": descriptor["family"],
                        "parameter_suffix": descriptor["parameter_suffix"],
                        **compute_tensor_value_metrics(left_tensor, right_tensor, torch_module),
                    }

    return {
        "left_tensor_count": len(left_keys),
        "right_tensor_count": len(right_keys),
        "shared_key_count": len(shared_keys),
        "missing_key_excluded_count": len(left_keys - right_keys) + len(right_keys - left_keys),
        "shape_mismatch_excluded_count": len(shape_mismatch_excluded),
        "dtype_mismatch_excluded_count": len(dtype_mismatch_excluded),
        "comparable_left_bytes": comparable_left_bytes,
        "comparable_right_bytes": comparable_right_bytes,
        "comparable_total_bytes": comparable_left_bytes + comparable_right_bytes,
        "tensor_metrics": tensor_metrics,
    }


def summarize_weight_pair(
    pair_name: str,
    models: tuple[str, str],
    tensor_metrics: dict[str, Any],
    left_tensor_count: int,
    right_tensor_count: int,
    shared_key_count: int,
    missing_key_excluded_count: int,
    shape_mismatch_excluded_count: int,
    dtype_mismatch_excluded_count: int,
    comparable_left_bytes: int,
    comparable_right_bytes: int,
    comparable_total_bytes: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    exact_count = sum(1 for item in tensor_metrics.values() if item["exact_equal"])
    low_delta_count = sum(
        1 for item in tensor_metrics.values() if item["relative_l2_delta"] <= WEIGHT_DELTA_THRESHOLD
    )
    comparable_count = len(tensor_metrics)
    relative_values = [item["relative_l2_delta"] for item in tensor_metrics.values()]
    mean_abs_values = [item["mean_absolute_delta"] for item in tensor_metrics.values()]

    layer_buckets: dict[str, dict[str, Any]] = {}
    for key, item in tensor_metrics.items():
        layer = layer_buckets.setdefault(
            item["layer_id"],
            {
                "layer_id": item["layer_id"],
                "subsystem": item["subsystem"],
                "family": item["family"],
                "comparable_tensor_count": 0,
                "exact_equal_tensor_count": 0,
                "low_delta_tensor_count": 0,
                "relative_l2_delta_sum": 0.0,
                "mean_absolute_delta_sum": 0.0,
                "max_mean_absolute_delta": 0.0,
                "l2_norm_delta_sq_sum": 0.0,
                "left_l2_norm_sq_sum": 0.0,
                "right_l2_norm_sq_sum": 0.0,
                "top_divergent_tensors": [],
            },
        )
        layer["comparable_tensor_count"] += 1
        layer["exact_equal_tensor_count"] += int(item["exact_equal"])
        layer["low_delta_tensor_count"] += int(item["relative_l2_delta"] <= WEIGHT_DELTA_THRESHOLD)
        layer["relative_l2_delta_sum"] += float(item["relative_l2_delta"])
        layer["mean_absolute_delta_sum"] += float(item["mean_absolute_delta"])
        layer["max_mean_absolute_delta"] = max(
            layer["max_mean_absolute_delta"], float(item["mean_absolute_delta"])
        )
        layer["l2_norm_delta_sq_sum"] += float(item["l2_norm_delta"]) ** 2
        layer["left_l2_norm_sq_sum"] += float(item["left_l2_norm"]) ** 2
        layer["right_l2_norm_sq_sum"] += float(item["right_l2_norm"]) ** 2
        layer["top_divergent_tensors"].append(
            {
                "tensor_key": key,
                "parameter_suffix": item["parameter_suffix"],
                "relative_l2_delta": item["relative_l2_delta"],
                "mean_absolute_delta": item["mean_absolute_delta"],
                "max_absolute_delta": item["max_absolute_delta"],
                "exact_equal": item["exact_equal"],
            }
        )

    summarized_layers: dict[str, Any] = {}
    for layer_id, layer in layer_buckets.items():
        comparable = max(layer["comparable_tensor_count"], 1)
        aggregate_denominator = max(
            math.sqrt(layer["left_l2_norm_sq_sum"]),
            math.sqrt(layer["right_l2_norm_sq_sum"]),
            1e-12,
        )
        relative_l2_delta = math.sqrt(layer["l2_norm_delta_sq_sum"]) / aggregate_denominator
        top_divergent_tensors = sorted(
            layer["top_divergent_tensors"],
            key=lambda item: (-item["relative_l2_delta"], -item["mean_absolute_delta"], item["tensor_key"]),
        )[:5]
        summarized_layers[layer_id] = {
            "layer_id": layer_id,
            "subsystem": layer["subsystem"],
            "family": layer["family"],
            "comparable_tensor_count": layer["comparable_tensor_count"],
            "exact_equal_tensor_count": layer["exact_equal_tensor_count"],
            "exact_tensor_match_ratio": round(layer["exact_equal_tensor_count"] / comparable, 4),
            "low_delta_tensor_count": layer["low_delta_tensor_count"],
            "low_delta_tensor_ratio": round(layer["low_delta_tensor_count"] / comparable, 4),
            "mean_relative_l2_delta": round(layer["relative_l2_delta_sum"] / comparable, 10),
            "relative_l2_delta": round(relative_l2_delta, 10),
            "mean_absolute_delta_mean": round(layer["mean_absolute_delta_sum"] / comparable, 10),
            "max_mean_absolute_delta": round(layer["max_mean_absolute_delta"], 10),
            "layer_weight_similarity_score": round(max(0.0, 1.0 - min(relative_l2_delta, 1.0)), 4),
            "top_divergent_tensors": top_divergent_tensors,
        }

    sorted_layers = sorted(
        summarized_layers.values(),
        key=lambda item: (item["layer_weight_similarity_score"], -item["relative_l2_delta"], item["layer_id"]),
    )
    top_divergent_tensors = sorted(
        tensor_metrics.values(),
        key=lambda item: (-item["relative_l2_delta"], -item["mean_absolute_delta"], item["tensor_key"]),
    )[:10]
    top_divergent_blocks = [
        {
            "layer_id": item["layer_id"],
            "subsystem": item["subsystem"],
            "family": item["family"],
            "relative_l2_delta": item["relative_l2_delta"],
            "exact_tensor_match_ratio": item["exact_tensor_match_ratio"],
            "low_delta_tensor_ratio": item["low_delta_tensor_ratio"],
            "comparable_tensor_count": item["comparable_tensor_count"],
        }
        for item in sorted_layers[:10]
    ]
    by_block = {item["layer_id"]: item for item in sorted_layers}
    by_subsystem: dict[str, dict[str, Any]] = {}
    for subsystem in SUBSYSTEM_ORDER:
        subset = [item for item in sorted_layers if item["subsystem"] == subsystem]
        if not subset:
            continue
        by_subsystem[subsystem] = {
            "block_count": len(subset),
            "mean_exact_tensor_match_ratio": round(
                sum(item["exact_tensor_match_ratio"] for item in subset) / len(subset), 4
            ),
            "mean_low_delta_tensor_ratio": round(
                sum(item["low_delta_tensor_ratio"] for item in subset) / len(subset), 4
            ),
            "mean_block_relative_l2_delta": round(
                sum(item["relative_l2_delta"] for item in subset) / len(subset), 10
            ),
            "worst_blocks": [
                {
                    "layer_id": item["layer_id"],
                    "relative_l2_delta": item["relative_l2_delta"],
                    "exact_tensor_match_ratio": item["exact_tensor_match_ratio"],
                }
                for item in sorted(
                    subset,
                    key=lambda value: (
                        value["layer_weight_similarity_score"],
                        -value["relative_l2_delta"],
                        value["layer_id"],
                    ),
                )[:5]
            ],
        }
    summary = {
        "pair_name": pair_name,
        "models": list(models),
        "left_tensor_count": left_tensor_count,
        "right_tensor_count": right_tensor_count,
        "shared_key_count": shared_key_count,
        "missing_key_excluded_count": missing_key_excluded_count,
        "shape_mismatch_excluded_count": shape_mismatch_excluded_count,
        "dtype_mismatch_excluded_count": dtype_mismatch_excluded_count,
        "comparable_left_bytes": int(comparable_left_bytes),
        "comparable_right_bytes": int(comparable_right_bytes),
        "comparable_total_bytes": int(comparable_total_bytes),
        "exclusion_accounting": {
            "missing_keys": missing_key_excluded_count,
            "shape_mismatch": shape_mismatch_excluded_count,
            "dtype_mismatch": dtype_mismatch_excluded_count,
        },
        "comparable_tensor_count": comparable_count,
        "exact_equal_tensor_count": exact_count,
        "exact_equal_tensor_ratio": round(exact_count / max(comparable_count, 1), 4),
        "low_delta_tensor_count": low_delta_count,
        "low_delta_tensor_ratio": round(low_delta_count / max(comparable_count, 1), 4),
        "mean_relative_l2_delta": round(sum(relative_values) / max(comparable_count, 1), 10),
        "max_relative_l2_delta": round(max(relative_values) if relative_values else 0.0, 10),
        "mean_mean_absolute_delta": round(sum(mean_abs_values) / max(comparable_count, 1), 10),
        "top_divergent_blocks": top_divergent_blocks,
        "top_divergent_layers": top_divergent_blocks,
        "top_divergent_tensors": [
            {
                "tensor_key": item["tensor_key"],
                "layer_id": item["layer_id"],
                "parameter_suffix": item["parameter_suffix"],
                "relative_l2_delta": item["relative_l2_delta"],
                "mean_absolute_delta": item["mean_absolute_delta"],
                "max_absolute_delta": item["max_absolute_delta"],
                "exact_equal": item["exact_equal"],
            }
            for item in top_divergent_tensors
        ],
        "by_block": by_block,
        "layers": by_block,
        "by_subsystem": by_subsystem,
    }
    details = {
        **summary,
        "tensor_metrics": dict(sorted(tensor_metrics.items())),
    }
    return summary, details


def build_weight_pairwise_analysis(
    manifests: dict[str, dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    torch_module, safe_open_fn = require_weight_analysis_runtime()
    pairwise_summary: dict[str, Any] = {}
    pairwise_details: dict[str, Any] = {}
    pair_timings: dict[str, float] = {}
    for pair_name, models in WEIGHT_ANALYSIS_PAIRWISE.items():
        pair_start = time.perf_counter()
        left_manifest = manifests[models[0]]
        right_manifest = manifests[models[1]]
        raw_metrics = load_pair_weight_tensor_metrics(left_manifest, right_manifest, torch_module, safe_open_fn)
        summary, details = summarize_weight_pair(
            pair_name=pair_name,
            models=models,
            tensor_metrics=raw_metrics["tensor_metrics"],
            left_tensor_count=raw_metrics["left_tensor_count"],
            right_tensor_count=raw_metrics["right_tensor_count"],
            shared_key_count=raw_metrics["shared_key_count"],
            missing_key_excluded_count=raw_metrics["missing_key_excluded_count"],
            shape_mismatch_excluded_count=raw_metrics["shape_mismatch_excluded_count"],
            dtype_mismatch_excluded_count=raw_metrics["dtype_mismatch_excluded_count"],
            comparable_left_bytes=raw_metrics["comparable_left_bytes"],
            comparable_right_bytes=raw_metrics["comparable_right_bytes"],
            comparable_total_bytes=raw_metrics["comparable_total_bytes"],
        )
        pair_timings[pair_name] = _safe_round(time.perf_counter() - pair_start, 4)
        pairwise_summary[pair_name] = summary
        pairwise_details[pair_name] = details
    runtime_profile = {
        "pair_value_pass_seconds": pair_timings,
    }
    return pairwise_summary, pairwise_details, runtime_profile


def build_block_review_summary(weight_pairwise: dict[str, Any]) -> dict[str, Any]:
    pair_summary: dict[str, Any] = {}
    subsystem_totals: dict[str, dict[str, Any]] = {}
    for pair_name, payload in weight_pairwise.items():
        pair_summary[pair_name] = {
            "models": payload["models"],
            "comparable_tensor_count": payload["comparable_tensor_count"],
            "exact_equal_tensor_ratio": payload["exact_equal_tensor_ratio"],
            "low_delta_tensor_ratio": payload["low_delta_tensor_ratio"],
            "mean_relative_l2_delta": payload["mean_relative_l2_delta"],
            "mean_block_similarity_score": round(
                sum(item["layer_weight_similarity_score"] for item in payload["by_block"].values())
                / max(len(payload["by_block"]), 1),
                4,
            ),
        }
        for subsystem, subsystem_payload in payload["by_subsystem"].items():
            stats = subsystem_totals.setdefault(
                subsystem,
                {"pair_count": 0, "exact_sum": 0.0, "low_delta_sum": 0.0, "relative_l2_sum": 0.0},
            )
            stats["pair_count"] += 1
            stats["exact_sum"] += subsystem_payload["mean_exact_tensor_match_ratio"]
            stats["low_delta_sum"] += subsystem_payload["mean_low_delta_tensor_ratio"]
            stats["relative_l2_sum"] += subsystem_payload["mean_block_relative_l2_delta"]
    subsystem_summary = {
        subsystem: {
            "pair_count": values["pair_count"],
            "mean_exact_tensor_match_ratio": round(values["exact_sum"] / values["pair_count"], 4),
            "mean_low_delta_tensor_ratio": round(values["low_delta_sum"] / values["pair_count"], 4),
            "mean_block_relative_l2_delta": round(values["relative_l2_sum"] / values["pair_count"], 10),
        }
        for subsystem, values in subsystem_totals.items()
    }
    return {"pairs": pair_summary, "subsystems": subsystem_summary}


def build_compatibility_matrix(manifests: dict[str, dict[str, Any]]) -> dict[str, Any]:
    from qwen_image_19.stage_1_analysis._report import render_similarity_visualization
    from datetime import datetime, timezone

    pairwise = build_pairwise_comparisons(manifests)
    layer_pairwise = build_layer_pairwise(manifests)
    mmdit_stats = compare_tensor_sets(
        manifests["qwen-image-2512"]["tensors"],
        manifests["qwen-image-edit-2511"]["tensors"],
        predicate=lambda key: subsystem_for_tensor(
            key,
            manifests["qwen-image-2512"]["tensors"].get(
                key, manifests["qwen-image-edit-2511"]["tensors"].get(key, {})
            ),
        )
        == "mmdit_backbone",
    )
    text_stats = compare_tensor_sets(
        manifests["qwen-image-base"]["tensors"],
        manifests["qwen-image-layered"]["tensors"],
        predicate=lambda key: subsystem_for_tensor(
            key,
            manifests["qwen-image-base"]["tensors"].get(
                key, manifests["qwen-image-layered"]["tensors"].get(key, {})
            ),
        )
        == "text_encoder",
    )
    vae_result = analyze_vae_compatibility(manifests)
    rope_result = probe_rope_compatibility(manifests)

    subsystems = [
        build_subsystem_result(
            subsystem="mmdit_backbone",
            models=["qwen-image-2512", "qwen-image-edit-2511"],
            stats=mmdit_stats,
            preferred_mode="delta-merge",
            reason="Use real shared-key and shape stats between 2512 and 2511 to justify a delta merge path without pretending the strategy is the same thing as structural parity.",
        ),
        build_subsystem_result(
            subsystem="text_encoder",
            models=["qwen-image-base", "qwen-image-layered", "qwen-image-2512"],
            stats=text_stats,
            preferred_mode="adapter-only",
            reason="Layered is compared against its ancestry base first, then mapped onto the 2512 foundation as adapter-only logic unless exact parity is proven.",
        ),
        vae_result,
        rope_result,
    ]
    strategy_summary = Counter(item["recommended_merge_strategy"] for item in subsystems)
    structural_summary = Counter(item["structural_compatibility"] for item in subsystems)
    matrix = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inspection_mode": "hf-cache-real-checkpoint",
        "model_summaries": {alias: summarize_manifest(manifest) for alias, manifest in manifests.items()},
        "pairwise_comparisons": pairwise,
        "subsystems": subsystems,
        "summary": {
            "direct_merge": strategy_summary.get("direct-merge", 0),
            "delta_merge": strategy_summary.get("delta-merge", 0),
            "adapter_only": strategy_summary.get("adapter-only", 0),
            "incompatible": strategy_summary.get("incompatible", 0),
        },
        "structural_summary": {
            "direct_merge": structural_summary.get("direct-merge", 0),
            "adapter_only": structural_summary.get("adapter-only", 0),
            "incompatible": structural_summary.get("incompatible", 0),
        },
        "layer_inventory": {
            alias: summarize_layer_inventory(manifest["layer_inventory"]) for alias, manifest in manifests.items()
        },
        "layer_pairwise": layer_pairwise,
    }
    matrix["visualization"] = render_similarity_visualization(matrix)
    return matrix


def build_layer_analysis_payload(
    manifests: dict[str, dict[str, Any]],
    matrix: dict[str, Any],
) -> dict[str, Any]:
    return {
        "generated_at": matrix["generated_at"],
        "inspection_mode": matrix["inspection_mode"],
        "models": {
            alias: manifests[alias]["layer_inventory"] for alias in MODEL_ORDER
        },
        "pairs": matrix["layer_pairwise"],
    }
