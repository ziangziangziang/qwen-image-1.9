from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any

from qwen_image_19.stage_1_analysis._hardware import (
    MODEL_ORDER,
    SUBSYSTEM_ORDER,
    WEIGHT_ANALYSIS_PAIRWISE,
    Stage1AnalysisError,
    bytes_to_gib,
    format_bytes,
)
from qwen_image_19.stage_1_analysis._cache import (
    pair_alias,
    pretty_pair_label,
    short_model_label,
)


ROADMAP_LAYER_PAIRS = (
    ("qwen-image-2512", "qwen-image-edit-2511"),
    ("qwen-image-base", "qwen-image-layered"),
    ("qwen-image-2512", "qwen-image-layered"),
)


def render_similarity_visualization(matrix: dict[str, Any]) -> dict[str, str]:
    fve = matrix["pairwise_comparisons"]["foundation_vs_edit"]
    bvl = matrix["pairwise_comparisons"]["base_vs_layered"]
    subsystems = "\n".join(
        f"    S{idx}[\"{item['subsystem']}\\n{item['recommended_merge_strategy']}\"]"
        for idx, item in enumerate(matrix["subsystems"], start=1)
    )
    links = "\n".join(f"    C --> S{idx}" for idx, _ in enumerate(matrix["subsystems"], start=1))
    mermaid = f"""flowchart TD
    A["2512 vs 2511\\nshared: {fve['shared_key_count']}\\nmissing: {fve['missing_key_count']}\\nshape mismatches: {fve['shape_mismatch_count']}"]
    B["Base vs Layered\\nshared: {bvl['shared_key_count']}\\nmissing: {bvl['missing_key_count']}\\nshape mismatches: {bvl['shape_mismatch_count']}"]
    C["Recommended merge strategies"]
{subsystems}
    A --> C
    B --> C
{links}
"""
    return {
        "type": "mermaid",
        "title": "Stage 1 DNA evidence map",
        "source": mermaid,
    }


def stage1_artifact_paths(target_dir: Path) -> dict[str, Path]:
    return {
        "artifact_dir": target_dir,
        "summary_markdown": target_dir / "summary.md",
        "matrix_json": target_dir / "compatibility-matrix.json",
        "layer_analysis_json": target_dir / "layer-analysis.json",
        "weight_analysis_json": target_dir / "weight-analysis.json",
        "figures_dir": target_dir / "figures",
        "component_overview_png": target_dir / "figures" / "component-overview.png",
        "pairwise_comparison_png": target_dir / "figures" / "pairwise-comparison.png",
        "layer_sharing_heatmap_png": target_dir / "figures" / "layer-sharing-heatmap.png",
        "layer_sharing_bars_png": target_dir / "figures" / "layer-sharing-bars.png",
    }


def build_stage1_compatibility_shims(target_dir: Path) -> dict[str, Path]:
    if target_dir.name == "stage-1":
        compat_dir = target_dir.parent
        return {
            "legacy_matrix_json": compat_dir / "stage-1-compatibility-matrix.json",
            "legacy_report_md": compat_dir / "stage-1-dna-report.md",
        }
    return {}


def generate_stage1_figures(matrix: dict[str, Any], figure_paths: dict[str, Path]) -> dict[str, str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ModuleNotFoundError as exc:
        raise Stage1AnalysisError(
            "matplotlib is required for Stage 1 figures. Install project dependencies before running q19 stage1 analyze."
        ) from exc

    figure_paths["figures_dir"].mkdir(parents=True, exist_ok=True)

    model_names = list(matrix["model_summaries"].keys())
    component_names = sorted(
        {
            component
            for summary in matrix["model_summaries"].values()
            for component in summary["component_tensor_counts"]
        }
    )
    fig, ax = plt.subplots(figsize=(11, max(4, 1.4 * len(model_names))))
    left = [0 for _ in model_names]
    palette = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F", "#EDC948", "#B07AA1"]
    display_labels = [short_model_label(name) for name in model_names]
    for index, component in enumerate(component_names):
        values = [
            matrix["model_summaries"][model]["component_tensor_counts"].get(component, 0)
            for model in model_names
        ]
        ax.barh(display_labels, values, left=left, label=component, color=palette[index % len(palette)])
        left = [current + value for current, value in zip(left, values)]
    max_total = max(left or [1])
    for model_name, total in zip(display_labels, left):
        ax.text(total + max(1, int(max_total * 0.01)), model_name, str(total), va="center", fontsize=9)
    ax.set_title("Stage 1 Component Overview")
    ax.set_xlabel("Tensor count")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_paths["component_overview_png"], dpi=180)
    plt.close(fig)

    pair_names = list(matrix["pairwise_comparisons"].keys())
    metrics = [
        ("shared_key_count", "Shared keys", "#4E79A7"),
        ("missing_key_count", "Missing keys", "#F28E2B"),
        ("shape_mismatch_count", "Shape mismatches", "#E15759"),
    ]
    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.22
    base_positions = list(range(len(pair_names)))
    for metric_index, (metric_key, label, color) in enumerate(metrics):
        positions = [position + (metric_index - 1) * width for position in base_positions]
        values = [matrix["pairwise_comparisons"][name][metric_key] for name in pair_names]
        bars = ax.bar(positions, values, width=width, label=label, color=color)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value, str(value), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(base_positions)
    ax.set_xticklabels(pair_names, rotation=10, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("Stage 1 Pairwise Comparison")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(figure_paths["pairwise_comparison_png"], dpi=180)
    plt.close(fig)

    heatmap_metrics = [
        ("overall", "Overall layer sharing"),
        ("mmdit_backbone", "MMDiT layer sharing"),
        ("text_encoder", "Text encoder layer sharing"),
        ("vae", "VAE layer sharing"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for axis, (metric_key, title) in zip(axes.flatten(), heatmap_metrics):
        data = []
        for left_alias in model_names:
            row = []
            for right_alias in model_names:
                if left_alias == right_alias:
                    row.append(1.0)
                    continue
                pair_key = pair_alias(*sorted((left_alias, right_alias), key=MODEL_ORDER.index))
                pair_stats = matrix["layer_pairwise"][pair_key]
                if metric_key == "overall":
                    ratio = pair_stats["overall"]["layer_shared_ratio"]
                else:
                    ratio = pair_stats["by_subsystem"][metric_key]["layer_shared_ratio"]
                row.append(ratio)
            data.append(row)
        image = axis.imshow(data, vmin=0.0, vmax=1.0, cmap="YlGnBu")
        axis.set_xticks(range(len(model_names)))
        axis.set_xticklabels(display_labels, rotation=20, ha="right")
        axis.set_yticks(range(len(model_names)))
        axis.set_yticklabels(display_labels)
        axis.set_title(title)
        for y_index, row in enumerate(data):
            for x_index, value in enumerate(row):
                axis.text(x_index, y_index, f"{value:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(figure_paths["layer_sharing_heatmap_png"], dpi=180)
    plt.close(fig)

    layer_pair_keys = list(matrix["layer_pairwise"].keys())
    layer_labels = [pretty_pair_label(matrix["layer_pairwise"][key]["models"]) for key in layer_pair_keys]
    layer_metrics = [
        ("exact_layer_match_count", "Exact shared", "#4E79A7"),
        ("partial_layer_match_count", "Partial shared", "#F28E2B"),
        ("left_only_layer_count", "Left-only", "#E15759"),
        ("right_only_layer_count", "Right-only", "#76B7B2"),
    ]
    fig, ax = plt.subplots(figsize=(13, 6))
    width = 0.18
    base_positions = list(range(len(layer_pair_keys)))
    for metric_index, (metric_key, label, color) in enumerate(layer_metrics):
        positions = [position + (metric_index - 1.5) * width for position in base_positions]
        values = [matrix["layer_pairwise"][name]["overall"][metric_key] for name in layer_pair_keys]
        bars = ax.bar(positions, values, width=width, label=label, color=color)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value, str(value), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(base_positions)
    ax.set_xticklabels(layer_labels, rotation=15, ha="right")
    ax.set_ylabel("Layer count")
    ax.set_title("Stage 1 Layer Sharing Breakdown")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(figure_paths["layer_sharing_bars_png"], dpi=180)
    plt.close(fig)

    return {
        "component_overview": str(figure_paths["component_overview_png"].name),
        "pairwise_comparison": str(figure_paths["pairwise_comparison_png"].name),
        "layer_sharing_heatmap": str(figure_paths["layer_sharing_heatmap_png"].name),
        "layer_sharing_bars": str(figure_paths["layer_sharing_bars_png"].name),
    }


def render_layer_pair_summary_rows(layer_pairwise: dict[str, Any]) -> str:
    rows = []
    for left, right in combinations(MODEL_ORDER, 2):
        pair_key = pair_alias(left, right)
        pair_entry = layer_pairwise[pair_key]
        overall = pair_entry["overall"]
        rows.append(
            f"| `{pretty_pair_label(pair_entry['models'])}` | `{overall['shared_layer_count']}` | `{overall['exact_layer_match_count']}` | `{overall['partial_layer_match_count']}` | `{overall['left_only_layer_count']}` | `{overall['right_only_layer_count']}` | `{overall['shape_mismatched_layer_count']}` | `{overall['layer_shared_ratio']}` |"
        )
    return "\n".join(rows)


def render_subsystem_layer_tables(layer_pairwise: dict[str, Any]) -> str:
    sections: list[str] = []
    for subsystem in SUBSYSTEM_ORDER:
        rows = []
        for left, right in combinations(MODEL_ORDER, 2):
            pair_key = pair_alias(left, right)
            pair_entry = layer_pairwise[pair_key]
            stats = pair_entry["by_subsystem"][subsystem]
            rows.append(
                f"| `{pretty_pair_label(pair_entry['models'])}` | `{stats['shared_layer_count']}` | `{stats['exact_layer_match_count']}` | `{stats['partial_layer_match_count']}` | `{stats['shape_mismatched_layer_count']}` | `{stats['layer_shared_ratio']}` |"
            )
        sections.append(
            "\n".join(
                [
                    f"### {subsystem}",
                    "| Pair | Shared layers | Exact | Partial | Shape-mismatched layers | Shared ratio |",
                    "| --- | --- | --- | --- | --- | --- |",
                    *rows,
                ]
            )
        )
    return "\n\n".join(sections)


def render_top_divergent_layers(layer_pairwise: dict[str, Any]) -> str:
    sections: list[str] = []
    for left, right in ROADMAP_LAYER_PAIRS:
        pair_key = pair_alias(left, right)
        if pair_key not in layer_pairwise:
            continue
        pair_entry = layer_pairwise[pair_key]
        rows = []
        for item in pair_entry["overall"]["top_divergent_layers"][:5]:
            rows.append(
                f"| `{item['layer_id']}` | `{item['reason']}` | `{item['left_parameter_count']}` | `{item['right_parameter_count']}` | `{item['shape_mismatch_count']}` | `{', '.join(item['left_only_parameter_samples']) or 'none'}` | `{', '.join(item['right_only_parameter_samples']) or 'none'}` |"
            )
        if not rows:
            rows.append("| `none` | `no divergent layers captured` | `0` | `0` | `0` | `none` | `none` |")
        sections.append(
            "\n".join(
                [
                    f"### {pretty_pair_label(pair_entry['models'])}",
                    "| Layer | Reason | Left params | Right params | Shape mismatches | Left-only samples | Right-only samples |",
                    "| --- | --- | --- | --- | --- | --- | --- |",
                    *rows,
                ]
            )
        )
    return "\n\n".join(sections)


def render_layer_inventory_rows(matrix: dict[str, Any]) -> str:
    rows = []
    for alias in MODEL_ORDER:
        summary = matrix["model_summaries"][alias]
        subsystem_counts = ", ".join(
            f"{name}:{count}" for name, count in summary["layer_counts_by_subsystem"].items()
        ) or "none"
        rows.append(
            f"| `{alias}` | `{summary['normalized_layer_count']}` | `{subsystem_counts}` |"
        )
    return "\n".join(rows)


def render_weight_pair_summary_rows(weight_pairwise: dict[str, Any]) -> str:
    rows = []
    for pair_name in WEIGHT_ANALYSIS_PAIRWISE:
        pair = weight_pairwise[pair_name]
        rows.append(
            f"| `{pretty_pair_label(pair['models'])}` | `{pair['comparable_tensor_count']}` | `{pair['exact_equal_tensor_count']}` | `{pair['exact_equal_tensor_ratio']}` | `{pair['low_delta_tensor_ratio']}` | `{pair['mean_relative_l2_delta']}` | `{pair['max_relative_l2_delta']}` | `{pair['exclusion_accounting']['missing_keys']}` | `{pair['exclusion_accounting']['shape_mismatch']}` | `{pair['exclusion_accounting']['dtype_mismatch']}` |"
        )
    return "\n".join(rows)


def render_weight_layer_tables(weight_pairwise: dict[str, Any]) -> str:
    sections: list[str] = []
    for pair_name in WEIGHT_ANALYSIS_PAIRWISE:
        pair = weight_pairwise[pair_name]
        subsystem_sections = [f"### {pretty_pair_label(pair['models'])}"]
        for subsystem in SUBSYSTEM_ORDER:
            rows = []
            subsystem_layers = [
                item for item in pair["by_block"].values() if item["subsystem"] == subsystem
            ]
            subsystem_layers = sorted(
                subsystem_layers,
                key=lambda item: (-item["relative_l2_delta"], item["layer_id"]),
            )[:12]
            for layer in subsystem_layers:
                rows.append(
                    f"| `{layer['layer_id']}` | `{layer['comparable_tensor_count']}` | `{layer['exact_tensor_match_ratio']}` | `{layer['low_delta_tensor_ratio']}` | `{layer['relative_l2_delta']}` | `{layer['layer_weight_similarity_score']}` |"
                )
            subsystem_sections.extend(
                [
                    f"#### {subsystem}",
                    "| Block | Comparable tensors | Exact ratio | Low-delta ratio | Relative L2 delta | Similarity score |",
                    "| --- | --- | --- | --- | --- | --- |",
                    *(rows or ["| `none` | `0` | `0.0` | `0.0` | `0.0` | `0.0` |"]),
                ]
            )
        sections.append("\n".join(subsystem_sections))
    return "\n\n".join(sections)


def render_weight_top_divergences(weight_pairwise: dict[str, Any]) -> str:
    sections: list[str] = []
    for pair_name in WEIGHT_ANALYSIS_PAIRWISE:
        pair = weight_pairwise[pair_name]
        layer_rows = []
        for layer in pair["top_divergent_blocks"][:5]:
            layer_rows.append(
                f"| `{layer['layer_id']}` | `{layer['relative_l2_delta']}` | `{layer['exact_tensor_match_ratio']}` | `{layer['low_delta_tensor_ratio']}` | `{layer['comparable_tensor_count']}` | `{layer['subsystem']}` |"
            )
        tensor_rows = []
        for tensor in pair["top_divergent_tensors"][:5]:
            tensor_rows.append(
                f"| `{tensor['tensor_key']}` | `{tensor['layer_id']}` | `{tensor['relative_l2_delta']}` | `{tensor['mean_absolute_delta']}` | `{tensor['max_absolute_delta']}` |"
            )
        sections.append(
            "\n".join(
                [
                    f"### {pretty_pair_label(pair['models'])}",
                    "",
                    "| Divergent layer | Relative L2 delta | Exact ratio | Low-delta ratio | Comparable tensors |",
                    "| --- | --- | --- | --- | --- | --- |",
                    *(layer_rows or ["| `none` | `0.0` | `1.0` | `1.0` | `0` | `none` |"]),
                    "",
                    "| Divergent tensor | Layer | Relative L2 delta | Mean abs delta | Max abs delta |",
                    "| --- | --- | --- | --- |  --- |",
                    *(tensor_rows or ["| `none` | `none` | `0.0` | `0.0` | `0.0` |"]),
                ]
            )
        )
    return "\n\n".join(sections)


def render_block_review_summary_rows(matrix: dict[str, Any]) -> str:
    rows = []
    for pair_name in WEIGHT_ANALYSIS_PAIRWISE:
        pair = matrix["block_review_summary"]["pairs"][pair_name]
        rows.append(
            f"| `{pretty_pair_label(pair['models'])}` | `{pair['comparable_tensor_count']}` | `{pair['exact_equal_tensor_ratio']}` | `{pair['low_delta_tensor_ratio']}` | `{pair['mean_relative_l2_delta']}` | `{pair['mean_block_similarity_score']}` |"
        )
    return "\n".join(rows)


def render_hardware_account_rows(resource_accounting: dict[str, Any]) -> str:
    hardware = resource_accounting.get("hardware", {})
    gpu_probe = hardware.get("gpu_probe", {})
    rows = [
        f"| Hostname | `{hardware.get('hostname', 'unknown')}` |",
        f"| OS | `{hardware.get('os', 'unknown')}` |",
        f"| Python | `{hardware.get('python_version', 'unknown')}` |",
        f"| CPU model | `{hardware.get('cpu_model', 'unknown')}` |",
        f"| Logical cores | `{hardware.get('logical_cores', 'unknown')}` |",
        f"| Total RAM | `{format_bytes(hardware.get('total_ram_bytes'))}` |",
        f"| GPU detected | `{gpu_probe.get('detected', False)}` ({gpu_probe.get('vendor_hint', 'unknown')}) |",
        f"| GPU used in Stage 1 | `{hardware.get('gpu_used', False)}` |",
        f"| HF home | `{hardware.get('hf_home', 'unknown')}` |",
        f"| Artifact dir | `{hardware.get('artifact_dir', 'unknown')}` |",
    ]
    return "\n".join(rows)


def render_timing_rows(resource_accounting: dict[str, Any]) -> str:
    timing = resource_accounting.get("timing", {})
    phases = timing.get("phases_seconds", {})
    percentages = timing.get("phase_percentages", {})
    rows: list[str] = []
    for name, seconds in phases.items():
        rows.append(
            f"| `{name}` | `{seconds}` | `{percentages.get(name, 0.0)}%` |"
        )
    if not rows:
        rows.append("| `none` | `0.0` | `0.0%` |")
    return "\n".join(rows)


def render_workload_rows(resource_accounting: dict[str, Any]) -> str:
    workload = resource_accounting.get("workload", {})
    roadmap_pairs = workload.get("roadmap_pairs", {})
    rows: list[str] = []
    for pair_name in WEIGHT_ANALYSIS_PAIRWISE:
        payload = roadmap_pairs.get(pair_name, {})
        models = payload.get("models", WEIGHT_ANALYSIS_PAIRWISE[pair_name])
        excluded = payload.get("excluded_counts", {})
        rows.append(
            f"| `{pretty_pair_label(tuple(models))}` | `{payload.get('comparable_tensor_count', 0)}` | `{format_bytes(payload.get('comparable_left_bytes'))}` | `{format_bytes(payload.get('comparable_right_bytes'))}` | `{format_bytes(payload.get('comparable_total_bytes'))}` | `{excluded.get('missing', 0)}` | `{excluded.get('shape_mismatch', 0)}` | `{excluded.get('dtype_mismatch', 0)}` |"
        )
    return "\n".join(rows)


def render_dna_report(
    matrix: dict[str, Any],
    remote_context: dict[str, Any],
    hf_home: Path,
    cache_alias_map: dict[str, str],
    figure_refs: dict[str, str],
) -> str:
    resource_accounting = matrix.get("resource_accounting", {})
    manifest_rows = "\n".join(
        f"| `{alias}` | `{summary['layout']}` | `{', '.join(summary['components'])}` | `{summary['commit']}` | `{summary['shard_count']}` | `{summary['tensor_count']}` | `{summary['normalized_layer_count']}` | `{summary['vae']['label']}` | `{summary['rope']['label']}` |"
        for alias, summary in matrix["model_summaries"].items()
    )
    component_rows = "\n".join(
        f"| `{alias}` | `{component}` | `{count}` |"
        for alias, summary in matrix["model_summaries"].items()
        for component, count in summary["component_tensor_counts"].items()
    )
    comparison_rows = "\n".join(
        f"| `{name}` | `{stats['shared_key_count']}` | `{stats['missing_key_count']}` | `{stats['shape_mismatch_count']}` | `{', '.join(item['prefix'] for item in stats['top_mismatching_prefixes'][:3]) or 'none'}` | `{', '.join(f'{k}:{v}' for k, v in stats['component_breakdown']['left'].items()) or 'none'}` | `{', '.join(f'{k}:{v}' for k, v in stats['component_breakdown']['right'].items()) or 'none'}` |"
        for name, stats in matrix["pairwise_comparisons"].items()
    )
    subsystem_rows = "\n".join(
        f"| `{item['subsystem']}` | {', '.join(item['models'])} | `{item['structural_compatibility']}` | `{item['recommended_merge_strategy']}` | `{item['evidence']['shared_key_count']}` | `{item['evidence']['missing_key_count']}` | `{item['evidence']['shape_mismatch_count']}` | {item['reason']} |"
        for item in matrix["subsystems"]
    )
    timing = resource_accounting.get("timing", {})
    workload = resource_accounting.get("workload", {})
    estimate = resource_accounting.get("estimate", {})
    cache_map_lines = "\n".join(f"- `{alias}` -> `{value}`" for alias, value in cache_alias_map.items())
    return f"""# Stage 1 Paper-Style Block Architecture Review

## Abstract
Stage 1 now combines structural checkpoint compatibility with value-level block comparison for roadmap pairs. The key result is whether models are not only architecturally aligned, but also numerically close enough per block to support low-risk fusion decisions.

## Setup
- Remote name: `{remote_context['name']}`
- Remote workdir: `{remote_context['workdir']}`
- Remote cache: `{remote_context['cache_dir']}`
- Remote artifact dir: `{remote_context['artifact_dir']}`
- HF home: `{hf_home}`
- Weight analysis available: `{matrix['weight_analysis_available']}`
- Low-delta threshold: `relative_l2_delta <= {matrix['weight_thresholds']['relative_l2_delta_low']}`

## Methods
Phase A: structural analysis from shard metadata (key overlap, missing keys, shape mismatches, layer normalization).

Phase B: value-level analysis from loaded tensor payloads on roadmap pairs, with block rollups:
- `exact_tensor_match_ratio`
- `low_delta_tensor_ratio`
- `block_relative_l2_delta`
- `block_mean_abs_delta` and `block_max_abs_delta`

## Cache Entries Inspected
{cache_map_lines}

## Results

### Model Snapshot Inventory
| Alias | Layout | Components | Commit | Shards | Tensor count | Normalized layers | VAE | RoPE hint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
{manifest_rows}

### Component Tensor Counts
| Alias | Component | Tensor count |
| --- | --- | --- |
{component_rows}

### Tensor Pairwise Comparison Stats
| Pair | Shared keys | Missing keys | Shape mismatches | Top mismatch prefixes | Left components | Right components |
| --- | --- | --- | --- | --- | --- | --- |
{comparison_rows}

### Layer Inventory Summary
| Alias | Normalized layers | Subsystem counts |
| --- | --- | --- |
{render_layer_inventory_rows(matrix)}

### Layer Sharing Across All Pairs
| Pair | Shared layers | Exact | Partial | Left-only | Right-only | Shape-mismatched layers | Shared ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
{render_layer_pair_summary_rows(matrix['layer_pairwise'])}

### Block Review Executive Summary
| Pair | Comparable tensors | Exact ratio | Low-delta ratio | Mean relative L2 delta | Mean block similarity |
| --- | --- | --- | --- | --- | --- |
{render_block_review_summary_rows(matrix)}

### Value-Level Weight Comparison
| Pair | Comparable tensors | Exact-equal tensors | Exact ratio | Low-delta ratio | Mean relative L2 delta | Max relative L2 delta | Missing excluded | Shape excluded | Dtype excluded |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
{render_weight_pair_summary_rows(matrix['weight_pairwise'])}

## Hardware Account + Time Usage
### Environment
| Item | Value |
| --- | --- |
{render_hardware_account_rows(resource_accounting)}

### Phase Timing
| Phase | Seconds | Percent of total |
| --- | --- | --- |
{render_timing_rows(resource_accounting)}

### Roadmap Pair Workload
| Pair | Comparable tensors | Left bytes | Right bytes | Total bytes | Missing excluded | Shape excluded | Dtype excluded |
| --- | --- | --- | --- | --- | --- | --- | --- |
{render_workload_rows(resource_accounting)}

### Runtime Estimate vs Observed
- Observed total wall time: `{timing.get('total_wall_seconds', 'unknown')}s`
- Value-analysis bytes processed: `{format_bytes(workload.get('value_analysis_total_bytes'))}` (`{workload.get('value_analysis_total_gib', 'unknown')} GiB`)
- Estimated total runtime (low/typical/high): `{estimate.get('total_seconds', {}).get('low', 'unknown')}s` / `{estimate.get('total_seconds', {}).get('typical', 'unknown')}s` / `{estimate.get('total_seconds', {}).get('high', 'unknown')}s`
- Operational note: Stage 1 value comparison is CPU and storage I/O bound; GPU is not required.

### Subsystem Compatibility And Strategy
| Subsystem | Models | Structural compatibility | Recommended merge strategy | Shared keys | Missing keys | Shape mismatches | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
{subsystem_rows}

### Structural Summary
- `direct-merge`: {matrix['structural_summary']['direct_merge']}
- `adapter-only`: {matrix['structural_summary']['adapter_only']}
- `incompatible`: {matrix['structural_summary']['incompatible']}

### Recommended Strategy Summary
- `direct-merge`: {matrix['summary']['direct_merge']}
- `delta-merge`: {matrix['summary']['delta_merge']}
- `adapter-only`: {matrix['summary']['adapter_only']}
- `incompatible`: {matrix['summary']['incompatible']}

### Evidence Confidence
- Structural evidence confidence: `high` for key/shape compatibility and component-level taxonomy.
- Value evidence confidence: `high` for compared tensors in roadmap pairs, `not-applicable` for excluded tensors (missing/shape/dtype mismatch).

### Primary Figures
![Layer sharing heatmap]({figure_refs['layer_sharing_heatmap']})

![Layer sharing breakdown]({figure_refs['layer_sharing_bars']})

### Supporting Figures
![Component overview]({figure_refs['component_overview']})

![Tensor pairwise comparison]({figure_refs['pairwise_comparison']})

### Layer Sharing By Subsystem
{render_subsystem_layer_tables(matrix['layer_pairwise'])}

### Top Divergent Layers
{render_top_divergent_layers(matrix['layer_pairwise'])}

### Block-By-Block Weight Tables
{render_weight_layer_tables(matrix['weight_pairwise'])}

### Weight-Level Divergences
{render_weight_top_divergences(matrix['weight_pairwise'])}

### Secondary Visualization
```mermaid
{matrix['visualization']['source'].rstrip()}
```

## Limitations
- Numeric comparisons are only performed on shared tensors with matching shape and dtype.
- This report does not measure prompt-level behavior or generation quality; it characterizes checkpoint architecture and weight drift.
- Non-roadmap pair value analysis is intentionally out of scope for runtime control.
"""


def render_stage1_compatibility_stub(target_dir: Path) -> str:
    return f"""# Stage 1 Analysis Report

Canonical Stage 1 report: [stage-1/README.md](stage-1/README.md)

This file is kept as a compatibility shim. Open `{target_dir / 'README.md'}` for the full Stage 1 DNA report.
"""


def build_stage1_terminal_summary(result: dict[str, Any]) -> str:
    matrix = result["matrix"]
    pairwise = matrix["pairwise_comparisons"]
    layer_pairwise = matrix["layer_pairwise"]
    weight_pairwise = matrix["weight_pairwise"]
    resource = matrix.get("resource_accounting", {})
    timing = resource.get("timing", {})
    workload = resource.get("workload", {})
    best_pair_key = max(
        layer_pairwise,
        key=lambda key: layer_pairwise[key]["overall"]["layer_shared_ratio"],
    )
    worst_pair_key = min(
        layer_pairwise,
        key=lambda key: layer_pairwise[key]["overall"]["layer_shared_ratio"],
    )
    block_summary = matrix.get("block_review_summary", {}).get("pairs", {})
    best_block_pair_key = max(
        block_summary,
        key=lambda key: block_summary[key]["mean_block_similarity_score"],
    )
    worst_block_pair_key = min(
        block_summary,
        key=lambda key: block_summary[key]["mean_block_similarity_score"],
    )
    layer_counts = " ".join(
        f"{short_model_label(alias)}={matrix['model_summaries'][alias]['normalized_layer_count']}"
        for alias in MODEL_ORDER
    )
    lines = [
        "Stage 1 DNA analysis",
        f"mode: {result['mode']}",
        f"hf_home: {result['hf_home']}",
        f"artifacts: {result['artifact_dir']}",
        (
            "strategies: "
            f"direct={matrix['summary']['direct_merge']} "
            f"delta={matrix['summary']['delta_merge']} "
            f"adapter={matrix['summary']['adapter_only']} "
            f"incompatible={matrix['summary']['incompatible']}"
        ),
        (
            "structural: "
            f"direct={matrix['structural_summary']['direct_merge']} "
            f"adapter={matrix['structural_summary']['adapter_only']} "
            f"incompatible={matrix['structural_summary']['incompatible']}"
        ),
        f"normalized_layers: {layer_counts}",
        (
            "best layer-sharing pair: "
            f"{pretty_pair_label(layer_pairwise[best_pair_key]['models'])} "
            f"ratio={layer_pairwise[best_pair_key]['overall']['layer_shared_ratio']}"
        ),
        (
            "worst layer-sharing pair: "
            f"{pretty_pair_label(layer_pairwise[worst_pair_key]['models'])} "
            f"ratio={layer_pairwise[worst_pair_key]['overall']['layer_shared_ratio']}"
        ),
        (
            "best block-similarity pair: "
            f"{pretty_pair_label(block_summary[best_block_pair_key]['models'])} "
            f"score={block_summary[best_block_pair_key]['mean_block_similarity_score']}"
        ),
        (
            "worst block-similarity pair: "
            f"{pretty_pair_label(block_summary[worst_block_pair_key]['models'])} "
            f"score={block_summary[worst_block_pair_key]['mean_block_similarity_score']}"
        ),
        (
            "2512 vs 2511 tensors: "
            f"shared={pairwise['foundation_vs_edit']['shared_key_count']} "
            f"missing={pairwise['foundation_vs_edit']['missing_key_count']} "
            f"shape_mismatch={pairwise['foundation_vs_edit']['shape_mismatch_count']}"
        ),
        (
            "base vs layered tensors: "
            f"shared={pairwise['base_vs_layered']['shared_key_count']} "
            f"missing={pairwise['base_vs_layered']['missing_key_count']} "
            f"shape_mismatch={pairwise['base_vs_layered']['shape_mismatch_count']}"
        ),
        (
            "2512 vs 2511 weights: "
            f"exact_ratio={weight_pairwise['foundation_vs_edit']['exact_equal_tensor_ratio']} "
            f"mean_rel_l2={weight_pairwise['foundation_vs_edit']['mean_relative_l2_delta']}"
        ),
        (
            "base vs layered weights: "
            f"exact_ratio={weight_pairwise['base_vs_layered']['exact_equal_tensor_ratio']} "
            f"mean_rel_l2={weight_pairwise['base_vs_layered']['mean_relative_l2_delta']}"
        ),
        (
            "runtime_total_seconds: "
            f"{timing.get('total_wall_seconds', 'unknown')}"
        ),
        (
            "value_analysis_bytes: "
            f"{format_bytes(workload.get('value_analysis_total_bytes'))}"
        ),
    ]
    artifact_refs = result.get("artifact_paths", {})
    if artifact_refs:
        lines.extend(
            [
                f"summary: {artifact_refs.get('summary_markdown', '')}",
                f"matrix: {artifact_refs.get('matrix_json', '')}",
                f"hardware_account: {artifact_refs.get('summary_markdown', '')} (section: Hardware Account + Time Usage)",
                f"layer_analysis: {artifact_refs.get('layer_analysis_json', '')}",
                f"weight_analysis: {artifact_refs.get('weight_analysis_json', '')}",
                f"figure(heatmap): {artifact_refs.get('layer_sharing_heatmap_png', '')}",
                f"figure(layer-bars): {artifact_refs.get('layer_sharing_bars_png', '')}",
            ]
        )
    if result.get("compatibility_shims"):
        lines.append(
            f"compatibility_matrix_shim: {result['compatibility_shims'].get('legacy_matrix_json', '')}"
        )
    lines.append("use --json for the full machine-readable payload")
    return "\n".join(lines)
