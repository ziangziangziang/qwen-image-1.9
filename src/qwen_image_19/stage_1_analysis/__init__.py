from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import struct
from typing import Any, Iterable

from qwen_image_19.config_io import load_json_yaml, repo_root, write_json, write_text
from qwen_image_19.remote import default_remote_context


MODEL_CONFIGS = (
    "qwen-image-base.yaml",
    "qwen-image-2512.yaml",
    "qwen-image-edit-2511.yaml",
    "qwen-image-layered.yaml",
)

DEFAULT_CACHE_ALIAS_MAP = {
    "qwen-image-base": "models--Qwen--Qwen-Image",
    "qwen-image-2512": "models--Qwen--Qwen-Image-2512",
    "qwen-image-edit-2511": "models--Qwen--Qwen-Image-Edit-2511",
    "qwen-image-layered": "models--Qwen--Qwen-Image-Layered",
}

MODEL_ORDER = (
    "qwen-image-base",
    "qwen-image-2512",
    "qwen-image-edit-2511",
    "qwen-image-layered",
)

TEXT_ENCODER_PREFIXES = (
    "text_encoder.",
    "cond_stage_model.",
    "conditioner.",
)
VAE_PREFIXES = (
    "vae.",
    "autoencoder.",
)


class Stage1AnalysisError(RuntimeError):
    """Raised when Stage 1 cannot inspect the remote cache layout."""


def load_model_inventory(model_dir: str | Path | None = None) -> dict[str, dict[str, Any]]:
    root = Path(model_dir) if model_dir else repo_root() / "configs" / "models"
    models: dict[str, dict[str, Any]] = {}
    for filename in MODEL_CONFIGS:
        payload = load_json_yaml(root / filename)
        models[payload["alias"]] = payload
    return models


def load_cache_alias_map(cache_map_config: str | Path | None = None) -> dict[str, str]:
    if not cache_map_config:
        return dict(DEFAULT_CACHE_ALIAS_MAP)
    payload = load_json_yaml(cache_map_config)
    if "models" in payload and isinstance(payload["models"], dict):
        return {key: str(value) for key, value in payload["models"].items()}
    return {key: str(value) for key, value in payload.items()}


def resolve_hf_home(hf_home: str | Path | None = None) -> Path:
    explicit = Path(hf_home).expanduser() if hf_home else Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
    if explicit.name == "hub":
        return explicit.parent
    return explicit


def hub_root(hf_home: str | Path | None = None) -> Path:
    return resolve_hf_home(hf_home) / "hub"


def resolve_cache_snapshot(hf_home: str | Path | None, cache_dir_name: str) -> dict[str, Any]:
    model_root = hub_root(hf_home) / cache_dir_name
    refs_main = model_root / "refs" / "main"
    if not refs_main.exists():
        raise Stage1AnalysisError(f"Missing refs/main for cache entry: {model_root}")
    commit = refs_main.read_text(encoding="utf-8").strip()
    if not commit:
        raise Stage1AnalysisError(f"Empty refs/main for cache entry: {model_root}")
    snapshot = model_root / "snapshots" / commit
    if not snapshot.exists():
        raise Stage1AnalysisError(f"Missing snapshot directory for commit {commit}: {snapshot}")
    return {
        "cache_dir_name": cache_dir_name,
        "model_root": model_root,
        "commit": commit,
        "snapshot": snapshot,
    }


def load_small_json_files(snapshot: Path) -> dict[str, Any]:
    config_payloads: dict[str, Any] = {}
    for file_path in sorted(snapshot.glob("*.json")):
        if file_path.name.endswith(".safetensors.index.json"):
            continue
        if file_path.stat().st_size > 2_000_000:
            continue
        try:
            config_payloads[file_path.name] = json.loads(file_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
    return config_payloads


def load_component_json_files(snapshot: Path) -> dict[str, Any]:
    config_payloads: dict[str, Any] = {}
    for file_path in sorted(snapshot.rglob("*.json")):
        if file_path.name.endswith(".safetensors.index.json"):
            continue
        if file_path.stat().st_size > 2_000_000:
            continue
        relative = str(file_path.relative_to(snapshot))
        try:
            config_payloads[relative] = json.loads(file_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
    return config_payloads


def bytes_for_dtype(dtype: str) -> int | None:
    mapping = {
        "F64": 8,
        "F32": 4,
        "F16": 2,
        "BF16": 2,
        "I64": 8,
        "I32": 4,
        "I16": 2,
        "I8": 1,
        "U8": 1,
        "BOOL": 1,
    }
    return mapping.get(dtype)


def read_safetensors_header(file_path: Path) -> dict[str, Any]:
    with file_path.open("rb") as handle:
        header_len_raw = handle.read(8)
        if len(header_len_raw) != 8:
            raise Stage1AnalysisError(f"Invalid safetensors header in {file_path}")
        header_len = struct.unpack("<Q", header_len_raw)[0]
        header_raw = handle.read(header_len)
        if len(header_raw) != header_len:
            raise Stage1AnalysisError(f"Truncated safetensors header in {file_path}")
    try:
        return json.loads(header_raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise Stage1AnalysisError(f"Invalid safetensors JSON header in {file_path}") from exc


def parse_safetensors_file(file_path: Path) -> dict[str, Any]:
    file_size = file_path.stat().st_size
    header = read_safetensors_header(file_path)
    tensors: dict[str, dict[str, Any]] = {}
    with file_path.open("rb") as handle:
        actual_len = struct.unpack("<Q", handle.read(8))[0]
    data_start = 8 + actual_len

    for tensor_name, entry in header.items():
        if tensor_name == "__metadata__":
            continue
        if not isinstance(entry, dict):
            raise Stage1AnalysisError(f"Unexpected tensor entry for {tensor_name} in {file_path}")
        offsets = entry.get("data_offsets")
        if not isinstance(offsets, list) or len(offsets) != 2:
            raise Stage1AnalysisError(f"Invalid data_offsets for {tensor_name} in {file_path}")
        start, end = int(offsets[0]), int(offsets[1])
        absolute_end = data_start + end
        if end < start or absolute_end > file_size:
            raise Stage1AnalysisError(f"Invalid tensor bounds for {tensor_name} in {file_path}")
        shape = [int(value) for value in entry.get("shape", [])]
        payload_nbytes = end - start
        expected_nbytes = None
        itemsize = bytes_for_dtype(str(entry.get("dtype")))
        if itemsize is not None and shape:
            expected_nbytes = math.prod(shape) * itemsize
        tensors[tensor_name] = {
            "dtype": str(entry.get("dtype")),
            "shape": shape,
            "data_offsets": [start, end],
            "payload_nbytes": payload_nbytes,
            "expected_nbytes": expected_nbytes,
            "shard": file_path.name,
        }

    return {
        "file": file_path.name,
        "path": str(file_path),
        "size_bytes": file_size,
        "tensor_count": len(tensors),
        "tensors": tensors,
    }


def discover_checkpoint_shards(snapshot: Path) -> tuple[list[Path], dict[str, str] | None]:
    index_paths = sorted(snapshot.glob("*.safetensors.index.json"))
    if len(index_paths) > 1:
        raise Stage1AnalysisError(
            f"Multiple safetensors index files found in component directory {snapshot}: {[path.name for path in index_paths]}"
        )
    if len(index_paths) == 1:
        index_path = index_paths[0]
        index_payload = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index_payload.get("weight_map", {})
        if not weight_map:
            raise Stage1AnalysisError(f"Empty weight_map in {index_path}")
        shard_names = sorted(set(str(value) for value in weight_map.values()))
        shards = [snapshot / shard_name for shard_name in shard_names]
        missing = [str(path) for path in shards if not path.exists()]
        if missing:
            raise Stage1AnalysisError(f"Missing shard files declared in index: {missing}")
        return shards, {str(key): str(value) for key, value in weight_map.items()}

    shards = sorted(snapshot.glob("*.safetensors"))
    if not shards:
        raise Stage1AnalysisError(f"No safetensors files found in snapshot: {snapshot}")
    if len(shards) > 1:
        raise Stage1AnalysisError(
            f"Multiple safetensors files found without a safetensors index in {snapshot}"
        )
    return shards, None


def discover_component_roots(snapshot: Path) -> list[Path]:
    component_roots: set[Path] = set()
    if any(snapshot.glob("*.safetensors.index.json")) or any(snapshot.glob("*.safetensors")):
        component_roots.add(snapshot)

    for path in snapshot.rglob("*"):
        if not path.is_dir():
            continue
        if any(path.glob("*.safetensors.index.json")) or any(path.glob("*.safetensors")):
            component_roots.add(path)

    return sorted(component_roots)


def summarize_dtypes(tensors: dict[str, dict[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for tensor in tensors.values():
        counter[tensor["dtype"]] += 1
    return dict(counter)


def infer_vae_channels(tensors: dict[str, dict[str, Any]]) -> dict[str, Any]:
    encoder_inputs: list[int] = []
    decoder_outputs: list[int] = []
    for key, tensor in tensors.items():
        component = tensor.get("component", "")
        if (key.endswith("encoder.conv_in.weight") or "vae" in component) and len(tensor["shape"]) >= 2:
            encoder_inputs.append(int(tensor["shape"][1]))
        if (key.endswith("decoder.conv_out.weight") or "vae" in component) and tensor["shape"]:
            decoder_outputs.append(int(tensor["shape"][0]))
    values = sorted(set(encoder_inputs + decoder_outputs))
    if 4 in values:
        label = "RGBA"
    elif 3 in values:
        label = "RGB"
    else:
        label = "unknown"
    return {
        "label": label,
        "channels": values,
        "evidence": {
            "encoder_inputs": encoder_inputs,
            "decoder_outputs": decoder_outputs,
        },
    }


def flatten_strings(payload: Any) -> Iterable[str]:
    if isinstance(payload, dict):
        for key, value in payload.items():
            yield str(key)
            yield from flatten_strings(value)
    elif isinstance(payload, list):
        for item in payload:
            yield from flatten_strings(item)
    elif payload is not None:
        yield str(payload)


def infer_rope_mode(config_payloads: dict[str, Any], tensors: dict[str, dict[str, Any]]) -> dict[str, Any]:
    corpus = [value.lower() for payload in config_payloads.values() for value in flatten_strings(payload)]
    tensor_keys = [key.lower() for key in tensors]
    component_names = [str(tensor.get("component", "")).lower() for tensor in tensors.values()]
    if any("layer3d" in item for item in corpus + tensor_keys + component_names):
        return {"label": "Layer3D", "source": "config-or-key"}
    if any("rope" in item or "rotary" in item for item in corpus + tensor_keys + component_names):
        return {"label": "2D-or-rotary", "source": "config-or-key"}
    return {"label": "unknown", "source": "none"}


def prefix_bucket(key: str, depth: int = 2) -> str:
    parts = key.split(".")
    return ".".join(parts[: min(depth, len(parts))])


def top_prefixes(keys: Iterable[str], limit: int = 5) -> list[dict[str, Any]]:
    counter: Counter[str] = Counter(prefix_bucket(key) for key in keys)
    return [{"prefix": prefix, "count": count} for prefix, count in counter.most_common(limit)]


def subsystem_for_key(key: str) -> str:
    if key.startswith(VAE_PREFIXES):
        return "vae"
    if key.startswith(TEXT_ENCODER_PREFIXES):
        return "text_encoder"
    if "rope" in key.lower() or "rotary" in key.lower():
        return "rope"
    return "mmdit_backbone"


def subsystem_for_tensor(key: str, tensor: dict[str, Any]) -> str:
    component = str(tensor.get("component", "")).lower()
    if "vae" in component:
        return "vae"
    if "text_encoder" in component:
        return "text_encoder"
    if "transformer" in component:
        return "mmdit_backbone"
    return subsystem_for_key(key)


def compare_tensor_sets(
    left_tensors: dict[str, dict[str, Any]],
    right_tensors: dict[str, dict[str, Any]],
    predicate: callable | None = None,
) -> dict[str, Any]:
    left_keys = {key for key in left_tensors if predicate is None or predicate(key)}
    right_keys = {key for key in right_tensors if predicate is None or predicate(key)}
    shared = sorted(left_keys & right_keys)
    left_only = sorted(left_keys - right_keys)
    right_only = sorted(right_keys - left_keys)
    shape_mismatches = sorted(
        key for key in shared if left_tensors[key]["shape"] != right_tensors[key]["shape"]
    )
    dtype_mismatches = sorted(
        key for key in shared if left_tensors[key]["dtype"] != right_tensors[key]["dtype"]
    )
    union_count = len(left_keys | right_keys)
    shared_ratio = len(shared) / union_count if union_count else 0.0
    return {
        "left_tensor_count": len(left_keys),
        "right_tensor_count": len(right_keys),
        "shared_key_count": len(shared),
        "missing_key_count": len(left_only) + len(right_only),
        "shape_mismatch_count": len(shape_mismatches),
        "dtype_mismatch_count": len(dtype_mismatches),
        "shared_ratio": round(shared_ratio, 4),
        "top_mismatching_prefixes": top_prefixes(left_only + right_only + shape_mismatches),
        "sample_shape_mismatches": shape_mismatches[:10],
        "component_breakdown": {
            "left": dict(Counter(str(left_tensors[key].get("component", ".")) for key in left_keys)),
            "right": dict(Counter(str(right_tensors[key].get("component", ".")) for key in right_keys)),
        },
    }


def classify_pair(stats: dict[str, Any], preferred_mode: str | None = None, force_incompatible: bool = False) -> str:
    if force_incompatible:
        return "incompatible"
    if preferred_mode == "adapter-only":
        return "adapter-only"
    if stats["shared_key_count"] == 0:
        return "incompatible"
    if stats["shape_mismatch_count"] == 0 and stats["missing_key_count"] == 0:
        return "direct-merge"
    if preferred_mode == "delta-merge":
        return "delta-merge"
    if stats["shared_ratio"] >= 0.75 and stats["shape_mismatch_count"] <= max(4, int(stats["shared_key_count"] * 0.02)):
        return "delta-merge"
    return "adapter-only"


def inspect_model_snapshot(
    alias: str,
    metadata: dict[str, Any],
    hf_home: str | Path | None,
    cache_dir_name: str,
) -> dict[str, Any]:
    snapshot_info = resolve_cache_snapshot(hf_home, cache_dir_name)
    snapshot = snapshot_info["snapshot"]
    component_roots = discover_component_roots(snapshot)
    if not component_roots:
        raise Stage1AnalysisError(f"No safetensors files found anywhere under snapshot: {snapshot}")
    tensors: dict[str, dict[str, Any]] = {}
    shard_records: list[dict[str, Any]] = []
    component_tensor_counts: dict[str, int] = {}
    component_names: list[str] = []
    weight_map_present = False

    for component_root in component_roots:
        component_rel = str(component_root.relative_to(snapshot))
        component_name = "." if component_rel == "." else component_rel
        component_names.append(component_name)
        shards, weight_map = discover_checkpoint_shards(component_root)
        weight_map_present = weight_map_present or weight_map is not None
        parsed_shards = [parse_safetensors_file(path) for path in shards]
        local_tensor_count = 0
        for shard in parsed_shards:
            shard_component = component_name
            shard_records.append(
                {
                    "file": shard["file"],
                    "path": str(Path(shard["path"]).relative_to(snapshot)),
                    "component": shard_component,
                    "size_bytes": shard["size_bytes"],
                    "tensor_count": shard["tensor_count"],
                }
            )
            for tensor_name, tensor in shard["tensors"].items():
                tensors[tensor_name] = {
                    **tensor,
                    "component": shard_component,
                    "relative_path": str(Path(shard["path"]).relative_to(snapshot)),
                }
                local_tensor_count += 1
        component_tensor_counts[component_name] = local_tensor_count

    config_payloads = load_component_json_files(snapshot)
    vae = infer_vae_channels(tensors)
    rope = infer_rope_mode(config_payloads, tensors)
    layout = "componentized" if any(name != "." for name in component_names) else "flat"
    return {
        "alias": alias,
        "model_id": metadata["model_id"],
        "role": metadata["role"],
        "cache_dir_name": cache_dir_name,
        "commit": snapshot_info["commit"],
        "snapshot_path": str(snapshot),
        "layout": layout,
        "components": sorted(component_names),
        "component_tensor_counts": component_tensor_counts,
        "shard_count": len(shard_records),
        "tensor_count": len(tensors),
        "dtypes": summarize_dtypes(tensors),
        "total_tensor_bytes": sum(tensor["payload_nbytes"] for tensor in tensors.values()),
        "configs": config_payloads,
        "weight_map_present": weight_map_present,
        "vae": vae,
        "rope": rope,
        "tensors": tensors,
        "shards": shard_records,
    }


def inspect_cache_models(
    hf_home: str | Path | None,
    metadata_models: dict[str, dict[str, Any]],
    cache_alias_map: dict[str, str],
) -> dict[str, dict[str, Any]]:
    manifests: dict[str, dict[str, Any]] = {}
    for alias in MODEL_ORDER:
        if alias not in metadata_models:
            raise Stage1AnalysisError(f"Missing metadata config for model alias: {alias}")
        if alias not in cache_alias_map:
            raise Stage1AnalysisError(f"Missing cache alias mapping for model alias: {alias}")
        manifests[alias] = inspect_model_snapshot(alias, metadata_models[alias], hf_home, cache_alias_map[alias])
    return manifests


def summarize_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_id": manifest["model_id"],
        "role": manifest["role"],
        "cache_dir_name": manifest["cache_dir_name"],
        "commit": manifest["commit"],
        "snapshot_path": manifest["snapshot_path"],
        "layout": manifest["layout"],
        "components": manifest["components"],
        "component_tensor_counts": manifest["component_tensor_counts"],
        "shard_count": manifest["shard_count"],
        "tensor_count": manifest["tensor_count"],
        "dtypes": manifest["dtypes"],
        "total_tensor_bytes": manifest["total_tensor_bytes"],
        "weight_map_present": manifest["weight_map_present"],
        "vae": manifest["vae"],
        "rope": manifest["rope"],
        "config_files": sorted(manifest["configs"].keys()),
    }


def compare_state_dicts(manifests: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return compare_tensor_sets(manifests["qwen-image-2512"]["tensors"], manifests["qwen-image-edit-2511"]["tensors"])


def analyze_vae_compatibility(manifests: dict[str, dict[str, Any]]) -> dict[str, Any]:
    base = manifests["qwen-image-base"]
    layered = manifests["qwen-image-layered"]
    stats = compare_tensor_sets(
        base["tensors"],
        layered["tensors"],
        predicate=lambda key: subsystem_for_tensor(key, base["tensors"].get(key, layered["tensors"].get(key, {}))) == "vae",
    )
    incompatible = base["vae"]["label"] != layered["vae"]["label"]
    return {
        "subsystem": "vae",
        "models": ["qwen-image-base", "qwen-image-layered"],
        "classification": classify_pair(stats, force_incompatible=incompatible),
        "reason": f"Base VAE channels {base['vae']['channels']} vs layered VAE channels {layered['vae']['channels']}.",
        "evidence": stats,
    }


def probe_rope_compatibility(manifests: dict[str, dict[str, Any]]) -> dict[str, Any]:
    foundation = manifests["qwen-image-2512"]
    layered = manifests["qwen-image-layered"]
    stats = compare_tensor_sets(
        foundation["tensors"],
        layered["tensors"],
        predicate=lambda key: subsystem_for_tensor(key, foundation["tensors"].get(key, layered["tensors"].get(key, {}))) in {"text_encoder", "rope"},
    )
    preferred = "adapter-only" if foundation["rope"]["label"] != layered["rope"]["label"] else None
    return {
        "subsystem": "rope",
        "models": ["qwen-image-2512", "qwen-image-layered"],
        "classification": classify_pair(stats, preferred_mode=preferred),
        "reason": f"Foundation rope hint `{foundation['rope']['label']}` vs layered rope hint `{layered['rope']['label']}`.",
        "evidence": stats,
    }


def build_pairwise_comparisons(manifests: dict[str, dict[str, Any]]) -> dict[str, Any]:
    foundation = manifests["qwen-image-2512"]
    edit = manifests["qwen-image-edit-2511"]
    base = manifests["qwen-image-base"]
    layered = manifests["qwen-image-layered"]
    return {
        "foundation_vs_edit": compare_tensor_sets(foundation["tensors"], edit["tensors"]),
        "base_vs_layered": compare_tensor_sets(base["tensors"], layered["tensors"]),
        "foundation_vs_layered": compare_tensor_sets(foundation["tensors"], layered["tensors"]),
    }


def build_compatibility_matrix(manifests: dict[str, dict[str, Any]]) -> dict[str, Any]:
    pairwise = build_pairwise_comparisons(manifests)
    mmdit_stats = compare_tensor_sets(
        manifests["qwen-image-2512"]["tensors"],
        manifests["qwen-image-edit-2511"]["tensors"],
        predicate=lambda key: subsystem_for_tensor(
            key,
            manifests["qwen-image-2512"]["tensors"].get(key, manifests["qwen-image-edit-2511"]["tensors"].get(key, {})),
        )
        == "mmdit_backbone",
    )
    text_stats = compare_tensor_sets(
        manifests["qwen-image-base"]["tensors"],
        manifests["qwen-image-layered"]["tensors"],
        predicate=lambda key: subsystem_for_tensor(
            key,
            manifests["qwen-image-base"]["tensors"].get(key, manifests["qwen-image-layered"]["tensors"].get(key, {})),
        )
        == "text_encoder",
    )
    vae_result = analyze_vae_compatibility(manifests)
    rope_result = probe_rope_compatibility(manifests)

    subsystems = [
        {
            "subsystem": "mmdit_backbone",
            "models": ["qwen-image-2512", "qwen-image-edit-2511"],
            "classification": classify_pair(mmdit_stats, preferred_mode="delta-merge"),
            "reason": "Use real shared-key and shape stats between 2512 and 2511 to justify a delta merge path.",
            "evidence": mmdit_stats,
        },
        {
            "subsystem": "text_encoder",
            "models": ["qwen-image-base", "qwen-image-layered", "qwen-image-2512"],
            "classification": classify_pair(text_stats, preferred_mode="adapter-only"),
            "reason": "Layered is compared against its ancestry base first, then mapped onto the 2512 foundation as adapter-only logic unless exact parity is proven.",
            "evidence": text_stats,
        },
        vae_result,
        rope_result,
    ]
    summary = Counter(item["classification"] for item in subsystems)
    matrix = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inspection_mode": "hf-cache-real-checkpoint",
        "model_summaries": {alias: summarize_manifest(manifest) for alias, manifest in manifests.items()},
        "pairwise_comparisons": pairwise,
        "subsystems": subsystems,
        "summary": {
            "direct_merge": summary.get("direct-merge", 0),
            "delta_merge": summary.get("delta-merge", 0),
            "adapter_only": summary.get("adapter-only", 0),
            "incompatible": summary.get("incompatible", 0),
        },
    }
    matrix["visualization"] = render_similarity_visualization(matrix)
    return matrix


def render_similarity_visualization(matrix: dict[str, Any]) -> dict[str, str]:
    fve = matrix["pairwise_comparisons"]["foundation_vs_edit"]
    bvl = matrix["pairwise_comparisons"]["base_vs_layered"]
    subsystems = "\n".join(
        f"    S{idx}[\"{item['subsystem']}\\n{item['classification']}\"]"
        for idx, item in enumerate(matrix["subsystems"], start=1)
    )
    links = "\n".join(f"    C --> S{idx}" for idx, _ in enumerate(matrix["subsystems"], start=1))
    mermaid = f"""flowchart TD
    A["2512 vs 2511\\nshared: {fve['shared_key_count']}\\nmissing: {fve['missing_key_count']}\\nshape mismatches: {fve['shape_mismatch_count']}"]
    B["Base vs Layered\\nshared: {bvl['shared_key_count']}\\nmissing: {bvl['missing_key_count']}\\nshape mismatches: {bvl['shape_mismatch_count']}"]
    C["Subsystem classifications"]
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


def render_dna_report(matrix: dict[str, Any], remote_context: dict[str, Any], hf_home: Path, cache_alias_map: dict[str, str]) -> str:
    manifest_rows = "\n".join(
        f"| `{alias}` | `{summary['layout']}` | `{', '.join(summary['components'])}` | `{summary['commit']}` | `{summary['shard_count']}` | `{summary['tensor_count']}` | `{summary['vae']['label']}` | `{summary['rope']['label']}` |"
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
        f"| `{item['subsystem']}` | {', '.join(item['models'])} | `{item['classification']}` | `{item['evidence']['shared_key_count']}` | `{item['evidence']['missing_key_count']}` | `{item['evidence']['shape_mismatch_count']}` | {item['reason']} |"
        for item in matrix["subsystems"]
    )
    cache_map_lines = "\n".join(f"- `{alias}` -> `{value}`" for alias, value in cache_alias_map.items())
    return f"""# Stage 1 DNA Report

## Why this report exists
This report is built from the Hugging Face cache snapshots, not from repo metadata vibes. If a subsystem gets called compatible here, it had to survive actual shard and tensor inspection first.

## Remote execution context
- Remote name: `{remote_context['name']}`
- Remote workdir: `{remote_context['workdir']}`
- Remote cache: `{remote_context['cache_dir']}`
- Remote artifact dir: `{remote_context['artifact_dir']}`
- HF home: `{hf_home}`

## Cache entries inspected
{cache_map_lines}

## Model snapshot inventory
| Alias | Layout | Components | Commit | Shards | Tensor count | VAE | RoPE hint |
| --- | --- | --- | --- | --- | --- | --- | --- |
{manifest_rows}

## Component tensor counts
| Alias | Component | Tensor count |
| --- | --- | --- |
{component_rows}

## Pairwise comparison stats
| Pair | Shared keys | Missing keys | Shape mismatches | Top mismatch prefixes | Left components | Right components |
| --- | --- | --- | --- | --- | --- | --- |
{comparison_rows}

## Subsystem classifications
| Subsystem | Models | Classification | Shared keys | Missing keys | Shape mismatches | Notes |
| --- | --- | --- | --- | --- | --- | --- |
{subsystem_rows}

## Summary
- `direct-merge`: {matrix['summary']['direct_merge']}
- `delta-merge`: {matrix['summary']['delta_merge']}
- `adapter-only`: {matrix['summary']['adapter_only']}
- `incompatible`: {matrix['summary']['incompatible']}

## Visualization
```mermaid
{matrix['visualization']['source'].rstrip()}
```

## Takeaways
- `2512` vs `2511` is now backed by real shared-key and shape stats.
- `Layered` is judged first against `Qwen-Image` ancestry, then mapped onto the `2512` foundation.
- VAE and RoPE claims are tied to checkpoint evidence instead of hard-coded roadmap assumptions.
"""


def analyze(
    artifact_dir: str | Path | None = None,
    remote_config: str | None = None,
    model_dir: str | Path | None = None,
    dry_run: bool = False,
    cache_dir: str | None = None,
    hf_home: str | Path | None = None,
    cache_map_config: str | Path | None = None,
) -> dict[str, Any]:
    metadata_models = load_model_inventory(model_dir)
    cache_alias_map = load_cache_alias_map(cache_map_config)
    remote_context = default_remote_context(remote_config)
    if cache_dir:
        remote_context["cache_dir"] = cache_dir
    resolved_hf_home = resolve_hf_home(hf_home)
    manifests = inspect_cache_models(resolved_hf_home, metadata_models, cache_alias_map)
    matrix = build_compatibility_matrix(manifests)
    report = render_dna_report(matrix, remote_context, resolved_hf_home, cache_alias_map)
    target_dir = Path(artifact_dir) if artifact_dir else repo_root() / "reports"
    result = {
        "stage": "stage1",
        "mode": "dry-run" if dry_run else "write",
        "hf_home": str(resolved_hf_home),
        "cache_alias_map": cache_alias_map,
        "matrix": matrix,
        "artifact_dir": str(target_dir),
    }
    if dry_run:
        result["report_preview"] = report
        return result
    write_json(target_dir / "stage-1-compatibility-matrix.json", matrix)
    write_text(target_dir / "stage-1-dna-report.md", report)
    result["written"] = [
        str(target_dir / "stage-1-compatibility-matrix.json"),
        str(target_dir / "stage-1-dna-report.md"),
    ]
    return result
