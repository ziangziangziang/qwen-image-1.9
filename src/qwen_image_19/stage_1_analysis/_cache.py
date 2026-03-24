from __future__ import annotations

from collections import Counter
from itertools import combinations
import json
import math
import os
from pathlib import Path
import struct
from typing import Any, Callable, Iterable

from qwen_image_19.config_io import load_json, repo_root
from qwen_image_19.stage_1_analysis._hardware import MODEL_ORDER, SUBSYSTEM_ORDER, Stage1AnalysisError


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

TEXT_ENCODER_PREFIXES = (
    "text_encoder.",
    "cond_stage_model.",
    "conditioner.",
)
VAE_PREFIXES = (
    "vae.",
    "autoencoder.",
)

SHORT_ALIAS = {
    "qwen-image-base": "base",
    "qwen-image-2512": "2512",
    "qwen-image-edit-2511": "edit-2511",
    "qwen-image-layered": "layered",
}


def load_model_inventory(model_dir: str | Path | None = None) -> dict[str, dict[str, Any]]:
    root = Path(model_dir) if model_dir else repo_root() / "configs" / "models"
    models: dict[str, dict[str, Any]] = {}
    for filename in MODEL_CONFIGS:
        payload = load_json(root / filename)
        models[payload["alias"]] = payload
    return models


def load_cache_alias_map(cache_map_config: str | Path | None = None) -> dict[str, str]:
    if not cache_map_config:
        return dict(DEFAULT_CACHE_ALIAS_MAP)
    payload = load_json(cache_map_config)
    if "models" in payload and isinstance(payload["models"], dict):
        return {key: str(value) for key, value in payload["models"].items()}
    return {key: str(value) for key, value in payload.items()}


def resolve_hf_home(hf_home: str | Path | None = None) -> Path:
    explicit = (
        Path(hf_home).expanduser()
        if hf_home
        else Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
    )
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


def collect_numeric_fields(payload: Any, field_name: str) -> list[int]:
    values: list[int] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key == field_name and isinstance(value, (int, float)):
                values.append(int(value))
            else:
                values.extend(collect_numeric_fields(value, field_name))
    elif isinstance(payload, list):
        for item in payload:
            values.extend(collect_numeric_fields(item, field_name))
    return values


def pick_channel(values: list[int]) -> int | None:
    if not values:
        return None
    counter = Counter(values)
    return sorted(counter.items(), key=lambda item: (-item[1], item[0]))[0][0]


def infer_vae_channels(config_payloads: dict[str, Any], tensors: dict[str, dict[str, Any]]) -> dict[str, Any]:
    encoder_inputs: list[int] = []
    decoder_outputs: list[int] = []
    for key, tensor in tensors.items():
        raw_name = str(tensor.get("raw_name", key))
        if key.endswith("encoder.conv_in.weight") or raw_name.endswith("encoder.conv_in.weight"):
            if len(tensor["shape"]) >= 2:
                encoder_inputs.append(int(tensor["shape"][1]))
        if key.endswith("decoder.conv_out.weight") or raw_name.endswith("decoder.conv_out.weight"):
            if tensor["shape"]:
                decoder_outputs.append(int(tensor["shape"][0]))

    config_inputs: list[int] = []
    config_outputs: list[int] = []
    for relative_path, payload in config_payloads.items():
        path_lower = relative_path.lower()
        if "vae" not in path_lower and payload.get("component") != "vae":
            continue
        config_inputs.extend(collect_numeric_fields(payload, "in_channels"))
        config_outputs.extend(collect_numeric_fields(payload, "out_channels"))

    values = sorted(set(encoder_inputs + decoder_outputs + config_inputs + config_outputs))
    input_channel = pick_channel(encoder_inputs + config_inputs)
    output_channel = pick_channel(decoder_outputs + config_outputs)
    if 4 in values:
        label = "RGBA"
    elif 3 in values:
        label = "RGB"
    else:
        label = "unknown"
    return {
        "label": label,
        "channels": values,
        "input_channels": input_channel,
        "output_channels": output_channel,
        "evidence": {
            "encoder_inputs": encoder_inputs,
            "decoder_outputs": decoder_outputs,
            "config_inputs": config_inputs,
            "config_outputs": config_outputs,
        },
    }


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


def pair_alias(left: str, right: str) -> str:
    return f"{left}_vs_{right}"


def short_model_label(alias: str) -> str:
    return SHORT_ALIAS.get(alias, alias)


def pretty_pair_label(models: list[str] | tuple[str, str]) -> str:
    left, right = models
    return f"{short_model_label(left)} vs {short_model_label(right)}"


def component_prefix(component_name: str) -> str:
    return component_name.replace("/", ".") if component_name not in {"", "."} else ""


def canonical_tensor_name(raw_name: str, component_name: str) -> str:
    prefix = component_prefix(component_name)
    if not prefix:
        return raw_name
    if raw_name == prefix or raw_name.startswith(f"{prefix}."):
        return raw_name
    return f"{prefix}.{raw_name}"


def tensor_component_scope(key: str, tensor: dict[str, Any], subsystem: str) -> str:
    component = component_prefix(str(tensor.get("component", ".")))
    if component:
        return component.split(".")[0]
    if subsystem == "mmdit_backbone":
        return "transformer"
    if subsystem == "text_encoder":
        return "text_encoder"
    if subsystem == "vae":
        return "vae"
    if subsystem == "rope":
        return "rope"
    return key.split(".", 1)[0] or "root"


def normalize_layer_descriptor(key: str, tensor: dict[str, Any]) -> dict[str, str]:
    subsystem = subsystem_for_tensor(key, tensor)
    scope = tensor_component_scope(key, tensor, subsystem)
    parts = [part for part in key.split(".") if part]

    indexed_families = {"layers", "blocks", "transformer_blocks", "single_transformer_blocks"}
    for index, part in enumerate(parts[:-1]):
        if part in indexed_families and index + 1 < len(parts) and parts[index + 1].isdigit():
            family = part
            layer_id = f"{subsystem}:{family}:{parts[index + 1]}"
            suffix = ".".join(parts[index + 2 :]) or "__self__"
            return {
                "subsystem": subsystem,
                "family": family,
                "layer_id": layer_id,
                "parameter_suffix": suffix,
                "component_scope": scope,
            }

    if subsystem == "vae":
        for index, part in enumerate(parts[:-2]):
            if part in {"encoder", "decoder"} and parts[index + 1] in {"down_blocks", "up_blocks"} and parts[index + 2].isdigit():
                family = parts[index + 1]
                layer_id = f"vae:{part}.{family}:{parts[index + 2]}"
                suffix = ".".join(parts[index + 3 :]) or "__self__"
                return {
                    "subsystem": subsystem,
                    "family": family,
                    "layer_id": layer_id,
                    "parameter_suffix": suffix,
                    "component_scope": scope,
                }
        for index, part in enumerate(parts[:-1]):
            if part in {"encoder", "decoder"} and parts[index + 1] == "mid_block":
                suffix = ".".join(parts[index + 2 :]) or "__self__"
                return {
                    "subsystem": subsystem,
                    "family": "mid_block",
                    "layer_id": f"vae:{part}.mid_block",
                    "parameter_suffix": suffix,
                    "component_scope": scope,
                }
            if part in {"encoder", "decoder"} and parts[index + 1] in {"conv_in", "conv_out"}:
                family = parts[index + 1]
                suffix = ".".join(parts[index + 2 :]) or "__self__"
                return {
                    "subsystem": subsystem,
                    "family": family,
                    "layer_id": f"vae:{part}.{family}",
                    "parameter_suffix": suffix,
                    "component_scope": scope,
                }
        for singleton in ("quant_conv", "post_quant_conv"):
            if singleton in parts:
                singleton_index = parts.index(singleton)
                suffix = ".".join(parts[singleton_index + 1 :]) or "__self__"
                return {
                    "subsystem": subsystem,
                    "family": singleton,
                    "layer_id": f"vae:{singleton}",
                    "parameter_suffix": suffix,
                    "component_scope": scope,
                }

    local_key = key
    scope_prefix = f"{scope}."
    if key.startswith(scope_prefix):
        local_key = key[len(scope_prefix) :]
    return {
        "subsystem": subsystem,
        "family": "__global__",
        "layer_id": f"{scope}:__global__",
        "parameter_suffix": local_key or "__self__",
        "component_scope": scope,
    }


def build_layer_inventory(tensors: dict[str, dict[str, Any]]) -> dict[str, Any]:
    layers: dict[str, dict[str, Any]] = {}
    layer_counts_by_subsystem: Counter[str] = Counter()
    layer_counts_by_family: Counter[str] = Counter()

    for key, tensor in tensors.items():
        descriptor = normalize_layer_descriptor(key, tensor)
        layer_id = descriptor["layer_id"]
        layer = layers.setdefault(
            layer_id,
            {
                "subsystem": descriptor["subsystem"],
                "family": descriptor["family"],
                "component_scope": descriptor["component_scope"],
                "tensor_count": 0,
                "total_tensor_bytes": 0,
                "sample_parameter_suffixes": [],
                "parameters": {},
            },
        )
        parameter_suffix = descriptor["parameter_suffix"]
        layer["tensor_count"] += 1
        layer["total_tensor_bytes"] += int(tensor["payload_nbytes"])
        if parameter_suffix not in layer["sample_parameter_suffixes"] and len(layer["sample_parameter_suffixes"]) < 6:
            layer["sample_parameter_suffixes"].append(parameter_suffix)
        layer["parameters"][parameter_suffix] = {
            "shape": tensor["shape"],
            "dtype": tensor["dtype"],
            "payload_nbytes": tensor["payload_nbytes"],
            "component": tensor["component"],
            "raw_name": tensor.get("raw_name", key),
        }

    for layer_id, layer in layers.items():
        layer["parameter_count"] = len(layer["parameters"])
        layer_counts_by_subsystem[layer["subsystem"]] += 1
        layer_counts_by_family[f"{layer['subsystem']}:{layer['family']}"] += 1
        layer["layer_id"] = layer_id

    return {
        "total_normalized_layers": len(layers),
        "layer_counts_by_subsystem": dict(layer_counts_by_subsystem),
        "layer_counts_by_family": dict(layer_counts_by_family),
        "layers": dict(sorted(layers.items())),
    }


def summarize_layer_inventory(inventory: dict[str, Any]) -> dict[str, Any]:
    sample_layers = []
    for layer_id, layer in list(inventory["layers"].items())[:8]:
        sample_layers.append(
            {
                "layer_id": layer_id,
                "subsystem": layer["subsystem"],
                "family": layer["family"],
                "tensor_count": layer["tensor_count"],
                "total_tensor_bytes": layer["total_tensor_bytes"],
                "sample_parameter_suffixes": layer["sample_parameter_suffixes"],
            }
        )
    return {
        "total_normalized_layers": inventory["total_normalized_layers"],
        "layer_counts_by_subsystem": inventory["layer_counts_by_subsystem"],
        "layer_counts_by_family": inventory["layer_counts_by_family"],
        "sample_layers": sample_layers,
    }


def filter_tensors(
    tensors: dict[str, dict[str, Any]],
    predicate: Callable[[str], bool] | None = None,
) -> set[str]:
    return {key for key in tensors if predicate is None or predicate(key)}


def compare_tensor_sets(
    left_tensors: dict[str, dict[str, Any]],
    right_tensors: dict[str, dict[str, Any]],
    predicate: Callable[[str], bool] | None = None,
) -> dict[str, Any]:
    left_keys = filter_tensors(left_tensors, predicate)
    right_keys = filter_tensors(right_tensors, predicate)
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


def classify_structural_compatibility(stats: dict[str, Any], force_incompatible: bool = False) -> str:
    if force_incompatible or stats["shared_key_count"] == 0:
        return "incompatible"
    if stats["shape_mismatch_count"] == 0 and stats["missing_key_count"] == 0:
        return "direct-merge"
    return "adapter-only"


def classify_merge_strategy(
    stats: dict[str, Any],
    preferred_mode: str | None = None,
    force_incompatible: bool = False,
) -> str:
    if force_incompatible or stats["shared_key_count"] == 0:
        return "incompatible"
    if preferred_mode == "adapter-only":
        return "adapter-only"
    if preferred_mode == "delta-merge":
        return "delta-merge"
    if stats["shape_mismatch_count"] == 0 and stats["missing_key_count"] == 0:
        return "direct-merge"
    if stats["shared_ratio"] >= 0.75 and stats["shape_mismatch_count"] <= max(4, int(stats["shared_key_count"] * 0.02)):
        return "delta-merge"
    return "adapter-only"


def build_subsystem_result(
    subsystem: str,
    models: list[str],
    stats: dict[str, Any],
    reason: str,
    preferred_mode: str | None = None,
    force_incompatible: bool = False,
) -> dict[str, Any]:
    structural = classify_structural_compatibility(stats, force_incompatible=force_incompatible)
    strategy = classify_merge_strategy(stats, preferred_mode=preferred_mode, force_incompatible=force_incompatible)
    return {
        "subsystem": subsystem,
        "models": models,
        "classification": strategy,
        "structural_compatibility": structural,
        "recommended_merge_strategy": strategy,
        "reason": reason,
        "evidence": stats,
    }


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
                canonical_name = canonical_tensor_name(tensor_name, shard_component)
                if canonical_name in tensors:
                    raise Stage1AnalysisError(
                        f"Duplicate canonical tensor name {canonical_name} in snapshot {snapshot}. "
                        f"Conflicting raw tensor name: {tensor_name}"
                    )
                tensors[canonical_name] = {
                    **tensor,
                    "component": shard_component,
                    "relative_path": str(Path(shard["path"]).relative_to(snapshot)),
                    "raw_name": tensor_name,
                    "comparison_key": canonical_name,
                }
                local_tensor_count += 1
        component_tensor_counts[component_name] = local_tensor_count

    config_payloads = load_component_json_files(snapshot)
    vae = infer_vae_channels(config_payloads, tensors)
    rope = infer_rope_mode(config_payloads, tensors)
    layer_inventory = build_layer_inventory(tensors)
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
        "layer_inventory": layer_inventory,
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


def collect_snapshot_inventory(
    hf_home: str | Path | None,
    cache_alias_map: dict[str, str],
) -> dict[str, dict[str, str]]:
    inventory: dict[str, dict[str, str]] = {}
    for alias in MODEL_ORDER:
        if alias not in cache_alias_map:
            raise Stage1AnalysisError(f"Missing cache alias mapping for model alias: {alias}")
        snapshot = resolve_cache_snapshot(hf_home, cache_alias_map[alias])
        inventory[alias] = {
            "cache_dir_name": str(snapshot["cache_dir_name"]),
            "commit": str(snapshot["commit"]),
            "snapshot_path": str(snapshot["snapshot"]),
        }
    return inventory


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
        "normalized_layer_count": manifest["layer_inventory"]["total_normalized_layers"],
        "layer_counts_by_subsystem": manifest["layer_inventory"]["layer_counts_by_subsystem"],
        "layer_counts_by_family": manifest["layer_inventory"]["layer_counts_by_family"],
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
        predicate=lambda key: subsystem_for_tensor(
            key, base["tensors"].get(key, layered["tensors"].get(key, {}))
        )
        == "vae",
    )
    incompatible = base["vae"]["label"] != layered["vae"]["label"]
    base_channels = f"{base['vae']['input_channels']}->{base['vae']['output_channels']}"
    layered_channels = f"{layered['vae']['input_channels']}->{layered['vae']['output_channels']}"
    return build_subsystem_result(
        subsystem="vae",
        models=["qwen-image-base", "qwen-image-layered"],
        stats=stats,
        reason=f"Base VAE channels {base['vae']['label']} ({base_channels}) vs layered VAE channels {layered['vae']['label']} ({layered_channels}).",
        force_incompatible=incompatible,
    )


def probe_rope_compatibility(manifests: dict[str, dict[str, Any]]) -> dict[str, Any]:
    foundation = manifests["qwen-image-2512"]
    layered = manifests["qwen-image-layered"]
    stats = compare_tensor_sets(
        foundation["tensors"],
        layered["tensors"],
        predicate=lambda key: subsystem_for_tensor(
            key, foundation["tensors"].get(key, layered["tensors"].get(key, {}))
        )
        == "rope",
    )
    preferred = "adapter-only" if foundation["rope"]["label"] != layered["rope"]["label"] else None
    return build_subsystem_result(
        subsystem="rope",
        models=["qwen-image-2512", "qwen-image-layered"],
        stats=stats,
        reason=f"Foundation rope hint `{foundation['rope']['label']}` vs layered rope hint `{layered['rope']['label']}`.",
        preferred_mode=preferred,
    )


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


def compare_layer_sets(
    left_layers: dict[str, dict[str, Any]],
    right_layers: dict[str, dict[str, Any]],
    subsystem: str | None = None,
) -> dict[str, Any]:
    if subsystem:
        left_layer_map = {
            layer_id: layer for layer_id, layer in left_layers.items() if layer["subsystem"] == subsystem
        }
        right_layer_map = {
            layer_id: layer for layer_id, layer in right_layers.items() if layer["subsystem"] == subsystem
        }
    else:
        left_layer_map = dict(left_layers)
        right_layer_map = dict(right_layers)

    left_ids = set(left_layer_map)
    right_ids = set(right_layer_map)
    shared_ids = sorted(left_ids & right_ids)
    left_only_ids = sorted(left_ids - right_ids)
    right_only_ids = sorted(right_ids - left_ids)
    exact = 0
    partial = 0
    shape_mismatched_layers = 0
    dtype_mismatched_layers = 0
    top_divergent_layers: list[dict[str, Any]] = []

    for layer_id in shared_ids:
        left_params = left_layer_map[layer_id]["parameters"]
        right_params = right_layer_map[layer_id]["parameters"]
        left_suffixes = set(left_params)
        right_suffixes = set(right_params)
        shared_suffixes = sorted(left_suffixes & right_suffixes)
        left_only_suffixes = sorted(left_suffixes - right_suffixes)
        right_only_suffixes = sorted(right_suffixes - left_suffixes)
        shape_mismatches = sorted(
            suffix for suffix in shared_suffixes if left_params[suffix]["shape"] != right_params[suffix]["shape"]
        )
        dtype_mismatches = sorted(
            suffix for suffix in shared_suffixes if left_params[suffix]["dtype"] != right_params[suffix]["dtype"]
        )
        if not left_only_suffixes and not right_only_suffixes and not shape_mismatches and not dtype_mismatches:
            exact += 1
            continue

        partial += 1
        if shape_mismatches:
            shape_mismatched_layers += 1
        if dtype_mismatches:
            dtype_mismatched_layers += 1

        reasons: list[str] = []
        if left_only_suffixes:
            reasons.append(f"left-only params={len(left_only_suffixes)}")
        if right_only_suffixes:
            reasons.append(f"right-only params={len(right_only_suffixes)}")
        if shape_mismatches:
            reasons.append(f"shape mismatches={len(shape_mismatches)}")
        if dtype_mismatches:
            reasons.append(f"dtype mismatches={len(dtype_mismatches)}")
        top_divergent_layers.append(
            {
                "layer_id": layer_id,
                "subsystem": left_layer_map[layer_id]["subsystem"],
                "family": left_layer_map[layer_id]["family"],
                "reason": ", ".join(reasons),
                "left_parameter_count": len(left_suffixes),
                "right_parameter_count": len(right_suffixes),
                "shared_parameter_count": len(shared_suffixes),
                "left_only_parameter_count": len(left_only_suffixes),
                "right_only_parameter_count": len(right_only_suffixes),
                "shape_mismatch_count": len(shape_mismatches),
                "dtype_mismatch_count": len(dtype_mismatches),
                "sample_shape_mismatches": shape_mismatches[:5],
                "left_only_parameter_samples": left_only_suffixes[:5],
                "right_only_parameter_samples": right_only_suffixes[:5],
            }
        )

    for layer_id in left_only_ids:
        top_divergent_layers.append(
            {
                "layer_id": layer_id,
                "subsystem": left_layer_map[layer_id]["subsystem"],
                "family": left_layer_map[layer_id]["family"],
                "reason": "left-only layer",
                "left_parameter_count": left_layer_map[layer_id]["parameter_count"],
                "right_parameter_count": 0,
                "shared_parameter_count": 0,
                "left_only_parameter_count": left_layer_map[layer_id]["parameter_count"],
                "right_only_parameter_count": 0,
                "shape_mismatch_count": 0,
                "dtype_mismatch_count": 0,
                "sample_shape_mismatches": [],
                "left_only_parameter_samples": sorted(left_layer_map[layer_id]["parameters"].keys())[:5],
                "right_only_parameter_samples": [],
            }
        )

    for layer_id in right_only_ids:
        top_divergent_layers.append(
            {
                "layer_id": layer_id,
                "subsystem": right_layer_map[layer_id]["subsystem"],
                "family": right_layer_map[layer_id]["family"],
                "reason": "right-only layer",
                "left_parameter_count": 0,
                "right_parameter_count": right_layer_map[layer_id]["parameter_count"],
                "shared_parameter_count": 0,
                "left_only_parameter_count": 0,
                "right_only_parameter_count": right_layer_map[layer_id]["parameter_count"],
                "shape_mismatch_count": 0,
                "dtype_mismatch_count": 0,
                "sample_shape_mismatches": [],
                "left_only_parameter_samples": [],
                "right_only_parameter_samples": sorted(right_layer_map[layer_id]["parameters"].keys())[:5],
            }
        )

    top_divergent_layers = sorted(
        top_divergent_layers,
        key=lambda item: (
            -(item["shape_mismatch_count"] * 10 + item["left_only_parameter_count"] + item["right_only_parameter_count"]),
            item["layer_id"],
        ),
    )[:10]

    union_count = len(left_ids | right_ids)
    return {
        "left_layer_count": len(left_ids),
        "right_layer_count": len(right_ids),
        "shared_layer_count": len(shared_ids),
        "left_only_layer_count": len(left_only_ids),
        "right_only_layer_count": len(right_only_ids),
        "exact_layer_match_count": exact,
        "partial_layer_match_count": partial,
        "shape_mismatched_layer_count": shape_mismatched_layers,
        "dtype_mismatched_layer_count": dtype_mismatched_layers,
        "layer_shared_ratio": round((len(shared_ids) / union_count) if union_count else 0.0, 4),
        "top_divergent_layers": top_divergent_layers,
    }


def build_layer_pairwise(manifests: dict[str, dict[str, Any]]) -> dict[str, Any]:
    pairs: dict[str, Any] = {}
    for left, right in combinations(MODEL_ORDER, 2):
        left_inventory = manifests[left]["layer_inventory"]
        right_inventory = manifests[right]["layer_inventory"]
        alias = pair_alias(left, right)
        pairs[alias] = {
            "models": [left, right],
            "overall": compare_layer_sets(left_inventory["layers"], right_inventory["layers"]),
            "by_subsystem": {
                subsystem: compare_layer_sets(
                    left_inventory["layers"],
                    right_inventory["layers"],
                    subsystem=subsystem,
                )
                for subsystem in SUBSYSTEM_ORDER
            },
        }
    return pairs
