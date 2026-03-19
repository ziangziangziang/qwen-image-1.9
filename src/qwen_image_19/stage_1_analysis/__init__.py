from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from itertools import combinations
import json
import math
import os
from pathlib import Path
import platform
import shutil
import socket
import struct
import time
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

SUBSYSTEM_ORDER = (
    "mmdit_backbone",
    "text_encoder",
    "vae",
    "rope",
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

ROADMAP_LAYER_PAIRS = (
    ("qwen-image-2512", "qwen-image-edit-2511"),
    ("qwen-image-base", "qwen-image-layered"),
    ("qwen-image-2512", "qwen-image-layered"),
)

WEIGHT_DELTA_THRESHOLD = 1e-6
WEIGHT_ANALYSIS_PAIRWISE = {
    "foundation_vs_edit": ("qwen-image-2512", "qwen-image-edit-2511"),
    "base_vs_layered": ("qwen-image-base", "qwen-image-layered"),
    "foundation_vs_layered": ("qwen-image-2512", "qwen-image-layered"),
}

ESTIMATE_READ_GBPS = {"low": 0.9, "typical": 1.8, "high": 3.2}
ESTIMATE_REDUCE_GBPS = {"low": 7.0, "typical": 14.0, "high": 28.0}

SHORT_ALIAS = {
    "qwen-image-base": "base",
    "qwen-image-2512": "2512",
    "qwen-image-edit-2511": "edit-2511",
    "qwen-image-layered": "layered",
}


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
    predicate: callable | None = None,
) -> set[str]:
    return {key for key in tensors if predicate is None or predicate(key)}


def compare_tensor_sets(
    left_tensors: dict[str, dict[str, Any]],
    right_tensors: dict[str, dict[str, Any]],
    predicate: callable | None = None,
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
                    "| --- | --- | --- | --- | --- |",
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
    return f"""# Stage 1 DNA Report

Canonical Stage 1 report: [stage-1/summary.md](stage-1/summary.md)

This file is kept as a compatibility shim. Open `{target_dir / 'summary.md'}` for the full analyst-facing report.
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


def analyze(
    artifact_dir: str | Path | None = None,
    remote_config: str | None = None,
    model_dir: str | Path | None = None,
    dry_run: bool = False,
    cache_dir: str | None = None,
    hf_home: str | Path | None = None,
    cache_map_config: str | Path | None = None,
) -> dict[str, Any]:
    target_dir = Path(artifact_dir) if artifact_dir else repo_root() / "reports" / "stage-1"
    artifact_paths = stage1_artifact_paths(target_dir)
    compatibility_shims = build_stage1_compatibility_shims(target_dir)
    total_start = time.perf_counter()
    phase_seconds: dict[str, float] = {}

    phase_start = time.perf_counter()
    metadata_models = load_model_inventory(model_dir)
    cache_alias_map = load_cache_alias_map(cache_map_config)
    remote_context = default_remote_context(remote_config)
    if cache_dir:
        remote_context["cache_dir"] = cache_dir
    resolved_hf_home = resolve_hf_home(hf_home)
    phase_seconds["setup_context"] = time.perf_counter() - phase_start

    phase_start = time.perf_counter()
    snapshot_inventory = collect_snapshot_inventory(resolved_hf_home, cache_alias_map)
    phase_seconds["cache_snapshot_discovery"] = time.perf_counter() - phase_start
    hardware_snapshot = build_hardware_snapshot(resolved_hf_home, target_dir, snapshot_inventory)

    phase_start = time.perf_counter()
    manifests = inspect_cache_models(resolved_hf_home, metadata_models, cache_alias_map)
    phase_seconds["structural_manifest_build"] = time.perf_counter() - phase_start

    phase_start = time.perf_counter()
    matrix = build_compatibility_matrix(manifests)
    layer_analysis = build_layer_analysis_payload(manifests, matrix)
    phase_seconds["pairwise_structural_layer"] = time.perf_counter() - phase_start

    phase_start = time.perf_counter()
    weight_result = build_weight_pairwise_analysis(manifests)
    if len(weight_result) == 2:
        weight_pairwise, weight_analysis = weight_result
        value_runtime_profile = {"pair_value_pass_seconds": {}}
    else:
        weight_pairwise, weight_analysis, value_runtime_profile = weight_result
    phase_seconds["value_level_weight_comparison"] = time.perf_counter() - phase_start

    matrix["weight_analysis_available"] = True
    matrix["weight_thresholds"] = {"relative_l2_delta_low": WEIGHT_DELTA_THRESHOLD}
    matrix["weight_pairwise"] = weight_pairwise
    matrix["block_review_summary"] = build_block_review_summary(weight_pairwise)

    figure_refs = {
        "component_overview": f"figures/{artifact_paths['component_overview_png'].name}",
        "pairwise_comparison": f"figures/{artifact_paths['pairwise_comparison_png'].name}",
        "layer_sharing_heatmap": f"figures/{artifact_paths['layer_sharing_heatmap_png'].name}",
        "layer_sharing_bars": f"figures/{artifact_paths['layer_sharing_bars_png'].name}",
    }
    matrix["artifact_layout"] = {
        "summary_markdown": str(artifact_paths["summary_markdown"].name),
        "matrix_json": str(artifact_paths["matrix_json"].name),
        "layer_analysis_json": str(artifact_paths["layer_analysis_json"].name),
        "weight_analysis_json": str(artifact_paths["weight_analysis_json"].name),
        "figures": figure_refs,
    }
    matrix["layer_visuals"] = figure_refs

    result = {
        "stage": "stage1",
        "mode": "dry-run" if dry_run else "write",
        "hf_home": str(resolved_hf_home),
        "cache_alias_map": cache_alias_map,
        "matrix": matrix,
        "artifact_dir": str(target_dir),
        "artifact_paths": {key: str(value) for key, value in artifact_paths.items() if key != "figures_dir"},
        "compatibility_shims": {key: str(value) for key, value in compatibility_shims.items()},
    }

    if dry_run:
        phase_seconds["figure_generation"] = 0.0
        phase_seconds["report_json_write"] = 0.0
        phase_seconds["total_wall"] = time.perf_counter() - total_start
        matrix["resource_accounting"] = build_resource_accounting(
            hardware_snapshot,
            phase_seconds,
            manifests,
            weight_pairwise,
            value_runtime_profile=value_runtime_profile,
        )
        weight_analysis["runtime_profile"] = {
            "pair_value_pass_seconds": matrix["resource_accounting"]["timing"].get("value_pair_seconds", {}),
            "roadmap_pair_bytes": matrix["resource_accounting"]["workload"].get("roadmap_pairs", {}),
        }
        report = render_dna_report(matrix, remote_context, resolved_hf_home, cache_alias_map, figure_refs)
        result["report_preview"] = report
        result["layer_analysis"] = layer_analysis
        result["weight_analysis"] = weight_analysis
        result["terminal_summary"] = build_stage1_terminal_summary(result)
        return result

    phase_start = time.perf_counter()
    generate_stage1_figures(matrix, artifact_paths)
    phase_seconds["figure_generation"] = time.perf_counter() - phase_start

    write_start = time.perf_counter()
    phase_seconds["report_json_write"] = 0.0
    phase_seconds["total_wall"] = time.perf_counter() - total_start
    matrix["resource_accounting"] = build_resource_accounting(
        hardware_snapshot,
        phase_seconds,
        manifests,
        weight_pairwise,
        value_runtime_profile=value_runtime_profile,
    )
    weight_analysis["runtime_profile"] = {
        "pair_value_pass_seconds": matrix["resource_accounting"]["timing"].get("value_pair_seconds", {}),
        "roadmap_pair_bytes": matrix["resource_accounting"]["workload"].get("roadmap_pairs", {}),
    }
    report = render_dna_report(matrix, remote_context, resolved_hf_home, cache_alias_map, figure_refs)
    write_json(artifact_paths["matrix_json"], matrix)
    write_json(artifact_paths["layer_analysis_json"], layer_analysis)
    write_json(artifact_paths["weight_analysis_json"], weight_analysis)
    write_text(artifact_paths["summary_markdown"], report)
    written = [
        str(artifact_paths["matrix_json"]),
        str(artifact_paths["layer_analysis_json"]),
        str(artifact_paths["weight_analysis_json"]),
        str(artifact_paths["summary_markdown"]),
        str(artifact_paths["component_overview_png"]),
        str(artifact_paths["pairwise_comparison_png"]),
        str(artifact_paths["layer_sharing_heatmap_png"]),
        str(artifact_paths["layer_sharing_bars_png"]),
    ]
    if compatibility_shims:
        write_json(compatibility_shims["legacy_matrix_json"], matrix)
        write_text(compatibility_shims["legacy_report_md"], render_stage1_compatibility_stub(target_dir))
        written.extend(str(path) for path in compatibility_shims.values())
    phase_seconds["report_json_write"] = time.perf_counter() - write_start
    phase_seconds["total_wall"] = time.perf_counter() - total_start
    matrix["resource_accounting"] = build_resource_accounting(
        hardware_snapshot,
        phase_seconds,
        manifests,
        weight_pairwise,
        value_runtime_profile=value_runtime_profile,
    )
    weight_analysis["runtime_profile"] = {
        "pair_value_pass_seconds": matrix["resource_accounting"]["timing"].get("value_pair_seconds", {}),
        "roadmap_pair_bytes": matrix["resource_accounting"]["workload"].get("roadmap_pairs", {}),
    }
    report = render_dna_report(matrix, remote_context, resolved_hf_home, cache_alias_map, figure_refs)
    write_json(artifact_paths["matrix_json"], matrix)
    write_json(artifact_paths["weight_analysis_json"], weight_analysis)
    write_text(artifact_paths["summary_markdown"], report)
    if compatibility_shims:
        write_json(compatibility_shims["legacy_matrix_json"], matrix)
        write_text(compatibility_shims["legacy_report_md"], render_stage1_compatibility_stub(target_dir))
    result["written"] = written
    result["terminal_summary"] = build_stage1_terminal_summary(result)
    return result
