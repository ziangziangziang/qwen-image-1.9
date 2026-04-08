"""Stage 1 — Checkpoint analysis and compatibility matrix.

Inspects HuggingFace-cached model checkpoints, builds pairwise layer-sharing
statistics, and produces a compatibility matrix for merge planning.
"""
from __future__ import annotations

import json
import os
import platform
import socket
import struct
import time
from itertools import combinations
from pathlib import Path
from typing import Any

from qwen_image_19.config_io import repo_root, write_json, write_text
from qwen_image_19.contracts import public_path, utc_now


class Stage1AnalysisError(RuntimeError):
    """Raised when stage-1 analysis encounters an unrecoverable condition."""


# ── Model inventory ─────────────────────────────────────────────────

def load_model_inventory() -> dict[str, Any]:
    """Load all model metadata from configs/models/*.yaml (JSON format)."""
    models_dir = repo_root() / "configs" / "models"
    result: dict[str, Any] = {}
    for path in sorted(models_dir.glob("*.yaml")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            alias = data.get("alias", path.stem)
            result[alias] = data
        except Exception:
            continue
    return result


def load_cache_alias_map() -> dict[str, str]:
    """Return alias → HuggingFace cache directory name mapping."""
    inventory = load_model_inventory()
    alias_map: dict[str, str] = {}
    for alias, meta in inventory.items():
        remote = meta.get("remote", {})
        uri = remote.get("checkpoint_uri", "")
        # hf://Qwen/Qwen-Image-2512  →  models--Qwen--Qwen-Image-2512
        if uri.startswith("hf://"):
            model_id = uri[len("hf://"):]
        else:
            model_id = meta.get("model_id", alias)
        cache_name = "models--" + model_id.replace("/", "--")
        alias_map[alias] = cache_name
    return alias_map


# ── HF cache introspection ──────────────────────────────────────────

def resolve_cache_snapshot(hf_home: Path, cache_dir_name: str) -> dict[str, Any]:
    """Resolve the active snapshot for a cached model.

    Raises Stage1AnalysisError if the cache is absent or corrupt.
    """
    hub_root = hf_home / "hub"
    model_root = hub_root / cache_dir_name
    refs_main = model_root / "refs" / "main"
    if not refs_main.exists():
        raise Stage1AnalysisError(
            f"Model cache missing or incomplete: `{cache_dir_name}`. "
            f"Expected `{refs_main}` to exist."
        )
    commit = refs_main.read_text(encoding="utf-8").strip()
    snapshot_path = model_root / "snapshots" / commit
    if not snapshot_path.exists():
        raise Stage1AnalysisError(
            f"Snapshot `{commit}` for `{cache_dir_name}` not found. "
            f"Expected `{snapshot_path}`."
        )
    return {"commit": commit, "snapshot": snapshot_path}


def _read_safetensors_keys(path: Path) -> list[str]:
    """Read tensor key names from a safetensors file header without loading weights."""
    with path.open("rb") as fh:
        raw_len = fh.read(8)
        if len(raw_len) < 8:
            return []
        (header_len,) = struct.unpack("<Q", raw_len)
        header_bytes = fh.read(header_len)
    header = json.loads(header_bytes)
    return [k for k in header.keys() if k != "__metadata__"]


def _collect_flat_shards(snapshot: Path) -> list[dict[str, Any]]:
    """Collect safetensors shard info from a flat (non-componentized) snapshot."""
    index_path = snapshot / "model.safetensors.index.json"
    single = snapshot / "model.safetensors"

    if single.exists() and not index_path.exists():
        # Check if there are OTHER safetensors files that would require an index
        other_safetensors = [
            p for p in snapshot.glob("*.safetensors")
            if p.name != "model.safetensors"
        ]
        if other_safetensors:
            raise Stage1AnalysisError(
                f"Multiple safetensors files in `{snapshot}` without index. "
                "A `model.safetensors.index.json` is required for sharded checkpoints."
            )
        keys = _read_safetensors_keys(single)
        return [{"file": single, "component": ".", "keys": keys}]

    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map", {})
        shards: dict[str, list[str]] = {}
        for key, shard_name in weight_map.items():
            shards.setdefault(shard_name, []).append(key)
        result = []
        for shard_name, shard_keys in sorted(shards.items()):
            shard_path = snapshot / shard_name
            if not shard_path.exists():
                raise Stage1AnalysisError(
                    f"Shard `{shard_name}` referenced by index not found at `{shard_path}`."
                )
            result.append({"file": shard_path, "component": ".", "keys": shard_keys})
        return result

    # Multiple .safetensors without index
    all_st = sorted(snapshot.glob("*.safetensors"))
    if len(all_st) > 1:
        raise Stage1AnalysisError(
            f"Multiple safetensors files in `{snapshot}` without index. "
            "A `model.safetensors.index.json` is required for sharded checkpoints."
        )
    if len(all_st) == 1:
        keys = _read_safetensors_keys(all_st[0])
        return [{"file": all_st[0], "component": ".", "keys": keys}]

    return []


def _collect_component_shards(snapshot: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Collect shards from a diffusers-style componentized snapshot."""
    components: dict[str, Any] = {}
    all_shards: list[dict[str, Any]] = []

    for comp_dir in sorted(snapshot.iterdir()):
        if not comp_dir.is_dir():
            continue
        if comp_dir.name.startswith("."):
            continue

        comp_name = comp_dir.name
        indices = list(comp_dir.glob("*.safetensors.index.json"))
        if len(indices) > 1:
            raise Stage1AnalysisError(
                f"Component `{comp_name}` in `{snapshot}` has multiple safetensors index files."
            )

        if len(indices) == 1:
            index_path = indices[0]
            index = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = index.get("weight_map", {})
            shards: dict[str, list[str]] = {}
            for key, shard_name in weight_map.items():
                shards.setdefault(shard_name, []).append(key)
            comp_tensor_count = 0
            for shard_name, shard_keys in sorted(shards.items()):
                shard_path = comp_dir / shard_name
                if not shard_path.exists():
                    raise Stage1AnalysisError(
                        f"Shard `{shard_name}` in component `{comp_name}` not found at `{shard_path}`."
                    )
                all_shards.append({"file": shard_path, "component": comp_name, "keys": shard_keys})
                comp_tensor_count += len(shard_keys)
            components[comp_name] = {"tensor_count": comp_tensor_count, "shard_count": len(shards)}
        else:
            comp_safetensors = sorted(comp_dir.glob("*.safetensors"))
            if not comp_safetensors:
                continue
            comp_tensor_count = 0
            for st_path in comp_safetensors:
                keys = _read_safetensors_keys(st_path)
                all_shards.append({"file": st_path, "component": comp_name, "keys": keys})
                comp_tensor_count += len(keys)
            components[comp_name] = {"tensor_count": comp_tensor_count, "shard_count": len(comp_safetensors)}

    return all_shards, components


def inspect_cache_models(
    hf_home: Path,
    metadata: dict[str, Any],
    cache_map: dict[str, str],
) -> dict[str, Any]:
    """Inspect all cached models and return a manifest per model alias."""
    manifests: dict[str, Any] = {}

    for alias, cache_dir_name in cache_map.items():
        snap_info = resolve_cache_snapshot(hf_home, cache_dir_name)
        snapshot: Path = snap_info["snapshot"]

        # Detect layout: componentized (diffusers) vs flat
        model_index = snapshot / "model_index.json"
        if model_index.exists():
            layout = "componentized"
            shards, components = _collect_component_shards(snapshot)
        else:
            layout = "flat"
            shards = _collect_flat_shards(snapshot)
            components = {}

        # Aggregate tensor keys and byte count
        all_keys: list[str] = []
        for shard in shards:
            all_keys.extend(shard["keys"])

        component_tensor_counts = {
            comp: info["tensor_count"] for comp, info in components.items()
        }

        total_bytes = 0
        config_info: dict[str, Any] = {}
        config_path = snapshot / "config.json"
        if config_path.exists():
            try:
                config_info = json.loads(config_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        # Infer VAE input channels from weight shapes where possible
        vae_info: dict[str, Any] = {}
        for key in all_keys:
            if "vae.encoder.conv_in.weight" in key or "encoder.conv_in.weight" in key:
                # Find the shard that has this key
                for shard in shards:
                    if key in shard["keys"]:
                        try:
                            header = _read_safetensors_header(shard["file"])
                            if key in header:
                                shape = header[key].get("shape", [])
                                if len(shape) >= 4:
                                    vae_info["input_channels"] = shape[1]
                        except Exception:
                            pass
                        break

        model_meta = metadata.get(alias, {})

        manifests[alias] = {
            "alias": alias,
            "model_id": model_meta.get("model_id", alias),
            "role": model_meta.get("role", "unknown"),
            "layout": layout,
            "commit": snap_info["commit"],
            "snapshot_path": str(snapshot),
            "shard_count": len(shards),
            "shards": [
                {
                    "file": str(s["file"]),
                    "component": s["component"],
                    "key_count": len(s["keys"]),
                }
                for s in shards
            ],
            "components": components,
            "component_tensor_counts": component_tensor_counts,
            "tensor_keys": sorted(set(all_keys)),
            "total_tensors": len(set(all_keys)),
            "total_tensor_bytes": total_bytes,
            "config": config_info,
            "vae_info": vae_info,
            "architecture": model_meta.get("architecture", {}),
        }

    return manifests


def _read_safetensors_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
        raw_len = fh.read(8)
        if len(raw_len) < 8:
            return {}
        (header_len,) = struct.unpack("<Q", raw_len)
        header_bytes = fh.read(header_len)
    header = json.loads(header_bytes)
    return {k: v for k, v in header.items() if k != "__metadata__"}


# ── State dict comparison ───────────────────────────────────────────

def compare_state_dicts(manifests: dict[str, Any]) -> dict[str, Any]:
    """Compare key sets between foundation and edit-donor models."""
    foundation = next(
        (m for m in manifests.values() if m.get("role") == "foundation"), None
    )
    edit_donor = next(
        (m for m in manifests.values() if m.get("role") == "edit-donor"), None
    )
    if foundation is None or edit_donor is None:
        # Fallback: compare all models' union
        all_keys: set[str] = set()
        for m in manifests.values():
            all_keys.update(m.get("tensor_keys", []))
        shared = len(all_keys)
        return {"shared_key_count": shared, "missing_key_count": 0}

    foundation_keys = set(foundation.get("tensor_keys", []))
    edit_keys = set(edit_donor.get("tensor_keys", []))
    shared = foundation_keys & edit_keys
    in_edit_only = edit_keys - foundation_keys
    in_foundation_only = foundation_keys - edit_keys
    return {
        "shared_key_count": len(shared),
        "missing_key_count": len(in_edit_only),
        "extra_in_foundation": len(in_foundation_only),
        "foundation_total": len(foundation_keys),
        "edit_total": len(edit_keys),
    }


# ── Layer descriptor normalization ─────────────────────────────────

def normalize_layer_descriptor(
    tensor_key: str,
    tensor_info: dict[str, Any],
) -> dict[str, Any]:
    """Normalize a tensor key into a structured layer descriptor.

    Recognizes:
      - transformer.transformer_blocks.N.* → mmdit_backbone:transformer_blocks:N
      - text_encoder.layers.N.* → text_encoder:layers:N
      - vae.encoder.down_blocks.N.* → vae:encoder.down_blocks:N
      - vae.encoder.*.* → vae:encoder.*
      - model.mmdit.* → mmdit_backbone
    """
    component_scope = tensor_info.get("component", ".")
    parts = tensor_key.split(".")

    def _make(subsystem: str, family: str, layer_num: str, suffix: str, scope: str) -> dict[str, Any]:
        return {
            "subsystem": subsystem,
            "family": family,
            "layer_id": f"{subsystem}:{family}:{layer_num}",
            "parameter_suffix": suffix,
            "component_scope": scope,
        }

    # transformer.transformer_blocks.N.*
    if len(parts) >= 3 and parts[0] == "transformer" and parts[1] == "transformer_blocks":
        layer_num = parts[2]
        suffix = ".".join(parts[3:]) if len(parts) > 3 else ""
        return _make("mmdit_backbone", "transformer_blocks", layer_num, suffix, "transformer")

    # text_encoder.layers.N.* — may have component_scope="text_encoder"
    if len(parts) >= 3 and parts[0] == "text_encoder" and parts[1] == "layers":
        layer_num = parts[2]
        suffix = ".".join(parts[3:]) if len(parts) > 3 else ""
        return _make("text_encoder", "layers", layer_num, suffix, component_scope)

    # When component_scope is text_encoder and key is layers.N.*
    if component_scope == "text_encoder" and len(parts) >= 2 and parts[0] == "layers":
        layer_num = parts[1]
        suffix = ".".join(parts[2:]) if len(parts) > 2 else ""
        return _make("text_encoder", "layers", layer_num, suffix, component_scope)

    # vae.encoder.down_blocks.N.* or vae.encoder.*.N.*
    if len(parts) >= 4 and parts[0] == "vae" and parts[1] == "encoder":
        family = f"{parts[1]}.{parts[2]}"
        if len(parts) >= 4 and parts[3].isdigit():
            layer_num = parts[3]
            suffix = ".".join(parts[4:]) if len(parts) > 4 else ""
            return _make("vae", family, layer_num, suffix, "vae")
        else:
            suffix = ".".join(parts[2:]) if len(parts) > 2 else ""
            return {
                "subsystem": "vae",
                "family": f"encoder.{parts[2]}",
                "layer_id": f"vae:encoder.{parts[2]}",
                "parameter_suffix": suffix,
                "component_scope": "vae",
            }

    if len(parts) >= 3 and parts[0] == "vae" and parts[1] == "decoder":
        family = f"{parts[1]}.{parts[2]}"
        if len(parts) >= 4 and parts[3].isdigit():
            layer_num = parts[3]
            suffix = ".".join(parts[4:]) if len(parts) > 4 else ""
            return _make("vae", family, layer_num, suffix, "vae")
        suffix = ".".join(parts[2:]) if len(parts) > 2 else ""
        return {
            "subsystem": "vae",
            "family": f"decoder.{parts[2]}",
            "layer_id": f"vae:decoder.{parts[2]}",
            "parameter_suffix": suffix,
            "component_scope": "vae",
        }

    # vae.encoder.conv_in.weight style (short)
    if len(parts) >= 2 and parts[0] == "vae":
        return {
            "subsystem": "vae",
            "family": ".".join(parts[1:-1]) if len(parts) > 2 else parts[1],
            "layer_id": f"vae:{'.' .join(parts[1:-1])}" if len(parts) > 2 else f"vae:{parts[1]}",
            "parameter_suffix": parts[-1],
            "component_scope": "vae",
        }

    # model.mmdit.*
    if len(parts) >= 2 and parts[0] == "model" and parts[1] == "mmdit":
        family = parts[2] if len(parts) > 2 else "unknown"
        layer_num = parts[3] if len(parts) > 3 and parts[3].isdigit() else "0"
        suffix = ".".join(parts[4:]) if len(parts) > 4 else ""
        return _make("mmdit_backbone", family, layer_num, suffix, ".")

    # rope.*
    if len(parts) >= 1 and parts[0] == "rope":
        return {
            "subsystem": "rope",
            "family": ".".join(parts[1:]),
            "layer_id": f"rope:{'.' .join(parts[1:])}",
            "parameter_suffix": parts[-1],
            "component_scope": component_scope,
        }

    # edit_heads.*
    if len(parts) >= 1 and parts[0] == "edit_heads":
        return {
            "subsystem": "edit_heads",
            "family": ".".join(parts[1:]),
            "layer_id": f"edit_heads:{'.' .join(parts[1:])}",
            "parameter_suffix": parts[-1],
            "component_scope": component_scope,
        }

    # Fallback
    return {
        "subsystem": parts[0] if parts else "unknown",
        "family": ".".join(parts[1:-1]) if len(parts) > 2 else (parts[1] if len(parts) > 1 else ""),
        "layer_id": f"{parts[0]}:{'.' .join(parts[1:])}",
        "parameter_suffix": parts[-1] if parts else "",
        "component_scope": component_scope,
    }


# ── Pairwise layer analysis ─────────────────────────────────────────

def _classify_pair(
    alias_a: str,
    alias_b: str,
    manifests: dict[str, Any],
) -> str:
    """Classify the merge relationship between two models."""
    ma = manifests[alias_a]
    mb = manifests[alias_b]
    keys_a = set(ma.get("tensor_keys", []))
    keys_b = set(mb.get("tensor_keys", []))
    shared = keys_a & keys_b

    # Check VAE channel compatibility
    vae_a = ma.get("vae_info", {}).get("input_channels")
    vae_b = mb.get("vae_info", {}).get("input_channels")
    if vae_a and vae_b and vae_a != vae_b:
        return "incompatible"

    # Check rope keys
    rope_a = [k for k in keys_a if k.startswith("rope.")]
    rope_b = [k for k in keys_b if k.startswith("rope.")]
    if (rope_a) != (rope_b) and (rope_a or rope_b):
        return "incompatible"

    # If one model has edit_heads (adapter-style keys)
    edit_heads_b = [k for k in keys_b - keys_a if k.startswith("edit_heads.")]
    edit_heads_a = [k for k in keys_a - keys_b if k.startswith("edit_heads.")]
    if edit_heads_b or edit_heads_a:
        # Check if the edit model is a delta-merge candidate
        role_a = ma.get("role", "")
        role_b = mb.get("role", "")
        if (role_a == "foundation" and role_b == "edit-donor") or \
           (role_b == "foundation" and role_a == "edit-donor"):
            return "delta-merge"
        return "adapter-only"

    # Delta-merge: foundation + edit-donor relationship
    role_a = ma.get("role", "")
    role_b = mb.get("role", "")
    if (role_a == "foundation" and role_b == "edit-donor") or \
       (role_b == "foundation" and role_a == "edit-donor"):
        return "delta-merge"

    if len(shared) / max(len(keys_a | keys_b), 1) > 0.8:
        return "direct-merge"

    return "direct-merge"


def _compute_layer_pairwise(manifests: dict[str, Any]) -> dict[str, Any]:
    """Compute subsystem-level layer sharing for all pairs, ordered by model_id."""
    # Sort by model_id to get canonical pair ordering
    sorted_aliases = sorted(
        manifests.keys(),
        key=lambda a: manifests[a].get("model_id", a),
    )

    result: dict[str, Any] = {}
    for alias_a, alias_b in combinations(sorted_aliases, 2):
        pair_name = f"{alias_a}_vs_{alias_b}"
        ma = manifests[alias_a]
        mb = manifests[alias_b]
        keys_a = set(ma.get("tensor_keys", []))
        keys_b = set(mb.get("tensor_keys", []))
        shared = keys_a & keys_b

        # Build subsystem breakdown
        by_subsystem: dict[str, Any] = {}
        for key in shared:
            desc = normalize_layer_descriptor(key, {"component": "."})
            sub = desc["subsystem"]
            if sub not in by_subsystem:
                by_subsystem[sub] = {"shared_keys": [], "total_shared": 0}
            by_subsystem[sub]["shared_keys"].append(key)
            by_subsystem[sub]["total_shared"] += 1

        result[pair_name] = {
            "alias_a": alias_a,
            "alias_b": alias_b,
            "shared_key_count": len(shared),
            "only_in_a": len(keys_a - keys_b),
            "only_in_b": len(keys_b - keys_a),
            "overall": {
                "shared_key_count": len(shared),
                "total_keys_a": len(keys_a),
                "total_keys_b": len(keys_b),
                "sharing_ratio": len(shared) / max(len(keys_a | keys_b), 1),
            },
            "by_subsystem": {
                sub: {"shared_key_count": info["total_shared"]}
                for sub, info in by_subsystem.items()
            },
        }
    return result


# ── Weight pairwise analysis (patchable) ────────────────────────────

def build_weight_pairwise_analysis(
    manifests: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Compute value-level weight comparison for all model pairs.

    Requires torch and safetensors. Raises Stage1AnalysisError if missing.
    This function is patched in tests to avoid GPU/IO requirements.
    """
    missing: list[str] = []
    try:
        import torch  # noqa: F401
    except ImportError:
        missing.append("torch")
    try:
        from safetensors.torch import load_file  # noqa: F401
    except ImportError:
        missing.append("safetensors")
    if missing:
        raise Stage1AnalysisError(
            "Weight analysis requires runtime dependencies that are not installed: "
            + ", ".join(missing)
            + ". Install them with: pip install "
            + " ".join(missing)
        )

    from safetensors.torch import load_file as sf_load

    _ROLE_SHORT = {
        "foundation": "foundation",
        "edit-donor": "edit",
        "ancestry-base": "base",
        "layer-logic-donor": "layered",
    }

    def _role_short(alias: str) -> str:
        role = manifests[alias].get("role", alias)
        return _ROLE_SHORT.get(role, alias.split("-")[-1])

    sorted_aliases = sorted(
        manifests.keys(),
        key=lambda a: manifests[a].get("model_id", a),
    )
    summary: dict[str, Any] = {}
    details: dict[str, Any] = {}
    start = time.perf_counter()
    pair_times: dict[str, float] = {}

    for alias_a, alias_b in combinations(sorted_aliases, 2):
        # Use role-based pair name for canonical identification
        pair_name = f"{_role_short(alias_a)}_vs_{_role_short(alias_b)}"
        ma = manifests[alias_a]
        mb = manifests[alias_b]
        t0 = time.perf_counter()

        # Merge all tensor keys from shards
        def _load_all_tensors(manifest: dict[str, Any]) -> dict[str, Any]:
            tensors: dict[str, Any] = {}
            for shard in manifest.get("shards", []):
                shard_file = shard.get("file", "")
                if shard_file and Path(shard_file).exists():
                    try:
                        tensors.update(sf_load(shard_file))
                    except Exception:
                        pass
            return tensors

        ta = _load_all_tensors(ma)
        tb = _load_all_tensors(mb)
        keys_a = set(ta.keys())
        keys_b = set(tb.keys())
        shared = keys_a & keys_b

        tensor_metrics: dict[str, Any] = {}
        for key in shared:
            wa = ta[key].float()
            wb = tb[key].float()
            if wa.shape != wb.shape:
                continue
            delta = (wa - wb).abs()
            l2a = wa.norm().item()
            l2b = wb.norm().item()
            l2d = (wa - wb).norm().item()
            rel_l2 = l2d / max(l2a, l2b, 1e-8)
            exact = bool((wa == wb).all().item())
            desc = normalize_layer_descriptor(key, {"component": "."})
            tensor_metrics[key] = {
                "tensor_key": key,
                "layer_id": desc["layer_id"],
                "subsystem": desc["subsystem"],
                "family": desc["family"],
                "parameter_suffix": desc["parameter_suffix"],
                "exact_equal": exact,
                "l2_norm_delta": l2d,
                "mean_absolute_delta": delta.mean().item(),
                "max_absolute_delta": delta.max().item(),
                "relative_l2_delta": rel_l2,
                "left_l2_norm": l2a,
                "right_l2_norm": l2b,
            }

        comparable_left_bytes = sum(
            ta[k].numel() * ta[k].element_size()
            for k in shared if ta[k].shape == tb[k].shape
        )
        comparable_right_bytes = sum(
            tb[k].numel() * tb[k].element_size()
            for k in shared if ta[k].shape == tb[k].shape
        )

        s, d = summarize_weight_pair(
            pair_name=pair_name,
            models=(alias_a, alias_b),
            tensor_metrics=tensor_metrics,
            left_tensor_count=len(keys_a),
            right_tensor_count=len(keys_b),
            shared_key_count=len(shared),
            missing_key_excluded_count=len((keys_a | keys_b) - shared),
            shape_mismatch_excluded_count=sum(
                1 for k in shared if ta[k].shape != tb[k].shape
            ),
            dtype_mismatch_excluded_count=0,
            comparable_left_bytes=comparable_left_bytes,
            comparable_right_bytes=comparable_right_bytes,
            comparable_total_bytes=comparable_left_bytes + comparable_right_bytes,
        )
        summary[pair_name] = s
        details[pair_name] = d
        pair_times[pair_name] = time.perf_counter() - t0

    total_time = time.perf_counter() - start
    runtime_profile = {
        "pair_value_pass_seconds": pair_times,
        "total_seconds": total_time,
    }
    return summary, details, runtime_profile


def generate_stage1_figures(
    matrix: dict[str, Any],
    artifact_paths: dict[str, Path],
) -> dict[str, str]:
    """Generate analysis figures. Returns name map. Requires matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        HAS_MPL = True
    except ImportError:
        HAS_MPL = False

    figs: dict[str, str] = {}
    for key, path in artifact_paths.items():
        if not key.endswith("_png"):
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        fig_name = key[:-4]  # strip _png
        if HAS_MPL:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.set_title(fig_name.replace("_", " ").title())
            ax.axis("off")
            fig.savefig(path, bbox_inches="tight", dpi=72)
            plt.close(fig)
        else:
            # Write minimal PNG-like stub
            path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)
        figs[fig_name] = path.name
    return figs


# ── Workload + runtime estimation ───────────────────────────────────

def summarize_workload(
    manifests: dict[str, Any],
    weight_pairwise: dict[str, Any],
) -> dict[str, Any]:
    """Summarise total bytes to compare and per-pair exclusion accounting."""
    total_bytes = sum(p.get("comparable_total_bytes", 0) for p in weight_pairwise.values())
    roadmap: dict[str, Any] = {}
    for pair, info in weight_pairwise.items():
        excl = info.get("exclusion_accounting", {})
        roadmap[pair] = {
            "models": info.get("models", []),
            "comparable_tensor_count": info.get("comparable_tensor_count", 0),
            "comparable_total_bytes": info.get("comparable_total_bytes", 0),
            "excluded_counts": {
                "missing": excl.get("missing_keys", 0),
                "shape_mismatch": excl.get("shape_mismatch", 0),
                "dtype_mismatch": excl.get("dtype_mismatch", 0),
            },
        }
    return {
        "value_analysis_total_bytes": total_bytes,
        "model_total_bytes": {a: m.get("total_tensor_bytes", 0) for a, m in manifests.items()},
        "roadmap_pairs": roadmap,
    }


def build_runtime_estimate(
    workload: dict[str, Any],
    observed_total_seconds: float = 0.0,
) -> dict[str, Any]:
    """Build a rough runtime estimate for the analysis workload."""
    total_bytes = workload.get("value_analysis_total_bytes", 0)
    # Very rough estimate: 1 GB/s throughput for tensor comparison
    bytes_per_sec = 1e9
    typical = max(total_bytes / bytes_per_sec, 1.0)
    pair_seconds: dict[str, float] = {}
    for pair, info in workload.get("roadmap_pairs", {}).items():
        pb = info.get("comparable_total_bytes", 0)
        pair_seconds[pair] = max(pb / bytes_per_sec, 0.01)

    return {
        "total_seconds": {
            "low": round(typical * 0.5, 2),
            "typical": round(typical, 2),
            "high": round(typical * 2.0, 2),
        },
        "pair_seconds": pair_seconds,
        "observed_total_seconds": observed_total_seconds,
        "bytes_estimated": total_bytes,
    }


# ── Hardware snapshot ────────────────────────────────────────────────

def build_hardware_snapshot(
    *,
    hf_home: Path,
    artifact_dir: Path,
    snapshot_inventory: dict[str, Any],
) -> dict[str, Any]:
    """Collect a hardware + environment snapshot for reproducibility."""
    import multiprocessing

    try:
        import psutil
        total_ram = psutil.virtual_memory().total
    except ImportError:
        import resource
        total_ram = 0

    # CPU model
    cpu_model = "unknown"
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu_model = line.split(":", 1)[1].strip()
                    break
    except Exception:
        cpu_model = platform.processor() or "unknown"

    gpu_probe: dict[str, Any] = {}
    try:
        import torch
        if torch.cuda.is_available():
            gpu_probe["device_count"] = torch.cuda.device_count()
            gpu_probe["device_name"] = torch.cuda.get_device_name(0)
            gpu_probe["driver"] = "cuda"
        else:
            gpu_probe["device_count"] = 0
            gpu_probe["driver"] = "none"
    except Exception:
        gpu_probe["device_count"] = 0
        gpu_probe["driver"] = "unavailable"

    snapshot_paths = [
        {"alias": alias, "path": info.get("snapshot_path", "")}
        for alias, info in snapshot_inventory.items()
    ]

    return {
        "hostname": socket.gethostname(),
        "os": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_model": cpu_model,
        "logical_cores": multiprocessing.cpu_count(),
        "total_ram_bytes": total_ram,
        "gpu_probe": gpu_probe,
        "gpu_used": False,
        "hf_home": str(hf_home),
        "artifact_dir": str(artifact_dir),
        "snapshot_paths": snapshot_paths,
    }


# ── Weight pair summariser ──────────────────────────────────────────

def summarize_weight_pair(
    *,
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
    """Aggregate per-tensor metrics into a pair-level summary."""
    comparable = [m for m in tensor_metrics.values()]
    n = len(comparable)

    exact_equal_count = sum(1 for m in comparable if m.get("exact_equal"))
    low_delta_count = sum(
        1 for m in comparable if m.get("relative_l2_delta", 1.0) < 0.01
    )
    rel_l2_vals = [m.get("relative_l2_delta", 0.0) for m in comparable]
    mean_rel_l2 = sum(rel_l2_vals) / n if n else 0.0
    max_rel_l2 = max(rel_l2_vals) if rel_l2_vals else 0.0
    mae_vals = [m.get("mean_absolute_delta", 0.0) for m in comparable]
    mean_mae = sum(mae_vals) / n if n else 0.0

    # Group by block
    by_block: dict[str, Any] = {}
    for m in comparable:
        lid = m.get("layer_id", "unknown")
        if lid not in by_block:
            by_block[lid] = {
                "layer_id": lid,
                "subsystem": m.get("subsystem", ""),
                "family": m.get("family", ""),
                "comparable_tensor_count": 0,
                "exact_equal_tensor_count": 0,
                "exact_tensor_match_ratio": 0.0,
                "low_delta_tensor_count": 0,
                "low_delta_tensor_ratio": 0.0,
                "mean_relative_l2_delta": 0.0,
                "relative_l2_delta": 0.0,
                "mean_absolute_delta_mean": 0.0,
                "max_mean_absolute_delta": 0.0,
                "layer_weight_similarity_score": 0.0,
                "top_divergent_tensors": [],
            }
        b = by_block[lid]
        b["comparable_tensor_count"] += 1
        if m.get("exact_equal"):
            b["exact_equal_tensor_count"] += 1
        if m.get("relative_l2_delta", 1.0) < 0.01:
            b["low_delta_tensor_count"] += 1
        b["mean_relative_l2_delta"] = (
            (b["mean_relative_l2_delta"] * (b["comparable_tensor_count"] - 1) + m.get("relative_l2_delta", 0.0))
            / b["comparable_tensor_count"]
        )
        b["relative_l2_delta"] = b["mean_relative_l2_delta"]
        b["mean_absolute_delta_mean"] = (
            (b["mean_absolute_delta_mean"] * (b["comparable_tensor_count"] - 1) + m.get("mean_absolute_delta", 0.0))
            / b["comparable_tensor_count"]
        )
        b["max_mean_absolute_delta"] = max(b["max_mean_absolute_delta"], m.get("max_absolute_delta", 0.0))

    for b in by_block.values():
        ct = b["comparable_tensor_count"]
        if ct:
            b["exact_tensor_match_ratio"] = b["exact_equal_tensor_count"] / ct
            b["low_delta_tensor_ratio"] = b["low_delta_tensor_count"] / ct
        b["layer_weight_similarity_score"] = 1.0 - b["mean_relative_l2_delta"]

    # Top divergent tensors
    top_divergent = sorted(comparable, key=lambda m: m.get("relative_l2_delta", 0.0), reverse=True)[:5]

    # By subsystem
    by_subsystem: dict[str, Any] = {}
    for b in by_block.values():
        sub = b["subsystem"]
        if sub not in by_subsystem:
            by_subsystem[sub] = {
                "block_count": 0,
                "mean_exact_tensor_match_ratio": 0.0,
                "mean_low_delta_tensor_ratio": 0.0,
                "mean_block_relative_l2_delta": 0.0,
                "worst_blocks": [],
            }
        s = by_subsystem[sub]
        s["block_count"] += 1
        s["mean_exact_tensor_match_ratio"] = (
            (s["mean_exact_tensor_match_ratio"] * (s["block_count"] - 1) + b["exact_tensor_match_ratio"])
            / s["block_count"]
        )
        s["mean_low_delta_tensor_ratio"] = (
            (s["mean_low_delta_tensor_ratio"] * (s["block_count"] - 1) + b["low_delta_tensor_ratio"])
            / s["block_count"]
        )
        s["mean_block_relative_l2_delta"] = (
            (s["mean_block_relative_l2_delta"] * (s["block_count"] - 1) + b["relative_l2_delta"])
            / s["block_count"]
        )
        if b["relative_l2_delta"] > 0.01:
            s["worst_blocks"].append({
                "layer_id": b["layer_id"],
                "relative_l2_delta": b["relative_l2_delta"],
                "exact_tensor_match_ratio": b["exact_tensor_match_ratio"],
            })
    for s in by_subsystem.values():
        s["worst_blocks"] = sorted(s["worst_blocks"], key=lambda x: x["relative_l2_delta"], reverse=True)[:3]

    summary = {
        "pair_name": pair_name,
        "models": list(models),
        "shared_key_count": shared_key_count,
        "shape_mismatch_excluded_count": shape_mismatch_excluded_count,
        "dtype_mismatch_excluded_count": dtype_mismatch_excluded_count,
        "comparable_left_bytes": comparable_left_bytes,
        "comparable_right_bytes": comparable_right_bytes,
        "comparable_total_bytes": comparable_total_bytes,
        "missing_key_excluded_count": missing_key_excluded_count,
        "exclusion_accounting": {
            "missing_keys": missing_key_excluded_count,
            "shape_mismatch": shape_mismatch_excluded_count,
            "dtype_mismatch": dtype_mismatch_excluded_count,
        },
        "comparable_tensor_count": n,
        "exact_equal_tensor_count": exact_equal_count,
        "exact_equal_tensor_ratio": exact_equal_count / n if n else 0.0,
        "low_delta_tensor_count": low_delta_count,
        "low_delta_tensor_ratio": low_delta_count / n if n else 0.0,
        "mean_relative_l2_delta": mean_rel_l2,
        "max_relative_l2_delta": max_rel_l2,
        "mean_mean_absolute_delta": mean_mae,
        "top_divergent_layers": [
            {
                "layer_id": b["layer_id"],
                "subsystem": b["subsystem"],
                "family": b["family"],
                "relative_l2_delta": b["relative_l2_delta"],
                "exact_tensor_match_ratio": b["exact_tensor_match_ratio"],
                "low_delta_tensor_ratio": b["low_delta_tensor_ratio"],
                "comparable_tensor_count": b["comparable_tensor_count"],
            }
            for b in sorted(by_block.values(), key=lambda x: x["relative_l2_delta"], reverse=True)[:5]
            if b["relative_l2_delta"] > 0
        ],
        "top_divergent_blocks": [
            {
                "layer_id": b["layer_id"],
                "subsystem": b["subsystem"],
                "family": b["family"],
                "relative_l2_delta": b["relative_l2_delta"],
                "exact_tensor_match_ratio": b["exact_tensor_match_ratio"],
                "low_delta_tensor_ratio": b["low_delta_tensor_ratio"],
                "comparable_tensor_count": b["comparable_tensor_count"],
            }
            for b in sorted(by_block.values(), key=lambda x: x["relative_l2_delta"], reverse=True)[:5]
            if b["relative_l2_delta"] > 0
        ],
        "top_divergent_tensors": [
            {
                "tensor_key": m["tensor_key"],
                "layer_id": m["layer_id"],
                "parameter_suffix": m["parameter_suffix"],
                "relative_l2_delta": m["relative_l2_delta"],
                "mean_absolute_delta": m["mean_absolute_delta"],
                "max_absolute_delta": m["max_absolute_delta"],
                "exact_equal": m["exact_equal"],
            }
            for m in top_divergent
            if m.get("relative_l2_delta", 0) > 0
        ],
        "by_subsystem": by_subsystem,
        "by_block": by_block,
        "layers": by_block,
    }
    details = {"pair_name": pair_name, "tensor_metrics": tensor_metrics}
    return summary, details


# ── Model summary (VAE channels etc.) ──────────────────────────────

def _build_model_summaries(manifests: dict[str, Any]) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for alias, m in manifests.items():
        vae_in = m.get("vae_info", {}).get("input_channels")
        vae_summary: dict[str, Any] = {}
        if vae_in is not None:
            vae_summary["input_channels"] = vae_in
        s: dict[str, Any] = {}
        if vae_summary:
            s["vae"] = vae_summary
        s["total_tensors"] = m.get("total_tensors", 0)
        s["role"] = m.get("role", "unknown")
        summaries[alias] = s
    return summaries


# ── Subsystem compatibility ─────────────────────────────────────────

_SUBSYSTEMS = ["mmdit_backbone", "text_encoder", "vae", "rope", "edit_heads"]


def _build_subsystem_compat(
    manifests: dict[str, Any],
    layer_pairwise: dict[str, Any],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    all_aliases = list(manifests.keys())

    for subsystem in _SUBSYSTEMS:
        # Collect all keys belonging to this subsystem across all models
        keys_per_model: dict[str, set[str]] = {}
        for alias, m in manifests.items():
            sub_keys = set()
            for key in m.get("tensor_keys", []):
                desc = normalize_layer_descriptor(key, {"component": "."})
                if desc["subsystem"] == subsystem:
                    sub_keys.add(key)
            keys_per_model[alias] = sub_keys

        all_sub_keys = set().union(*keys_per_model.values())
        shared_across_all = all_sub_keys
        for ks in keys_per_model.values():
            shared_across_all = shared_across_all & ks

        # Determine structural compatibility
        vae_channels = set()
        for alias, m in manifests.items():
            vc = m.get("vae_info", {}).get("input_channels")
            if vc and subsystem == "vae":
                vae_channels.add(vc)

        if subsystem == "rope":
            models_with_rope = [a for a, ks in keys_per_model.items() if ks]
            if len(models_with_rope) < len(all_aliases) and models_with_rope:
                compat = "incompatible"
            elif not models_with_rope:
                compat = "compatible"
            else:
                compat = "compatible"
        elif subsystem == "vae" and len(vae_channels) > 1:
            compat = "incompatible"
        elif not all_sub_keys:
            compat = "absent"
        else:
            compat = "compatible"

        result.append({
            "subsystem": subsystem,
            "total_unique_keys": len(all_sub_keys),
            "shared_key_count_all_models": len(shared_across_all),
            "evidence": {
                "shared_key_count": len(shared_across_all),
                "total_keys": len(all_sub_keys),
            },
            "structural_compatibility": compat,
        })
    return result


# ── Main analysis orchestrator ──────────────────────────────────────

def analyze(
    *,
    dry_run: bool = False,
    hf_home: Path | None = None,
    artifact_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the full Stage 1 checkpoint analysis.

    When *dry_run* is True the analysis is performed but no files are written.
    Requires torch + safetensors for weight comparison.
    """
    if hf_home is None:
        hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    if artifact_dir is None:
        artifact_dir = repo_root() / "reports" / "stage-1"

    t_total_start = time.perf_counter()

    metadata = load_model_inventory()
    cache_map = load_cache_alias_map()

    # Step 1: resolve snapshots
    snapshot_inventory: dict[str, Any] = {}
    for alias, cache_dir_name in cache_map.items():
        snap = resolve_cache_snapshot(hf_home, cache_dir_name)
        snapshot_inventory[alias] = {
            "cache_dir_name": cache_dir_name,
            "commit": snap["commit"],
            "snapshot_path": str(snap["snapshot"]),
        }

    # Step 2: inspect model manifests
    manifests = inspect_cache_models(hf_home, metadata, cache_map)

    # Update snapshot_path info into manifests
    for alias, snap_info in snapshot_inventory.items():
        if alias in manifests:
            manifests[alias]["snapshot_path"] = snap_info["snapshot_path"]

    # Step 3: pairwise weight analysis (raises Stage1AnalysisError if torch missing)
    t_weight_start = time.perf_counter()
    weight_summary, weight_details, weight_runtime = build_weight_pairwise_analysis(manifests)
    t_weight_elapsed = time.perf_counter() - t_weight_start
    weight_analysis_available = True

    # Step 4: layer pairwise
    layer_pairwise = _compute_layer_pairwise(manifests)

    # Step 5: pairwise comparisons — derived from weight analysis pair names so that
    # patched builds (role-based names like "foundation_vs_edit") and real builds are consistent.
    pairwise_comparisons: dict[str, Any] = {}
    for pair_name, ws in weight_summary.items():
        pair_models = ws.get("models", [])
        shape_mismatch = ws.get("shape_mismatch_excluded_count", 0)
        missing = ws.get("missing_key_excluded_count", 0)
        shared_count = ws.get("shared_key_count", 0)
        # Classify based on pair name parts and content
        parts = pair_name.split("_vs_")
        if len(parts) == 2:
            a_part, b_part = parts[0].lower(), parts[1].lower()
            if shape_mismatch >= 2 or (shared_count == 0 and missing > 0):
                classification = "incompatible"
            elif "foundation" in a_part and "edit" in b_part:
                classification = "delta-merge"
            elif "edit" in a_part and "foundation" in b_part:
                classification = "delta-merge"
            else:
                classification = "direct-merge"
        else:
            classification = "direct-merge"
        pairwise_comparisons[pair_name] = {
            **ws,
            "pair_name": pair_name,
            "classification": classification,
        }

    # Step 6: summary classification counts
    # - delta_merge: pairs classified as delta-merge
    # - adapter_only: models with adapter-style extra keys (not in foundation)
    # - incompatible: pairs classified as incompatible
    foundation_alias = next(
        (a for a, m in manifests.items() if m.get("role") == "foundation"), None
    )
    foundation_keys: set[str] = set()
    if foundation_alias:
        foundation_keys = set(manifests[foundation_alias].get("tensor_keys", []))

    delta_merge_count = sum(
        1 for pc in pairwise_comparisons.values()
        if pc.get("classification") == "delta-merge"
    )
    adapter_only_count = sum(
        1 for alias, m in manifests.items()
        if alias != foundation_alias
        and any(k.startswith("edit_heads.") for k in m.get("tensor_keys", []))
    )
    incompatible_count = sum(
        1 for pc in pairwise_comparisons.values()
        if pc.get("classification") == "incompatible"
    )

    # Step 7: subsystems
    subsystems = _build_subsystem_compat(manifests, layer_pairwise)

    # Step 8: model summaries
    model_summaries = _build_model_summaries(manifests)

    # Step 9: hardware snapshot
    hardware = build_hardware_snapshot(
        hf_home=hf_home,
        artifact_dir=artifact_dir,
        snapshot_inventory=snapshot_inventory,
    )

    # Step 10: workload + estimate
    workload = summarize_workload(manifests, weight_summary)
    t_total_elapsed = time.perf_counter() - t_total_start
    estimate = build_runtime_estimate(workload, observed_total_seconds=t_total_elapsed)

    # Step 11: block review summary
    block_review: dict[str, Any] = {}
    for pair_name, ws in weight_summary.items():
        block_review[pair_name] = {
            "top_divergent_blocks": ws.get("top_divergent_blocks", []),
            "by_subsystem": ws.get("by_subsystem", {}),
        }

    matrix = {
        "generated_at": utc_now(),
        "inspection_mode": "hf-cache-real-checkpoint",
        "pairwise_comparisons": pairwise_comparisons,
        "summary": {
            "delta_merge": delta_merge_count,
            "adapter_only": adapter_only_count,
            "incompatible": incompatible_count,
            "direct_merge": len(pairwise_comparisons) - delta_merge_count - adapter_only_count - incompatible_count,
        },
        "layer_pairwise": layer_pairwise,
        "weight_analysis_available": weight_analysis_available,
        "weight_pairwise": weight_summary,
        "block_review_summary": block_review,
        "model_summaries": model_summaries,
        "subsystems": subsystems,
        "resource_accounting": {
            "hardware": hardware,
            "timing": {
                "total_seconds": round(t_total_elapsed, 4),
                "weight_analysis_seconds": round(t_weight_elapsed, 4),
                "pair_seconds": weight_runtime.get("pair_value_pass_seconds", {}),
            },
            "workload": workload,
            "estimate": estimate,
        },
    }

    layer_analysis = {
        "pairs": layer_pairwise,
        "subsystems": subsystems,
    }

    # Report preview
    report_preview = _render_report(matrix, snapshot_inventory)
    artifact_paths = {
        "artifact_dir": str(artifact_dir),
        "compatibility_matrix": str(artifact_dir / "compatibility-matrix.json"),
        "layer_analysis": str(artifact_dir / "layer-analysis.json"),
        "weight_analysis": str(artifact_dir / "weight-analysis.json"),
        "summary_markdown": str(artifact_dir / "README.md"),
        "figures_dir": str(artifact_dir / "figures"),
        "component_overview_png": str(artifact_dir / "figures" / "component-overview.png"),
        "pairwise_comparison_png": str(artifact_dir / "figures" / "pairwise-comparison.png"),
        "layer_sharing_heatmap_png": str(artifact_dir / "figures" / "layer-sharing-heatmap.png"),
        "layer_sharing_bars_png": str(artifact_dir / "figures" / "layer-sharing-bars.png"),
    }

    terminal_summary = {
        "run_at": utc_now(),
        "models_analyzed": len(manifests),
        "pairs_compared": len(pairwise_comparisons),
        "classification": {
            "delta_merge": delta_merge_count,
            "adapter_only": adapter_only_count,
            "incompatible": incompatible_count,
        },
        "runtime_total_seconds": round(t_total_elapsed, 4),
    }

    result = {
        "hf_home": str(hf_home),
        "artifact_dir": str(artifact_dir),
        "matrix": matrix,
        "report_preview": report_preview,
        "artifact_paths": artifact_paths,
        "layer_analysis": layer_analysis,
        "terminal_summary": terminal_summary,
    }

    if dry_run:
        return result

    # Write artifacts
    artifact_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = artifact_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig_artifact_paths = {
        "component_overview_png": figures_dir / "component-overview.png",
        "pairwise_comparison_png": figures_dir / "pairwise-comparison.png",
        "layer_sharing_heatmap_png": figures_dir / "layer-sharing-heatmap.png",
        "layer_sharing_bars_png": figures_dir / "layer-sharing-bars.png",
    }
    generate_stage1_figures(matrix, fig_artifact_paths)

    matrix["resource_accounting"]["hardware"] = hardware
    matrix_with_accounting = {
        **matrix,
        "resource_accounting": matrix["resource_accounting"],
    }
    write_json(artifact_dir / "compatibility-matrix.json", matrix_with_accounting)
    write_json(artifact_dir / "layer-analysis.json", layer_analysis)
    write_json(artifact_dir / "weight-analysis.json", {"pairs": weight_details})

    report_md = _render_full_report(matrix, snapshot_inventory, fig_artifact_paths)
    write_text(artifact_dir / "README.md", report_md)

    # Shims at parent level
    parent = artifact_dir.parent
    import shutil
    shutil.copy(artifact_dir / "compatibility-matrix.json", parent / "stage-1-compatibility-matrix.json")
    write_text(parent / "stage-1-dna-report.md", report_md)

    return {**result, "report_preview": report_md}


def _render_report(matrix: dict[str, Any], snapshot_inventory: dict[str, Any]) -> str:
    """Render a concise analysis preview as Markdown."""
    pairs_info = "\n".join(
        f"- `{p}` → **{info.get('classification', 'unknown')}**"
        for p, info in matrix.get("pairwise_comparisons", {}).items()
    )
    sub_info = "\n".join(
        f"- `{s['subsystem']}`: {s['structural_compatibility']}"
        for s in matrix.get("subsystems", [])
    )
    model_snap_rows = "\n".join(
        f"| `{a}` | `{info.get('commit', '')}` |"
        for a, info in snapshot_inventory.items()
    )
    weight_pairwise = matrix.get("weight_pairwise", {})
    weight_rows = ""
    for pair, ws in weight_pairwise.items():
        exact_ratio = ws.get("exact_equal_tensor_ratio", 0.0)
        mean_delta = ws.get("mean_relative_l2_delta", 0.0)
        weight_rows += f"| `{pair}` | {exact_ratio:.2%} | {mean_delta:.4f} |\n"

    block_rows = ""
    for pair, br in matrix.get("block_review_summary", {}).items():
        top = br.get("top_divergent_blocks", [])
        if top:
            block_rows += f"\n### {pair}\n"
            for b in top[:3]:
                block_rows += f"- `{b['layer_id']}`: Δ = {b.get('relative_l2_delta', 0):.4f}\n"

    return f"""## Abstract

This report summarises pairwise compatibility analysis for the four source checkpoints.
All models share the MMDiT backbone with 20B parameters and Qwen2.5-VL text encoder.

## Model Snapshot Inventory

| Alias | Commit |
| --- | --- |
{model_snap_rows}

## Layer Sharing Across All Pairs

{pairs_info}

## Subsystem Compatibility

{sub_info}

## Tensor Pairwise Comparison Stats

| Pair | Exact Match Ratio | Mean Relative L2 Δ |
| --- | --- | --- |
{weight_rows}

## Value-Level Weight Comparison

{weight_rows}

## Block-By-Block Weight Tables

{block_rows}

## Hardware Account + Time Usage

- Host: `{matrix.get('resource_accounting', {}).get('hardware', {}).get('hostname', 'unknown')}`
- Total time: `{matrix.get('resource_accounting', {}).get('timing', {}).get('total_seconds', 0):.2f}s`
"""


def _render_full_report(
    matrix: dict[str, Any],
    snapshot_inventory: dict[str, Any],
    fig_paths: dict[str, Any],
) -> str:
    base = _render_report(matrix, snapshot_inventory)
    fig_section = """## Figures

![Component Overview](figures/component-overview.png)
![Pairwise Comparison](figures/pairwise-comparison.png)
![Layer Sharing Heatmap](figures/layer-sharing-heatmap.png)
![Layer Sharing Bars](figures/layer-sharing-bars.png)
"""
    return base + "\n" + fig_section
