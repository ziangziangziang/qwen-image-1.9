"""Stage 2 — Delta-edit checkpoint fusion.

Supports two merge strategies:

1. **delta-edit** (whole-backbone):
       edit_delta  = edit_donor_weights - delta_base_weights
       merged      = foundation_weights + coefficient * edit_delta

2. **delta-edit-windowed** (bridge-window only):
       layer_delta = layer_donor_weights - delta_base_weights
       merged[key] = prev_merged[key] + coefficient * layer_delta[key]
                     applied only to transformer blocks in [block_start, block_end)

The tri-capability recipe runs stage 1 (editing delta, full backbone) then
stage 2 (layering delta, blocks 40-60), producing a single merged checkpoint
that handles image generation, image editing, and image layering.

Only MMDiT backbone tensors are blended; text_encoder, VAE, and
RoPE tensors are passed through from the foundation checkpoint.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from qwen_image_19.config_io import repo_root, write_json
from qwen_image_19.contracts import public_path, utc_now


class FusionError(RuntimeError):
    """Raised when merge planning or execution fails."""


# ── Subsystem routing ────────────────────────────────────────────────

# Tensor key prefixes that belong to the MMDiT backbone and should be merged
_MERGE_PREFIXES = (
    "transformer.",
    "model.mmdit.",
    "mmdit.",
    "dit.",
)

# Prefixes that are passed through from the foundation without modification
_PASSTHROUGH_PREFIXES = (
    "text_encoder.",
    "vae.",
    "rope.",
    "tokenizer.",
    "scheduler.",
    "feature_extractor.",
)


def _is_merge_tensor(key: str) -> bool:
    return any(key.startswith(p) for p in _MERGE_PREFIXES)


def _is_passthrough_tensor(key: str) -> bool:
    return any(key.startswith(p) for p in _PASSTHROUGH_PREFIXES)


# ── Config helpers ───────────────────────────────────────────────────

def _load_merge_delta_config() -> dict[str, Any]:
    path = repo_root() / "configs" / "merge" / "stage-2-delta-edit.yaml"
    if path.exists():
        try:
            import yaml
            return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            pass
    return {}


# ── Planning ─────────────────────────────────────────────────────────

def plan_fusion(
    *,
    foundation_id: str,
    edit_donor_id: str,
    delta_base_id: str,
    coefficient: float = 0.35,
    run_dir: Path,
    remote_config: str | None = None,
) -> dict[str, Any]:
    """Build a merge execution plan without loading any weights."""
    from qwen_image_19.remote import default_remote_context
    ctx = default_remote_context(remote_config)
    output_dir = run_dir / "merge" / "merged-checkpoint"
    log_path = run_dir / "merge" / "merge.log"
    merge_config = {
        "strategy": "delta-edit",
        "foundation": foundation_id,
        "edit_donor": edit_donor_id,
        "delta_base": delta_base_id,
        "coefficient": coefficient,
        "merge_subsystems": list(_MERGE_PREFIXES),
        "passthrough_subsystems": list(_PASSTHROUGH_PREFIXES),
    }
    return {
        "foundation_id": foundation_id,
        "edit_donor_id": edit_donor_id,
        "delta_base_id": delta_base_id,
        "coefficient": coefficient,
        "output_checkpoint": str(output_dir),
        "declared_output_checkpoint": (
            f"{ctx['artifact_dir']}/runs/{run_dir.name}/merge/merged-checkpoint"
        ),
        "log_path": str(log_path),
        "merge_config": merge_config,
        "remote_job": {
            "name": "delta-edit-merge",
            "workdir": ctx["workdir"],
            "artifact_dir": ctx["artifact_dir"],
            "status": "planned",
        },
        "metrics": {
            "strategy": "delta-edit",
            "coefficient": coefficient,
        },
        "command": [
            "python3", "-m", "qwen_image_19.stage_2_fusion",
            "--execute-worker",
            "--foundation", foundation_id,
            "--edit-donor", edit_donor_id,
            "--delta-base", delta_base_id,
            "--coefficient", str(coefficient),
            "--output", str(output_dir),
        ],
    }


# ── Execution ────────────────────────────────────────────────────────

def _require_runtime_deps() -> None:
    missing: list[str] = []
    try:
        import torch  # noqa: F401
    except ImportError:
        missing.append("torch")
    try:
        from safetensors.torch import load_file, save_file  # noqa: F401
    except ImportError:
        missing.append("safetensors")
    if missing:
        raise FusionError(
            "Merge execution requires: " + ", ".join(missing)
        )


def _download_or_locate(model_id: str, cache_dir: Path | None = None) -> Path:
    """Resolve a local path for *model_id* (HF repo ID or local path)."""
    local = Path(model_id)
    if local.exists():
        return local
    # Try HF cache
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    cache_name = "models--" + model_id.replace("/", "--")
    refs_main = hf_home / "hub" / cache_name / "refs" / "main"
    if refs_main.exists():
        commit = refs_main.read_text(encoding="utf-8").strip()
        snapshot = hf_home / "hub" / cache_name / "snapshots" / commit
        if snapshot.exists():
            return snapshot
    # Try downloading via huggingface_hub
    try:
        from huggingface_hub import snapshot_download
        local_path = snapshot_download(
            repo_id=model_id,
            cache_dir=str(cache_dir) if cache_dir else None,
        )
        return Path(local_path)
    except Exception as exc:
        raise FusionError(
            f"Cannot locate `{model_id}`. "
            "Either download it first or ensure it is in the HF cache."
        ) from exc


def _collect_tensor_keys(checkpoint_path: Path) -> list[str]:
    """Read all tensor key names from a safetensors checkpoint."""
    import struct as _struct, json as _json
    all_keys: list[str] = []
    if checkpoint_path.is_file():
        files = [checkpoint_path]
    else:
        index_path = checkpoint_path / "model.safetensors.index.json"
        if index_path.exists():
            index = _json.loads(index_path.read_text(encoding="utf-8"))
            files = sorted(
                {checkpoint_path / shard for shard in index["weight_map"].values()}
            )
        else:
            files = sorted(checkpoint_path.glob("*.safetensors"))
    for f in files:
        with f.open("rb") as fh:
            raw = fh.read(8)
            if len(raw) < 8:
                continue
            (hlen,) = _struct.unpack("<Q", raw)
            header = _json.loads(fh.read(hlen))
        all_keys.extend(k for k in header if k != "__metadata__")
    return all_keys


def _load_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    """Load all tensors from a safetensors checkpoint into a dict.

    Handles three layouts:
    1. Single ``model.safetensors`` file
    2. Sharded flat layout with ``model.safetensors.index.json``
    3. Diffusers componentized layout (``model_index.json`` + component subdirs)
       — tensor keys are prefixed with the component name (e.g. ``transformer.``)
    """
    from safetensors.torch import load_file
    import json as _json

    if checkpoint_path.is_file():
        return load_file(str(checkpoint_path))

    # Diffusers-style componentized layout
    model_index = checkpoint_path / "model_index.json"
    if model_index.exists():
        state: dict[str, Any] = {}
        for comp_dir in sorted(checkpoint_path.iterdir()):
            if not comp_dir.is_dir() or comp_dir.name.startswith("."):
                continue
            comp_name = comp_dir.name
            # Check for index file in component dir
            comp_index = comp_dir / "model.safetensors.index.json"
            if comp_index.exists():
                idx = _json.loads(comp_index.read_text(encoding="utf-8"))
                shard_files = sorted(set(idx["weight_map"].values()))
                for shard in shard_files:
                    for k, v in load_file(str(comp_dir / shard)).items():
                        state[f"{comp_name}.{k}"] = v
            else:
                for f in sorted(comp_dir.glob("*.safetensors")):
                    for k, v in load_file(str(f)).items():
                        state[f"{comp_name}.{k}"] = v
        return state

    # Flat sharded layout with index
    index_path = checkpoint_path / "model.safetensors.index.json"
    if index_path.exists():
        index = _json.loads(index_path.read_text(encoding="utf-8"))
        shard_files = sorted(set(index["weight_map"].values()))
        state = {}
        for shard in shard_files:
            state.update(load_file(str(checkpoint_path / shard)))
        return state

    # Multiple *.safetensors without index
    all_st = sorted(checkpoint_path.glob("*.safetensors"))
    state = {}
    for f in all_st:
        state.update(load_file(str(f)))
    return state


def _save_checkpoint(state_dict: dict[str, Any], output_path: Path) -> None:
    """Save a state dict to safetensors, grouping by component prefix.

    If all keys share a ``component.`` prefix structure (diffusers layout),
    each component is written to its own subdirectory with a single
    ``model.safetensors`` file.  Otherwise a flat single-file layout is used.
    """
    from safetensors.torch import save_file

    # Detect if the keys have a component prefix (diffusers layout)
    # A key like "transformer.block.0.weight" has component="transformer"
    # A key like "weight" has no component → flat layout
    components: dict[str, dict[str, Any]] = {}
    flat: dict[str, Any] = {}
    for key, tensor in state_dict.items():
        dot = key.find(".")
        if dot > 0:
            comp = key[:dot]
            rest = key[dot + 1:]
            components.setdefault(comp, {})[rest] = tensor
        else:
            flat[key] = tensor

    output_path.mkdir(parents=True, exist_ok=True)

    if flat:
        # Fell back to flat layout
        save_file(state_dict, str(output_path / "model.safetensors"))
    else:
        # Componentized layout — each component gets its own dir
        for comp_name, tensors in components.items():
            comp_dir = output_path / comp_name
            comp_dir.mkdir(parents=True, exist_ok=True)
            save_file(tensors, str(comp_dir / "model.safetensors"))
            # Diffusers also looks for diffusion_pytorch_model.safetensors
            alias = comp_dir / "diffusion_pytorch_model.safetensors"
            if not alias.exists():
                alias.symlink_to("model.safetensors")


def _copy_config_files(src: Path, dst: Path) -> None:
    """Copy non-weight config files from checkpoint dir to output dir."""
    if not src.is_dir():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for candidate in src.iterdir():
        if candidate.is_dir():
            continue
        # Skip stale shard index files — our output is always single-file
        if candidate.name == "diffusion_pytorch_model.safetensors.index.json":
            continue
        if candidate.suffix in {".json", ".txt", ".model", ".yaml", ".yml"} \
                or candidate.name.endswith(".jinja"):
            shutil.copy(candidate, dst / candidate.name)


def execute_fusion(plan: dict[str, Any]) -> dict[str, Any]:
    """Execute the delta-edit merge and write the merged checkpoint.

    Requires torch + safetensors. All arithmetic is done in bf16 to
    conserve VRAM on the MI300X.
    """
    _require_runtime_deps()
    import torch

    foundation_id = plan["foundation_id"]
    edit_donor_id = plan["edit_donor_id"]
    delta_base_id = plan["delta_base_id"]
    coefficient = float(plan["coefficient"])
    output_path = Path(plan["output_checkpoint"])
    log_path = Path(plan["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()
    t0 = time.perf_counter()

    with log_path.open("w", encoding="utf-8") as log:

        def _log(msg: str) -> None:
            log.write(msg + "\n")
            log.flush()

        _log(f"[merge] started={started_at}")
        _log(f"[merge] foundation={foundation_id}")
        _log(f"[merge] edit_donor={edit_donor_id}")
        _log(f"[merge] delta_base={delta_base_id}")
        _log(f"[merge] coefficient={coefficient}")

        _log("[merge] locating foundation checkpoint …")
        foundation_path = _download_or_locate(foundation_id)
        _log(f"[merge] foundation → {foundation_path}")

        _log("[merge] locating edit-donor checkpoint …")
        edit_path = _download_or_locate(edit_donor_id)
        _log(f"[merge] edit_donor → {edit_path}")

        _log("[merge] locating delta-base checkpoint …")
        base_path = _download_or_locate(delta_base_id)
        _log(f"[merge] delta_base → {base_path}")

        _log("[merge] loading foundation tensors …")
        foundation = _load_checkpoint(foundation_path)
        _log(f"[merge] foundation loaded: {len(foundation)} tensors")

        _log("[merge] loading edit-donor tensors …")
        edit = _load_checkpoint(edit_path)
        _log(f"[merge] edit_donor loaded: {len(edit)} tensors")

        _log("[merge] loading delta-base tensors …")
        base = _load_checkpoint(base_path)
        _log(f"[merge] delta_base loaded: {len(base)} tensors")

        # Compute merged state dict
        _log("[merge] computing delta-edit blend …")
        merged: dict[str, Any] = {}
        modified = 0
        passthrough = 0
        skipped = 0

        try:
            from tqdm.auto import tqdm as _tqdm
            iterator = _tqdm(list(foundation.keys()), desc="merging", unit="tensor")
        except ImportError:
            iterator = list(foundation.keys())  # type: ignore[assignment]

        for key in iterator:
            if _is_passthrough_tensor(key) or not _is_merge_tensor(key):
                # Keep foundation tensor as-is
                merged[key] = foundation[key]
                passthrough += 1
                continue

            if key not in edit or key not in base:
                # Not present in donor or base — keep foundation
                merged[key] = foundation[key]
                skipped += 1
                continue

            f_tensor = foundation[key].to(torch.bfloat16)
            e_tensor = edit[key].to(torch.bfloat16)
            b_tensor = base[key].to(torch.bfloat16)

            if f_tensor.shape != e_tensor.shape or f_tensor.shape != b_tensor.shape:
                # Shape mismatch — keep foundation
                merged[key] = foundation[key]
                skipped += 1
                continue

            delta = e_tensor - b_tensor
            merged[key] = (f_tensor + coefficient * delta).contiguous()
            modified += 1

        _log(f"[merge] modified={modified} passthrough={passthrough} skipped={skipped}")

        # Write merged checkpoint
        output_path.mkdir(parents=True, exist_ok=True)
        _save_checkpoint(merged, output_path)
        _copy_config_files(Path(foundation_path), output_path)
        # Copy component-level config files (config.json, generation_config.json, etc.)
        # and key non-weight subdirs from foundation so downstream tools can load components.
        for comp_dir in Path(foundation_path).iterdir():
            if not comp_dir.is_dir() or comp_dir.name.startswith("."):
                continue
            out_comp = output_path / comp_dir.name
            out_comp.mkdir(parents=True, exist_ok=True)
            for f in comp_dir.iterdir():
                if not f.is_file():
                    continue
                # Skip sharded index files — the merged output uses a single model.safetensors
                if f.name in {
                    "model.safetensors.index.json",
                    "diffusion_pytorch_model.safetensors.index.json",
                }:
                    continue
                if f.suffix in {".json", ".txt", ".model", ".yaml", ".yml"} \
                        or f.name.endswith(".jinja"):
                    dst = out_comp / f.name
                    if not dst.exists():
                        shutil.copy2(str(f.resolve()), str(dst))
        # Copy tokenizer and scheduler from foundation (needed for direction measurement)
        # Use symlinks=False to dereference HF cache symlinks into real files.
        for subdir in ("tokenizer", "tokenizer_2", "scheduler", "feature_extractor"):
            src_sub = Path(foundation_path) / subdir
            if src_sub.is_dir():
                dst_sub = output_path / subdir
                if not dst_sub.exists():
                    shutil.copytree(str(src_sub), str(dst_sub), symlinks=False)
                    _log(f"[merge] copied {subdir}/ from foundation (dereferenced symlinks)")
        _log(f"[merge] written to {output_path}")

    duration = time.perf_counter() - t0
    return {
        "status": "succeeded",
        "started_at": started_at,
        "ended_at": utc_now(),
        "duration_seconds": round(duration, 2),
        "output_checkpoint": str(output_path),
        "declared_output_checkpoint": plan["declared_output_checkpoint"],
        "modified_tensors": modified,
        "passthrough_tensors": passthrough,
        "skipped_tensors": skipped,
        "total_tensors": len(merged),
    }


def fuse(
    *,
    foundation_id: str,
    edit_donor_id: str,
    delta_base_id: str,
    coefficient: float = 0.35,
    output_dir: Path,
    remote_config: str | None = None,
) -> dict[str, Any]:
    """High-level wrapper: plan and execute the delta-edit merge."""
    plan = plan_fusion(
        foundation_id=foundation_id,
        edit_donor_id=edit_donor_id,
        delta_base_id=delta_base_id,
        coefficient=coefficient,
        run_dir=output_dir.parent,
        remote_config=remote_config,
    )
    return execute_fusion(plan)


# ── Tri-capability (windowed layer-delta) ────────────────────────────

def _is_block_in_window(key: str, block_start: int, block_end: int) -> bool:
    """Return True if *key* refers to a transformer block in [block_start, block_end)."""
    import re
    # Matches patterns like:
    #   transformer.transformer_blocks.42.attn.to_q.weight
    #   model.mmdit.blocks.42.ff.net.0.weight
    #   dit.blocks.42.norm1.weight
    m = re.search(r"\.(?:transformer_)?blocks?\.(\d+)\.", key)
    if m:
        return block_start <= int(m.group(1)) < block_end
    return False


def plan_layer_delta(
    *,
    prev_checkpoint: str,
    layer_donor_id: str,
    delta_base_id: str,
    coefficient: float = 0.25,
    block_start: int = 40,
    block_end: int = 60,
    run_dir: Path,
    remote_config: str | None = None,
) -> dict[str, Any]:
    """Build a windowed layer-delta merge plan (no weights loaded)."""
    from qwen_image_19.remote import default_remote_context
    ctx = default_remote_context(remote_config)
    output_dir = run_dir / "merge" / "merged-tri-capability-checkpoint"
    log_path = run_dir / "merge" / "layer-delta.log"
    return {
        "prev_checkpoint": prev_checkpoint,
        "layer_donor_id": layer_donor_id,
        "delta_base_id": delta_base_id,
        "coefficient": coefficient,
        "block_start": block_start,
        "block_end": block_end,
        "output_checkpoint": str(output_dir),
        "declared_output_checkpoint": (
            f"{ctx['artifact_dir']}/runs/{run_dir.name}/merge/merged-tri-capability-checkpoint"
        ),
        "log_path": str(log_path),
        "merge_config": {
            "strategy": "delta-edit-windowed",
            "layer_donor": layer_donor_id,
            "delta_base": delta_base_id,
            "coefficient": coefficient,
            "block_window": {"start": block_start, "end": block_end},
        },
        "remote_job": {
            "name": "layer-delta-merge",
            "workdir": ctx["workdir"],
            "artifact_dir": ctx["artifact_dir"],
            "status": "planned",
        },
        "metrics": {
            "strategy": "delta-edit-windowed",
            "layer_coefficient": coefficient,
            "block_window": f"{block_start}-{block_end}",
        },
    }


def execute_layer_delta(plan: dict[str, Any]) -> dict[str, Any]:
    """Apply the windowed layer-delta onto an already-merged checkpoint.

    Loads the previous merged checkpoint, blends layering capability into
    transformer blocks [block_start, block_end), and saves the tri-capable
    checkpoint.  All arithmetic is done in bf16.
    """
    _require_runtime_deps()
    import torch

    prev_path = Path(plan["prev_checkpoint"])
    layer_donor_id = plan["layer_donor_id"]
    delta_base_id = plan["delta_base_id"]
    coefficient = float(plan["coefficient"])
    block_start = int(plan["block_start"])
    block_end = int(plan["block_end"])
    output_path = Path(plan["output_checkpoint"])
    log_path = Path(plan["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()
    t0 = time.perf_counter()

    with log_path.open("w", encoding="utf-8") as log:

        def _log(msg: str) -> None:
            log.write(msg + "\n")
            log.flush()

        _log(f"[layer-delta] started={started_at}")
        _log(f"[layer-delta] prev_checkpoint={prev_path}")
        _log(f"[layer-delta] layer_donor={layer_donor_id}")
        _log(f"[layer-delta] delta_base={delta_base_id}")
        _log(f"[layer-delta] coefficient={coefficient}")
        _log(f"[layer-delta] block_window=[{block_start}, {block_end})")

        _log("[layer-delta] loading previous merged checkpoint …")
        prev = _load_checkpoint(prev_path)
        _log(f"[layer-delta] prev loaded: {len(prev)} tensors")

        _log("[layer-delta] locating layer-donor checkpoint …")
        donor_path = _download_or_locate(layer_donor_id)
        _log(f"[layer-delta] layer_donor → {donor_path}")

        _log("[layer-delta] locating delta-base checkpoint …")
        base_path_obj = _download_or_locate(delta_base_id)
        _log(f"[layer-delta] delta_base → {base_path_obj}")

        _log("[layer-delta] loading layer-donor tensors …")
        donor = _load_checkpoint(donor_path)
        _log(f"[layer-delta] layer_donor loaded: {len(donor)} tensors")

        _log("[layer-delta] loading delta-base tensors …")
        base = _load_checkpoint(base_path_obj)
        _log(f"[layer-delta] delta_base loaded: {len(base)} tensors")

        _log("[layer-delta] applying windowed layer-delta blend …")
        merged: dict[str, Any] = dict(prev)
        modified = 0
        skipped = 0

        try:
            from tqdm.auto import tqdm as _tqdm
            iterator = _tqdm(list(prev.keys()), desc="layer-delta", unit="tensor")
        except ImportError:
            iterator = list(prev.keys())  # type: ignore[assignment]

        for key in iterator:
            if not _is_merge_tensor(key):
                continue
            if not _is_block_in_window(key, block_start, block_end):
                continue
            if key not in donor or key not in base:
                skipped += 1
                continue

            p_tensor = merged[key].to(torch.bfloat16)
            d_tensor = donor[key].to(torch.bfloat16)
            b_tensor = base[key].to(torch.bfloat16)

            if p_tensor.shape != d_tensor.shape or p_tensor.shape != b_tensor.shape:
                skipped += 1
                continue

            delta = d_tensor - b_tensor
            merged[key] = (p_tensor + coefficient * delta).contiguous()
            modified += 1

        _log(f"[layer-delta] modified={modified} skipped={skipped}")

        output_path.mkdir(parents=True, exist_ok=True)
        _save_checkpoint(merged, output_path)
        _copy_config_files(prev_path, output_path)
        for comp_dir in prev_path.iterdir():
            if not comp_dir.is_dir() or comp_dir.name.startswith("."):
                continue
            out_comp = output_path / comp_dir.name
            out_comp.mkdir(parents=True, exist_ok=True)
            for f in comp_dir.iterdir():
                if not f.is_file():
                    continue
                if f.name in {
                    "model.safetensors.index.json",
                    "diffusion_pytorch_model.safetensors.index.json",
                }:
                    continue
                if f.suffix in {".json", ".txt", ".model", ".yaml", ".yml"} or f.name.endswith(".jinja"):
                    dst = out_comp / f.name
                    if not dst.exists():
                        shutil.copy2(str(f.resolve()), str(dst))
        _log(f"[layer-delta] written to {output_path}")

    duration = time.perf_counter() - t0
    return {
        "status": "succeeded",
        "started_at": started_at,
        "ended_at": utc_now(),
        "duration_seconds": round(duration, 2),
        "output_checkpoint": str(output_path),
        "declared_output_checkpoint": plan["declared_output_checkpoint"],
        "modified_tensors": modified,
        "skipped_tensors": skipped,
        "total_tensors": len(merged),
        "block_window": {"start": block_start, "end": block_end},
    }


def fuse_tri_capability(
    *,
    foundation_id: str,
    edit_donor_id: str,
    layer_donor_id: str,
    delta_base_id: str,
    edit_coefficient: float = 0.35,
    layer_coefficient: float = 0.25,
    layer_block_start: int = 40,
    layer_block_end: int = 60,
    run_dir: Path,
    remote_config: str | None = None,
) -> dict[str, Any]:
    """Plan and execute the full tri-capability merge:

    Stage A: whole-backbone delta-edit (generation + editing)
    Stage B: windowed layer-delta on blocks [40,60) (layering)

    Returns a combined result dict with both stage results and the
    final tri-capable checkpoint path.
    """
    # Stage A — core delta-edit
    plan_a = plan_fusion(
        foundation_id=foundation_id,
        edit_donor_id=edit_donor_id,
        delta_base_id=delta_base_id,
        coefficient=edit_coefficient,
        run_dir=run_dir,
        remote_config=remote_config,
    )
    result_a = execute_fusion(plan_a)
    if result_a["status"] != "succeeded":
        raise FusionError(f"Stage A (core delta-edit) failed: {result_a}")

    # Stage B — windowed layer-delta
    plan_b = plan_layer_delta(
        prev_checkpoint=result_a["output_checkpoint"],
        layer_donor_id=layer_donor_id,
        delta_base_id=delta_base_id,
        coefficient=layer_coefficient,
        block_start=layer_block_start,
        block_end=layer_block_end,
        run_dir=run_dir,
        remote_config=remote_config,
    )
    result_b = execute_layer_delta(plan_b)
    if result_b["status"] != "succeeded":
        raise FusionError(f"Stage B (layer-delta) failed: {result_b}")

    return {
        "status": "succeeded",
        "output_checkpoint": result_b["output_checkpoint"],
        "declared_output_checkpoint": result_b["declared_output_checkpoint"],
        "stage_a": result_a,
        "stage_b": result_b,
        "capabilities": ["image-generation", "image-editing", "image-layering"],
        "total_duration_seconds": round(
            result_a["duration_seconds"] + result_b["duration_seconds"], 2
        ),
        "metrics": {
            "edit_coefficient": edit_coefficient,
            "layer_coefficient": layer_coefficient,
            "block_window": f"{layer_block_start}-{layer_block_end}",
            "stage_a_modified": result_a["modified_tensors"],
            "stage_b_modified": result_b["modified_tensors"],
        },
    }


# ── Selective SLERP (generation + editing only) ──────────────────────

_SLERP_BLOCK_WEIGHTS: dict[str, float] = {
    # non-block layers (embeddings, heads, I/O projections)  — more edit weight
    "non_block": 0.4,
    # block 0 — input patch embedding, moderately edit-specific
    "block_0": 0.3,
    # blocks 1–39 — early/mid backbone, generation quality lives here
    "block_early": 0.1,
    # blocks 40–58 — late semantic blocks, edit semantics concentrated here
    "block_late": 0.25,
    # block 59 — final output block, divergent but sensitive
    "block_final": 0.15,
}


def _slerp(v0: "torch.Tensor", v1: "torch.Tensor", t: float) -> "torch.Tensor":
    """Spherical linear interpolation between two flat tensors."""
    import torch
    import math
    v0f = v0.float().flatten()
    v1f = v1.float().flatten()
    dot = torch.clamp((v0f * v1f).sum() / (v0f.norm() * v1f.norm() + 1e-8), -1.0, 1.0)
    omega = torch.acos(dot).item()
    if abs(omega) < 1e-6:
        # Nearly identical — linear interpolation is fine
        result = (1.0 - t) * v0f + t * v1f
    else:
        sin_omega = math.sin(omega)
        result = (math.sin((1.0 - t) * omega) / sin_omega) * v0f + \
                 (math.sin(t * omega) / sin_omega) * v1f
    return result.reshape(v0.shape).to(v0.dtype)


def _slerp_weight_for_key(key: str, weights: dict[str, float]) -> float:
    """Return the SLERP t value (edit fraction) for a given tensor key."""
    import re
    m = re.search(r"\.(?:transformer_)?blocks?\.(\d+)\.", key)
    if not m:
        return weights["non_block"]
    idx = int(m.group(1))
    if idx == 0:
        return weights["block_0"]
    if idx <= 39:
        return weights["block_early"]
    if idx <= 58:
        return weights["block_late"]
    return weights["block_final"]  # block 59


def fuse_slerp_selective(
    *,
    gen_model_id: str,
    edit_model_id: str,
    run_dir: Path,
    block_weights: dict[str, float] | None = None,
    remote_config: str | None = None,
) -> dict[str, Any]:
    """Selective per-block SLERP between Gen2512 and Edit2511.

    Unlike delta-edit, this merges *directly* between the two fine-tuned
    models without needing the base model.  Each tensor is interpolated with
    a t value chosen by its block position:

    - Non-block layers (embeddings, heads):   t=0.4 (more edit)
    - Block 0 (input patch embed):            t=0.3
    - Blocks 1–39 (early/mid backbone):       t=0.1 (preserve gen quality)
    - Blocks 40–58 (late semantic):           t=0.25 (edit semantics live here)
    - Block 59 (final output):                t=0.15

    t=0 means 100% gen, t=1 means 100% edit.
    """
    _require_runtime_deps()
    import torch

    bw = {**_SLERP_BLOCK_WEIGHTS, **(block_weights or {})}
    output_path = run_dir / "merge" / "merged-tri-capability-checkpoint"
    log_path = run_dir / "merge" / "merge.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()
    t0 = time.perf_counter()

    from qwen_image_19.remote import default_remote_context
    ctx = default_remote_context(remote_config)

    with log_path.open("w", encoding="utf-8") as log:
        def _log(msg: str) -> None:
            log.write(msg + "\n")
            log.flush()

        _log(f"[slerp-selective] started={started_at}")
        _log(f"[slerp-selective] gen={gen_model_id}  edit={edit_model_id}")
        _log(f"[slerp-selective] block_weights={bw}")

        gen_path = _download_or_locate(gen_model_id)
        edit_path = _download_or_locate(edit_model_id)
        _log(f"[slerp-selective] gen_path={gen_path}")
        _log(f"[slerp-selective] edit_path={edit_path}")

        _log("[slerp-selective] loading gen weights …")
        gen_w = _load_checkpoint(gen_path)
        _log(f"[slerp-selective] gen loaded: {len(gen_w)} tensors")

        _log("[slerp-selective] loading edit weights …")
        edit_w = _load_checkpoint(edit_path)
        _log(f"[slerp-selective] edit loaded: {len(edit_w)} tensors")

        merged: dict[str, Any] = {}
        blended = passthrough = skipped = 0

        try:
            from tqdm.auto import tqdm as _tqdm
            iterator = _tqdm(list(gen_w.keys()), desc="merging", unit="tensor")
        except ImportError:
            iterator = list(gen_w.keys())  # type: ignore[assignment]

        for key in iterator:
            g_tensor = gen_w[key]

            if key not in edit_w:
                # Edit model missing this key — keep gen
                merged[key] = g_tensor
                passthrough += 1
                continue

            e_tensor = edit_w[key]

            if g_tensor.shape != e_tensor.shape:
                merged[key] = g_tensor
                skipped += 1
                continue

            # Only SLERP transformer backbone tensors
            if not (key.startswith("transformer_blocks.")
                    or "transformer_blocks." in key
                    or "joint_transformer_blocks." in key):
                # Non-block tensor: use non_block weight
                t = bw["non_block"]
            else:
                t = _slerp_weight_for_key(key, bw)

            if t <= 0.0:
                merged[key] = g_tensor
            elif t >= 1.0:
                merged[key] = e_tensor
            else:
                merged[key] = _slerp(g_tensor, e_tensor, t)
                blended += 1
                continue

            passthrough += 1

        _log(f"[slerp-selective] blended={blended}  passthrough={passthrough}  skipped={skipped}")

        output_path.mkdir(parents=True, exist_ok=True)
        _save_checkpoint(merged, output_path)
        _copy_config_files(gen_path, output_path)

        # Copy pipeline component dirs (tokenizer, scheduler, vae, etc.) from gen
        for comp_dir in gen_path.iterdir():
            if not comp_dir.is_dir() or comp_dir.name.startswith("."):
                continue
            out_comp = output_path / comp_dir.name
            out_comp.mkdir(parents=True, exist_ok=True)
            for f in comp_dir.iterdir():
                if not f.is_file():
                    continue
                if f.name in {
                    "model.safetensors.index.json",
                    "diffusion_pytorch_model.safetensors.index.json",
                }:
                    continue
                if f.suffix in {".json", ".txt", ".model", ".yaml", ".yml"} or f.name.endswith(".jinja"):
                    dst = out_comp / f.name
                    if not dst.exists():
                        shutil.copy2(str(f.resolve()), str(dst))

        # Copy processor from edit model (required by QwenImageEditPipeline)
        # Resolve symlinks explicitly — HF cache stores blobs as relative symlinks
        # that break when the directory is copied outside the cache tree.
        edit_path = Path(edit_model_id)
        processor_src = edit_path / "processor"
        if processor_src.is_dir():
            processor_dst = output_path / "processor"
            processor_dst.mkdir(parents=True, exist_ok=True)
            for f in processor_src.iterdir():
                if f.is_file() or f.is_symlink():
                    shutil.copy2(str(f.resolve()), str(processor_dst / f.name))
            _log(f"[slerp-selective] copied processor from edit model")

        # Update model_index.json to declare edit pipeline class and processor
        index_path = output_path / "model_index.json"
        if index_path.exists():
            import json as _json
            idx = _json.loads(index_path.read_text())
            idx["_class_name"] = "QwenImageEditPlusPipeline"
            idx["processor"] = ["transformers", "Qwen2VLProcessor"]
            index_path.write_text(_json.dumps(idx, indent=2, sort_keys=True) + "\n")

        _log(f"[slerp-selective] written to {output_path}")

    duration = time.perf_counter() - t0
    return {
        "status": "succeeded",
        "started_at": started_at,
        "ended_at": utc_now(),
        "duration_seconds": round(duration, 2),
        "output_checkpoint": str(output_path),
        "declared_output_checkpoint": (
            f"{ctx['artifact_dir']}/runs/{run_dir.name}/merge/merged-tri-capability-checkpoint"
        ),
        "blended_tensors": blended,
        "passthrough_tensors": passthrough,
        "skipped_tensors": skipped,
        "total_tensors": len(merged),
        "block_weights": bw,
        "capabilities": ["image-generation", "image-editing"],
        "metrics": {
            "strategy": "slerp-selective",
            "blended": blended,
            "block_weights": bw,
        },
    }
