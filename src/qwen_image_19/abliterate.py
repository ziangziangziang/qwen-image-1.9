from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any

try:
    import torch
except Exception:  # pragma: no cover - optional at import time
    torch = None  # type: ignore[assignment]

try:
    import yaml
except Exception:  # pragma: no cover - optional at import time
    yaml = None  # type: ignore[assignment]

try:
    from safetensors.torch import load_file, save_file
except Exception:  # pragma: no cover - optional at import time
    load_file = None  # type: ignore[assignment]
    save_file = None  # type: ignore[assignment]

from qwen_image_19.config_io import repo_root, write_json
from qwen_image_19.contracts import public_path, utc_now
from qwen_image_19.remote import default_remote_context

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional progress dependency
    tqdm = None  # type: ignore[assignment]


class AbliterationError(RuntimeError):
    """Raised when abliteration planning or execution fails."""


def _require_runtime_dependencies() -> None:
    missing: list[str] = []
    if torch is None:
        missing.append("torch")
    if yaml is None:
        missing.append("PyYAML")
    if load_file is None or save_file is None:
        missing.append("safetensors")
    if missing:
        raise AbliterationError(
            "Abliteration execution requires runtime dependencies that are not installed: "
            + ", ".join(missing)
        )


def _worker_output_root(run_dir: Path) -> Path:
    return run_dir / "abliterate" / "execution"


def _resolve_local_input(input_checkpoint: str) -> Path:
    path = Path(input_checkpoint)
    candidate = path if path.is_absolute() else repo_root() / path
    if not candidate.exists():
        raise AbliterationError(
            "Abliteration requires a local checkpoint path for execution. "
            f"`{input_checkpoint}` was not found from repo root."
        )
    return candidate


def _suggest_output_path(input_path: Path, worker_root: Path) -> Path:
    if input_path.is_dir():
        return worker_root / "abliterated-model"
    suffix = input_path.suffix or ".safetensors"
    return worker_root / f"abliterated-model{suffix}"


def plan_abliteration(
    *,
    input_checkpoint: str,
    run_dir: Path,
    recipe_config: str | None,
    remote_config: str | None = None,
    require_local_input: bool = True,
) -> dict[str, Any]:
    remote_context = default_remote_context(remote_config)
    worker_root = _worker_output_root(run_dir)
    worker_root.mkdir(parents=True, exist_ok=True)
    local_input = _resolve_local_input(input_checkpoint) if require_local_input else Path(input_checkpoint)
    local_output = _suggest_output_path(local_input if require_local_input else Path("abliterated-model.safetensors"), worker_root)
    execution_manifest = worker_root / "execution-manifest.json"
    log_path = worker_root / "execution.log"
    declared_output_checkpoint = (
        f"{remote_context['artifact_dir']}/runs/{run_dir.name}/abliterate/{local_output.name}"
    )
    command = [
        str(remote_context.get("python") or "python3"),
        "-m",
        "qwen_image_19.abliterate",
        "--execute-worker",
        "--input-checkpoint",
        str(local_input),
        "--output-checkpoint",
        str(local_output),
        "--declared-output-checkpoint",
        declared_output_checkpoint,
        "--execution-manifest",
        str(execution_manifest),
    ]
    if recipe_config:
        command.extend(["--recipe-config", recipe_config])
    return {
        "input_checkpoint": str(local_input),
        "output_checkpoint": str(local_output),
        "declared_output_checkpoint": declared_output_checkpoint,
        "execution_manifest": str(execution_manifest),
        "log_path": str(log_path),
        "recipe_config": recipe_config,
        "command": command,
        "remote_job": {
            "name": "abliterate-refusal-direction",
            "workdir": remote_context["workdir"],
            "artifact_dir": remote_context["artifact_dir"],
            "status": "planned",
        },
        "metrics": {
            "target_subspace": "refusal-behavior",
            "safety_review_required": True,
        },
    }


def execute_abliteration(plan: dict[str, Any]) -> dict[str, Any]:
    if not plan.get("recipe_config"):
        raise AbliterationError(
            "A real abliteration run requires `--recipe-config`. "
            "The previous placeholder worker has been removed."
        )
    log_file = Path(plan["log_path"])
    log_file.parent.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()
    started = time.perf_counter()
    with log_file.open("w", encoding="utf-8") as handle:
        env = os.environ.copy()
        src_path = str(repo_root() / "src")
        env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else f"{src_path}{os.pathsep}{env['PYTHONPATH']}"
        process = subprocess.run(
            plan["command"],
            cwd=repo_root(),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    duration = time.perf_counter() - started
    output_checkpoint = Path(plan["output_checkpoint"])
    execution_manifest = Path(plan["execution_manifest"])
    if process.returncode != 0:
        raise AbliterationError(
            "Abliteration worker failed. "
            f"See `{public_path(log_file)}` for details."
        )
    if not output_checkpoint.exists() or not execution_manifest.exists():
        raise AbliterationError(
            "Abliteration worker did not produce expected outputs. "
            f"Missing `{public_path(output_checkpoint)}` or `{public_path(execution_manifest)}`."
        )
    return {
        "status": "succeeded",
        "started_at": started_at,
        "ended_at": utc_now(),
        "duration_seconds": round(duration, 4),
        "log_path": str(log_file),
        "execution_manifest": str(execution_manifest),
        "output_checkpoint": str(output_checkpoint),
        "declared_output_checkpoint": plan["declared_output_checkpoint"],
        "exit_code": process.returncode,
    }


def magnitude_sparsify(tensor: torch.Tensor, fraction: float) -> torch.Tensor:
    if fraction >= 1.0:
        return tensor
    k = int(tensor.numel() * fraction)
    if k == 0:
        return torch.zeros_like(tensor)
    flat = tensor.flatten()
    threshold = torch.topk(flat.abs(), k, largest=True, sorted=False)[0].min()
    mask = tensor.abs() >= threshold
    return tensor * mask


def modify_tensor(W: torch.Tensor, direction: torch.Tensor, scale_factor: float = 1.0) -> torch.Tensor:
    original_dtype = W.dtype
    work = W.to(dtype=torch.float32)
    direction = torch.nn.functional.normalize(direction.to(dtype=torch.float32).view(-1), dim=0)
    rank = work.dim()
    if rank == 2:
        working = work.T
    elif rank == 3:
        working = work.permute(0, 2, 1)
    else:
        raise AbliterationError(f"Unsupported tensor rank for abliteration: {tuple(W.shape)}")
    projection = torch.matmul(working, direction)
    working = working - scale_factor * projection.unsqueeze(-1) * direction
    if rank == 2:
        result = working.T
    else:
        result = working.permute(0, 2, 1)
    return result.to(dtype=original_dtype)


def modify_tensor_norm_preserved(W: torch.Tensor, direction: torch.Tensor, scale_factor: float = 1.0) -> torch.Tensor:
    original_dtype = W.dtype
    work = W.to(dtype=torch.float32)
    direction = torch.nn.functional.normalize(direction.to(dtype=torch.float32).view(-1), dim=0)
    rank = work.dim()
    if rank == 2:
        working = work.T
    elif rank == 3:
        working = work.permute(0, 2, 1)
    else:
        raise AbliterationError(f"Unsupported tensor rank for abliteration: {tuple(W.shape)}")
    norms = torch.norm(working, dim=-1, keepdim=True)
    directions = torch.nn.functional.normalize(working, dim=-1)
    projection = torch.matmul(directions, direction)
    directions = directions - scale_factor * projection.unsqueeze(-1) * direction
    directions = torch.nn.functional.normalize(directions, dim=-1)
    modified = norms * directions
    if rank == 2:
        result = modified.T
    else:
        result = modified.permute(0, 2, 1)
    return result.to(dtype=original_dtype)


def modify_tensor_directional_scaling(W: torch.Tensor, direction: torch.Tensor, scale_factor: float = 1.0) -> torch.Tensor:
    original_dtype = W.dtype
    work = W.to(dtype=torch.float32)
    direction = torch.nn.functional.normalize(direction.to(dtype=torch.float32).view(-1), dim=0)
    rank = work.dim()
    if rank == 2:
        working = work.T
    elif rank == 3:
        working = work.permute(0, 2, 1)
    else:
        raise AbliterationError(f"Unsupported tensor rank for abliteration: {tuple(W.shape)}")
    alpha_minus_one = -scale_factor
    projection = torch.matmul(working, direction)
    working = working + alpha_minus_one * projection.unsqueeze(-1) * direction
    if rank == 2:
        result = working.T
    else:
        result = working.permute(0, 2, 1)
    return result.to(dtype=original_dtype)


def _load_recipe(recipe_config: str) -> dict[str, Any]:
    _require_runtime_dependencies()
    path = Path(recipe_config)
    if not path.is_absolute():
        path = repo_root() / path
    if not path.exists():
        raise AbliterationError(f"Recipe config `{recipe_config}` was not found.")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _load_measurements(recipe: dict[str, Any], recipe_path: str) -> dict[str, Any]:
    _require_runtime_dependencies()
    measurements_path = recipe.get("measurements")
    if not measurements_path:
        raise AbliterationError(f"Recipe `{recipe_path}` must define `measurements`.")
    path = Path(measurements_path)
    if not path.is_absolute():
        path = (Path(recipe_path).resolve().parent / path).resolve()
    if not path.exists():
        raise AbliterationError(f"Measurement file `{path}` was not found.")
    return torch.load(path, map_location="cpu")


def _build_default_patterns(layer: int) -> list[str]:
    return [
        f".layers.{layer}.self_attn.o_proj.weight",
        f".layers.{layer}.mlp.down_proj.weight",
        f".layers.{layer}.ffn.down_proj.weight",
    ]


def _compile_orders(recipe: dict[str, Any], measurements: dict[str, Any]) -> list[dict[str, Any]]:
    ablations = recipe.get("ablate", [])
    if not ablations:
        raise AbliterationError("Recipe must define at least one `ablate` entry.")
    projected = bool(recipe.get("projected", False))
    invert = bool(recipe.get("invert", False))
    global_scale = float(recipe.get("scale", 1.0))
    orders = []
    for item in ablations:
        layer = int(item["layer"])
        measurement = int(item.get("measurement", layer))
        refusal_key = f"refusenorm_{measurement}"
        harmless_key = f"harmless_{layer}"
        if refusal_key not in measurements:
            raise AbliterationError(f"Measurement file is missing `{refusal_key}`.")
        if projected and harmless_key not in measurements:
            raise AbliterationError(f"Projected abliteration requires `{harmless_key}`.")
        scale = float(item.get("scale", 1.0)) * global_scale
        if invert:
            scale = -scale
        orders.append(
            {
                "layer": layer,
                "measurement": measurement,
                "scale": scale,
                "sparsity": float(item.get("sparsity", 0.0)),
                "tensor_patterns": item.get("tensor_patterns", _build_default_patterns(layer)),
            }
        )
    return orders


def _measurement_direction(
    *,
    measurements: dict[str, Any],
    layer: int,
    measurement: int,
    projected: bool,
    sparsity: float,
) -> torch.Tensor:
    refusal = measurements[f"refusenorm_{measurement}"].double()
    refusal = torch.nn.functional.normalize(refusal, dim=0)
    if projected:
        harmless = measurements[f"harmless_{layer}"].double()
        harmless = torch.nn.functional.normalize(harmless, dim=0)
        refusal = refusal - (refusal @ harmless) * harmless
        refusal = refusal - (refusal @ harmless) * harmless
        refusal = torch.nn.functional.normalize(refusal, dim=0)
    if sparsity > 0.0:
        refusal = magnitude_sparsify(refusal, fraction=sparsity)
        refusal = torch.nn.functional.normalize(refusal, dim=0)
    return refusal


def _tensor_matches(key: str, patterns: list[str]) -> bool:
    return any(pattern in key for pattern in patterns)


def _copy_supporting_files(input_path: Path, output_path: Path) -> None:
    if input_path.is_file():
        return
    output_path.mkdir(parents=True, exist_ok=True)
    for candidate in input_path.iterdir():
        if candidate.is_dir():
            continue
        if candidate.suffix in {".json", ".txt", ".model"} or candidate.name.endswith(".jinja"):
            shutil.copy(candidate, output_path / candidate.name)


def _ablate_single_safetensors(
    input_path: Path,
    output_path: Path,
    orders: list[dict[str, Any]],
    measurements: dict[str, Any],
    recipe: dict[str, Any],
) -> dict[str, Any]:
    state_dict = load_file(str(input_path))
    modified_keys: list[str] = []
    iterator = state_dict.keys()
    if tqdm is not None:
        iterator = tqdm(list(state_dict.keys()), desc="abliterate tensors", unit="tensor")
    for key in iterator:
        for order in orders:
            if not _tensor_matches(key, order["tensor_patterns"]):
                continue
            direction = _measurement_direction(
                measurements=measurements,
                layer=order["layer"],
                measurement=order["measurement"],
                projected=bool(recipe.get("projected", False)),
                sparsity=float(order["sparsity"]),
            )
            if bool(recipe.get("directional", False)):
                state_dict[key] = modify_tensor_directional_scaling(state_dict[key], direction, order["scale"]).contiguous()
            elif bool(recipe.get("normpreserve", False)):
                state_dict[key] = modify_tensor_norm_preserved(state_dict[key], direction, order["scale"]).contiguous()
            else:
                state_dict[key] = modify_tensor(state_dict[key], direction, order["scale"]).contiguous()
            modified_keys.append(key)
    save_file(state_dict, str(output_path))
    if tqdm is not None:
        iterator.close()  # type: ignore[union-attr]
    return {"modified_keys": sorted(set(modified_keys))}


def _abliterate_sharded_checkpoint(
    input_path: Path,
    output_path: Path,
    orders: list[dict[str, Any]],
    measurements: dict[str, Any],
    recipe: dict[str, Any],
) -> dict[str, Any]:
    index_path = input_path / "model.safetensors.index.json"
    if not index_path.exists():
        raise AbliterationError(f"Sharded checkpoint `{input_path}` is missing model.safetensors.index.json.")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map", {})
    shard_to_keys: dict[str, list[str]] = {}
    for key, shard in weight_map.items():
        for order in orders:
            if _tensor_matches(key, order["tensor_patterns"]):
                shard_to_keys.setdefault(shard, []).append(key)
                break
    output_path.mkdir(parents=True, exist_ok=True)
    modified_keys: list[str] = []
    shard_names = sorted(set(weight_map.values()))
    iterator = shard_names
    if tqdm is not None:
        iterator = tqdm(shard_names, desc="abliterate shards", unit="shard")
    for shard_name in iterator:
        source = input_path / shard_name
        target = output_path / shard_name
        if shard_name not in shard_to_keys:
            shutil.copy(source, target)
            continue
        state_dict = load_file(str(source))
        for key in shard_to_keys[shard_name]:
            for order in orders:
                if not _tensor_matches(key, order["tensor_patterns"]):
                    continue
                direction = _measurement_direction(
                    measurements=measurements,
                    layer=order["layer"],
                    measurement=order["measurement"],
                    projected=bool(recipe.get("projected", False)),
                    sparsity=float(order["sparsity"]),
                )
                if bool(recipe.get("directional", False)):
                    state_dict[key] = modify_tensor_directional_scaling(state_dict[key], direction, order["scale"]).contiguous()
                elif bool(recipe.get("normpreserve", False)):
                    state_dict[key] = modify_tensor_norm_preserved(state_dict[key], direction, order["scale"]).contiguous()
                else:
                    state_dict[key] = modify_tensor(state_dict[key], direction, order["scale"]).contiguous()
                modified_keys.append(key)
                break
        save_file(state_dict, str(target))
    shutil.copy(index_path, output_path / "model.safetensors.index.json")
    _copy_supporting_files(input_path, output_path)
    if tqdm is not None:
        iterator.close()  # type: ignore[union-attr]
    return {"modified_keys": sorted(set(modified_keys))}


def run_worker(
    *,
    input_checkpoint: str,
    output_checkpoint: str,
    declared_output_checkpoint: str,
    execution_manifest: str,
    recipe_config: str,
) -> dict[str, Any]:
    _require_runtime_dependencies()
    input_path = Path(input_checkpoint)
    output_path = Path(output_checkpoint)
    manifest_path = Path(execution_manifest)
    recipe = _load_recipe(recipe_config)
    measurements = _load_measurements(recipe, recipe_config)
    orders = _compile_orders(recipe, measurements)
    if input_path.is_dir():
        result = _abliterate_sharded_checkpoint(input_path, output_path, orders, measurements, recipe)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result = _abliterate_single_safetensors(input_path, output_path, orders, measurements, recipe)
    manifest_payload = {
        "status": "succeeded",
        "created_at": utc_now(),
        "input_checkpoint": public_path(input_path),
        "output_checkpoint": public_path(output_path),
        "declared_output_checkpoint": declared_output_checkpoint,
        "recipe_config": public_path(Path(recipe_config)),
        "modified_keys": result["modified_keys"],
        "order_count": len(orders),
    }
    write_json(manifest_path, manifest_payload)
    return manifest_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m qwen_image_19.abliterate")
    parser.add_argument("--execute-worker", action="store_true", help="Run the abliteration worker.")
    parser.add_argument("--input-checkpoint", required=True, help="Input checkpoint path.")
    parser.add_argument("--output-checkpoint", required=True, help="Output checkpoint path.")
    parser.add_argument("--declared-output-checkpoint", required=True, help="Operator-facing output checkpoint reference.")
    parser.add_argument("--execution-manifest", required=True, help="Execution manifest JSON path.")
    parser.add_argument("--recipe-config", help="Abliteration recipe YAML path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.execute_worker:
        raise SystemExit("`--execute-worker` is required when invoking the abliteration module directly.")
    if not args.recipe_config:
        raise SystemExit("`--recipe-config` is required for real abliteration execution.")
    run_worker(
        input_checkpoint=args.input_checkpoint,
        output_checkpoint=args.output_checkpoint,
        declared_output_checkpoint=args.declared_output_checkpoint,
        execution_manifest=args.execution_manifest,
        recipe_config=args.recipe_config,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
