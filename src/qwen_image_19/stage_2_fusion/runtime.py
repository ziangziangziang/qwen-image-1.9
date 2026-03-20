from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover - handled by runtime checks
    torch = None  # type: ignore


class Stage2HardwareError(RuntimeError):
    """Raised when Stage 2 runtime hardware requirements are not satisfied."""


@dataclass(frozen=True)
class Stage2DiffusionRuntimeConfig:
    required_gpus: int
    required_total_vram_gb: float
    selected_gpu_indices: tuple[int, ...]
    detected_gpu_count: int
    detected_total_vram_gb: float
    selected_gpu_vram_gb: tuple[float, ...]
    max_memory: dict[int, str]

    @property
    def primary_device(self) -> str:
        return f"cuda:{self.selected_gpu_indices[0]}"

    @property
    def pipeline_load_kwargs(self) -> dict[str, Any]:
        return {
            "device_map": "balanced",
            "max_memory": self.max_memory,
        }

    def summary(self) -> str:
        selected = ", ".join(f"cuda:{idx}={vram:.2f}GB" for idx, vram in zip(self.selected_gpu_indices, self.selected_gpu_vram_gb))
        return (
            f"required={self.required_gpus} GPUs / {self.required_total_vram_gb:.2f}GB total, "
            f"detected={self.detected_gpu_count} GPUs / {self.detected_total_vram_gb:.2f}GB total, "
            f"selected=[{selected}]"
        )


def _to_decimal_gb(bytes_value: int) -> float:
    return bytes_value / 1_000_000_000


def _bounded_max_memory_mib(total_bytes: int) -> int:
    # Keep allocator headroom to reduce fragmentation/OOM spikes.
    reserved_bytes = max(int(total_bytes * 0.05), 1 * 1024**3)
    bounded = max(total_bytes - reserved_bytes, 1024**3)
    return max(1024, bounded // (1024**2))


def resolve_stage2_diffusion_runtime(
    *,
    required_gpus: int,
    required_total_vram_gb: float,
) -> Stage2DiffusionRuntimeConfig:
    if required_gpus < 1:
        raise Stage2HardwareError(f"`required_gpus` must be >= 1, got {required_gpus}.")
    if required_total_vram_gb <= 0:
        raise Stage2HardwareError(
            f"`required_total_vram_gb` must be > 0, got {required_total_vram_gb}."
        )
    if torch is None:
        raise Stage2HardwareError(
            "torch is required for Stage 2 diffusion runtime checks but is not installed."
        )
    if not torch.cuda.is_available():
        raise Stage2HardwareError("CUDA is not available; Stage 2 diffusion requires GPU.")

    detected_gpu_count = int(torch.cuda.device_count())
    if detected_gpu_count < required_gpus:
        raise Stage2HardwareError(
            "Insufficient visible GPUs for Stage 2 diffusion. "
            f"Required {required_gpus}, detected {detected_gpu_count}. "
            "Set CUDA_VISIBLE_DEVICES to expose enough GPUs."
        )

    selected_gpu_indices = tuple(range(required_gpus))
    all_total_bytes = [
        int(torch.cuda.get_device_properties(index).total_memory)
        for index in range(detected_gpu_count)
    ]
    selected_bytes = [all_total_bytes[index] for index in selected_gpu_indices]
    selected_vram_gb = tuple(_to_decimal_gb(value) for value in selected_bytes)
    detected_total_vram_gb = _to_decimal_gb(sum(all_total_bytes))
    selected_total_vram_gb = sum(selected_vram_gb)
    required_per_gpu_gb = required_total_vram_gb / required_gpus

    underpowered = [
        (index, capacity)
        for index, capacity in zip(selected_gpu_indices, selected_vram_gb)
        if capacity + 1e-6 < required_per_gpu_gb
    ]
    if underpowered:
        details = ", ".join(
            f"cuda:{index}={capacity:.2f}GB (< {required_per_gpu_gb:.2f}GB required)"
            for index, capacity in underpowered
        )
        raise Stage2HardwareError(
            "Selected GPUs do not meet per-device VRAM requirements for Stage 2 diffusion. "
            + details
        )
    if selected_total_vram_gb + 1e-6 < required_total_vram_gb:
        raise Stage2HardwareError(
            "Selected GPUs do not meet total VRAM requirement for Stage 2 diffusion. "
            f"Required {required_total_vram_gb:.2f}GB, selected total {selected_total_vram_gb:.2f}GB."
        )

    max_memory: dict[int, str] = {}
    for index, total_bytes in zip(selected_gpu_indices, selected_bytes):
        max_memory[index] = f"{_bounded_max_memory_mib(total_bytes)}MiB"

    return Stage2DiffusionRuntimeConfig(
        required_gpus=required_gpus,
        required_total_vram_gb=float(required_total_vram_gb),
        selected_gpu_indices=selected_gpu_indices,
        detected_gpu_count=detected_gpu_count,
        detected_total_vram_gb=detected_total_vram_gb,
        selected_gpu_vram_gb=selected_vram_gb,
        max_memory=max_memory,
    )
