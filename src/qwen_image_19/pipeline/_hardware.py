"""Hardware detection helpers."""
from __future__ import annotations


def require_gpus(*, min_gpus: int = 1, min_vram_gb: float = 40.0) -> None:
    """Raise if required GPU resources are not available."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for GPU operations") from exc

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    count = torch.cuda.device_count()
    if count < min_gpus:
        raise RuntimeError(f"Need {min_gpus} GPU(s) but found {count}")

    for i in range(count):
        props = torch.cuda.get_device_properties(i)
        vram_gb = props.total_mem / (1024 ** 3)
        if vram_gb < min_vram_gb:
            raise RuntimeError(
                f"GPU {i} has {vram_gb:.1f} GB VRAM, need {min_vram_gb:.1f} GB"
            )
