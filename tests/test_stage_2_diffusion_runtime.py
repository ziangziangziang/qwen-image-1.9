from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from qwen_image_19.stage_2_fusion.runtime import (
    Stage2HardwareError,
    resolve_stage2_diffusion_runtime,
)


def _device_props(total_memory: int) -> SimpleNamespace:
    return SimpleNamespace(total_memory=total_memory)


def _fake_torch(*, cuda_available: bool, device_memories: list[int]) -> SimpleNamespace:
    def get_device_properties(index: int) -> SimpleNamespace:
        return _device_props(device_memories[index])

    cuda = SimpleNamespace(
        is_available=lambda: cuda_available,
        device_count=lambda: len(device_memories),
        get_device_properties=get_device_properties,
    )
    return SimpleNamespace(cuda=cuda)


class Stage2DiffusionRuntimeTests(unittest.TestCase):
    def test_resolve_runtime_fails_without_cuda(self) -> None:
        fake_torch = _fake_torch(cuda_available=False, device_memories=[])
        with patch("qwen_image_19.stage_2_fusion.runtime.torch", new=fake_torch):
            with self.assertRaises(Stage2HardwareError):
                resolve_stage2_diffusion_runtime(required_gpus=2, required_total_vram_gb=160)

    def test_resolve_runtime_fails_when_gpu_count_is_too_small(self) -> None:
        fake_torch = _fake_torch(cuda_available=True, device_memories=[80_000_000_000])
        with patch("qwen_image_19.stage_2_fusion.runtime.torch", new=fake_torch):
            with self.assertRaises(Stage2HardwareError):
                resolve_stage2_diffusion_runtime(required_gpus=2, required_total_vram_gb=160)

    def test_resolve_runtime_fails_when_selected_vram_is_insufficient(self) -> None:
        fake_torch = _fake_torch(
            cuda_available=True,
            device_memories=[70_000_000_000, 70_000_000_000],
        )
        with patch("qwen_image_19.stage_2_fusion.runtime.torch", new=fake_torch):
            with self.assertRaises(Stage2HardwareError):
                resolve_stage2_diffusion_runtime(required_gpus=2, required_total_vram_gb=160)

    def test_resolve_runtime_returns_balanced_multi_gpu_config(self) -> None:
        fake_torch = _fake_torch(
            cuda_available=True,
            device_memories=[
                80_000_000_000,
                80_000_000_000,
                40_000_000_000,
                40_000_000_000,
            ],
        )
        with patch("qwen_image_19.stage_2_fusion.runtime.torch", new=fake_torch):
            config = resolve_stage2_diffusion_runtime(required_gpus=2, required_total_vram_gb=160)

        self.assertEqual(config.selected_gpu_indices, (0, 1))
        self.assertEqual(config.primary_device, "cuda:0")
        self.assertEqual(config.pipeline_load_kwargs["device_map"], "balanced")
        self.assertEqual(set(config.pipeline_load_kwargs["max_memory"].keys()), {0, 1})
        self.assertTrue(all(value.endswith("MiB") for value in config.pipeline_load_kwargs["max_memory"].values()))


if __name__ == "__main__":
    unittest.main()
