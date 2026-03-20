from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


class FakePipe:
    def __init__(
        self,
        model_id: str,
        load_kwargs: dict[str, object],
        output_factory=None,
    ) -> None:
        self.model_id = model_id
        self.load_kwargs = load_kwargs
        self.calls: list[dict[str, object]] = []
        self._counter = 0
        self._output_factory = output_factory

    def set_progress_bar_config(self, disable: bool) -> None:
        _ = disable

    def __call__(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        self._counter += 1
        if self._output_factory is None:
            image = SimpleNamespace(tag=f"{self.model_id}-image-{self._counter}")
        else:
            image = self._output_factory(self, kwargs)
        return SimpleNamespace(images=[image])


class FakeDiffusionPipeline:
    created: list[FakePipe] = []

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs: object) -> FakePipe:
        pipe = FakePipe(model_id, load_kwargs=dict(kwargs))
        cls.created.append(pipe)
        return pipe


class FakeImage:
    def __init__(self, mode: str, size: tuple[int, int], color=None) -> None:
        self.mode = mode
        self.size = size
        self.color = color

    def convert(self, mode: str):
        return FakeImage(mode, self.size, self.color)

    def save(self, _path) -> None:
        return None


def build_fake_torch_module() -> ModuleType:
    module = ModuleType("torch")

    class FakeOutOfMemoryError(RuntimeError):
        pass

    class FakeGenerator:
        def __init__(self, device: str | None = None) -> None:
            self.device = device
            self.seed = None

        def manual_seed(self, seed: int):
            self.seed = seed
            return self

    module.OutOfMemoryError = FakeOutOfMemoryError  # type: ignore[attr-defined]
    module.Generator = FakeGenerator  # type: ignore[attr-defined]
    module.bfloat16 = object()  # type: ignore[attr-defined]
    module.float32 = object()  # type: ignore[attr-defined]
    module.cuda = SimpleNamespace(empty_cache=lambda: None)  # type: ignore[attr-defined]
    return module


def build_fake_pil_modules() -> tuple[ModuleType, ModuleType]:
    pil_module = ModuleType("PIL")
    pil_image_module = ModuleType("PIL.Image")
    pil_image_module.Image = FakeImage  # type: ignore[attr-defined]
    pil_image_module.new = lambda mode, size, color=None: FakeImage(mode, size, color)  # type: ignore[attr-defined]
    pil_image_module.alpha_composite = (  # type: ignore[attr-defined]
        lambda left, right: FakeImage("RGBA", left.size, (left, right))
    )
    pil_module.Image = pil_image_module  # type: ignore[attr-defined]
    return pil_module, pil_image_module


def load_script_module(script_filename: str, module_name: str):
    root = Path(__file__).resolve().parent.parent
    script_path = root / "scripts" / script_filename
    fake_torch = build_fake_torch_module()
    fake_diffusers = ModuleType("diffusers")
    fake_diffusers.DiffusionPipeline = FakeDiffusionPipeline  # type: ignore[attr-defined]
    fake_safetensors = ModuleType("safetensors")
    fake_safetensors_torch = ModuleType("safetensors.torch")
    fake_safetensors_torch.save_file = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    fake_safetensors.torch = fake_safetensors_torch  # type: ignore[attr-defined]
    fake_pil, fake_pil_image = build_fake_pil_modules()
    FakeDiffusionPipeline.created = []

    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError(f"Unable to load module spec for {script_path}")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(
        "sys.modules",
        {
            "torch": fake_torch,
            "diffusers": fake_diffusers,
            "safetensors": fake_safetensors,
            "safetensors.torch": fake_safetensors_torch,
            "PIL": fake_pil,
            "PIL.Image": fake_pil_image,
        },
        clear=False,
    ):
        spec.loader.exec_module(module)
    return module


class Stage2EditInvocationTests(unittest.TestCase):
    def test_core_flow_passes_source_image_into_edit_call(self) -> None:
        module = load_script_module("stage-2-build-edit-delta.py", "stage2_build_edit_delta_test")
        runtime = SimpleNamespace(
            pipeline_load_kwargs={"device_map": "balanced", "max_memory": {0: "76000MiB", 1: "76000MiB"}},
            primary_device="cuda:0",
            summary=lambda: "runtime-summary",
        )
        with patch.object(module, "resolve_stage2_diffusion_runtime", return_value=runtime):
            source_image, edited_image = module.render_source_and_edit_images(
                foundation_model="Qwen/Qwen-Image-2512",
                edit_model="Qwen/Qwen-Image-Edit-2511",
                edit_instruction="replace background with white",
                steps=6,
                width=512,
                height=512,
                seed=1234,
                required_gpus=2,
                required_total_vram_gb=160,
                negative_prompt="low quality, blurry",
                true_cfg_scale=4.0,
                guidance_scale=1.0,
            )

        self.assertEqual(len(FakeDiffusionPipeline.created), 2)
        foundation_pipe, edit_pipe = FakeDiffusionPipeline.created
        self.assertEqual(foundation_pipe.model_id, "Qwen/Qwen-Image-2512")
        self.assertEqual(edit_pipe.model_id, "Qwen/Qwen-Image-Edit-2511")
        self.assertNotIn("image", foundation_pipe.calls[0])
        self.assertEqual(edit_pipe.calls[0]["prompt"], "replace background with white")
        self.assertIs(edit_pipe.calls[0]["image"], source_image)
        self.assertIsNot(source_image, edited_image)

    def test_dataset_edit_pair_uses_edit_instruction_with_image(self) -> None:
        module = load_script_module("stage-2-generate-teacher-dataset.py", "stage2_dataset_generation_test")
        runtime = SimpleNamespace(primary_device="cuda:0")
        source_pipe = FakePipe("Qwen/Qwen-Image-2512", load_kwargs={})
        edit_pipe = FakePipe("Qwen/Qwen-Image-Edit-2511", load_kwargs={})

        source_image, edited_image = module.render_edit_pair(
            source_pipe=source_pipe,
            edit_pipe=edit_pipe,
            runtime=runtime,
            source_prompt="a dog on the beach",
            edit_instruction="turn it into a watercolor painting",
            steps=6,
            width=512,
            height=512,
            seed=999,
        )

        self.assertEqual(len(source_pipe.calls), 1)
        self.assertEqual(source_pipe.calls[0]["prompt"], "a dog on the beach")
        self.assertNotIn("image", source_pipe.calls[0])
        self.assertEqual(len(edit_pipe.calls), 1)
        self.assertEqual(edit_pipe.calls[0]["prompt"], "turn it into a watercolor painting")
        self.assertIs(edit_pipe.calls[0]["image"], source_image)
        self.assertIsNot(source_image, edited_image)

    def test_dataset_non_edit_render_is_text_only(self) -> None:
        module = load_script_module("stage-2-generate-teacher-dataset.py", "stage2_dataset_non_edit_test")
        runtime = SimpleNamespace(primary_device="cuda:0")
        pipe = FakePipe("Qwen/Qwen-Image-2512", load_kwargs={})

        _ = module.render_image(
            pipe=pipe,
            runtime=runtime,
            prompt="studio product photo of a camera",
            steps=6,
            width=512,
            height=512,
            seed=7,
        )

        self.assertEqual(len(pipe.calls), 1)
        self.assertEqual(pipe.calls[0]["prompt"], "studio product photo of a camera")
        self.assertNotIn("image", pipe.calls[0])

    def test_layered_pair_uses_image_conditioned_call_and_defaults(self) -> None:
        module = load_script_module("stage-2-generate-teacher-dataset.py", "stage2_dataset_layered_pair_test")
        runtime = SimpleNamespace(primary_device="cuda:0")
        source_pipe = FakePipe(
            "Qwen/Qwen-Image-2512",
            load_kwargs={},
            output_factory=lambda _pipe, _kwargs: module.Image.new("RGB", (8, 8), (20, 40, 60)),
        )
        layered_pipe = FakePipe(
            "Qwen/Qwen-Image-Layered",
            load_kwargs={},
            output_factory=lambda _pipe, _kwargs: [
                module.Image.new("RGBA", (8, 8), (255, 0, 0, 120)),
                module.Image.new("RGBA", (8, 8), (0, 0, 255, 120)),
            ],
        )

        source_image, layered_image = module.render_layered_pair(
            source_pipe=source_pipe,
            layered_pipe=layered_pipe,
            runtime=runtime,
            prompt="single toy robot centered on seamless paper backdrop",
            steps=6,
            width=512,
            height=512,
            seed=121,
            negative_prompt="low quality",
            true_cfg_scale=4.0,
            guidance_scale=1.0,
        )

        self.assertEqual(source_image.mode, "RGB")
        self.assertEqual(layered_image.mode, "RGB")
        self.assertEqual(len(source_pipe.calls), 1)
        self.assertEqual(source_pipe.calls[0]["prompt"], "single toy robot centered on seamless paper backdrop")
        self.assertNotIn("image", source_pipe.calls[0])

        self.assertEqual(len(layered_pipe.calls), 1)
        layered_kwargs = layered_pipe.calls[0]
        self.assertIn("image", layered_kwargs)
        self.assertEqual(layered_kwargs["image"].mode, "RGBA")
        self.assertEqual(layered_kwargs["resolution"], 640)
        self.assertEqual(layered_kwargs["layers"], 4)
        self.assertTrue(layered_kwargs["cfg_normalize"])
        self.assertTrue(layered_kwargs["use_en_prompt"])
        self.assertNotIn("width", layered_kwargs)
        self.assertNotIn("height", layered_kwargs)

    def test_flatten_layered_output_to_rgb(self) -> None:
        module = load_script_module("stage-2-generate-teacher-dataset.py", "stage2_dataset_layered_flatten_test")
        flattened = module.flatten_layered_output_to_rgb(
            [
                module.Image.new("RGBA", (4, 4), (255, 0, 0, 128)),
                module.Image.new("RGBA", (4, 4), (0, 255, 0, 128)),
            ]
        )
        self.assertIsInstance(flattened, module.Image.Image)
        self.assertEqual(flattened.mode, "RGB")
        self.assertEqual(flattened.size, (4, 4))

    def test_layered_resolution_bucket_policy(self) -> None:
        module = load_script_module("stage-2-generate-teacher-dataset.py", "stage2_layered_resolution_bucket_test")
        self.assertEqual(module.resolve_layered_resolution_bucket(512, 512), 640)
        self.assertEqual(module.resolve_layered_resolution_bucket(928, 640), 1024)


if __name__ == "__main__":
    unittest.main()
