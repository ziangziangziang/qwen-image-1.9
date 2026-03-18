from __future__ import annotations

import json
import os
from pathlib import Path
import struct
import tempfile
import unittest

from qwen_image_19.stage_1_analysis import (
    Stage1AnalysisError,
    analyze,
    compare_state_dicts,
    inspect_cache_models,
    load_cache_alias_map,
    load_model_inventory,
    resolve_cache_snapshot,
)


def write_fake_safetensors(path: Path, tensors: dict[str, dict[str, object]]) -> None:
    header: dict[str, dict[str, object]] = {}
    offset = 0
    payload = bytearray()
    for name, spec in tensors.items():
        shape = [int(value) for value in spec["shape"]]  # type: ignore[index]
        nbytes = int(spec.get("nbytes", 4))  # type: ignore[union-attr]
        total_nbytes = nbytes
        for dim in shape:
            total_nbytes *= dim
        header[name] = {
            "dtype": spec.get("dtype", "F32"),
            "shape": shape,
            "data_offsets": [offset, offset + total_nbytes],
        }
        payload.extend(b"\x00" * total_nbytes)
        offset += total_nbytes
    header_bytes = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + payload)


def write_cache_model(
    hub_root: Path,
    cache_dir_name: str,
    commit: str,
    shards: dict[str, dict[str, dict[str, object]]],
    config: dict[str, object],
    write_index: bool = True,
) -> Path:
    model_root = hub_root / cache_dir_name
    snapshot = model_root / "snapshots" / commit
    snapshot.mkdir(parents=True, exist_ok=True)
    (model_root / "refs").mkdir(parents=True, exist_ok=True)
    (model_root / "refs" / "main").write_text(commit, encoding="utf-8")
    (snapshot / "config.json").write_text(json.dumps(config), encoding="utf-8")

    weight_map: dict[str, str] = {}
    for shard_name, tensor_map in shards.items():
        write_fake_safetensors(snapshot / shard_name, tensor_map)
        for tensor_name in tensor_map:
            weight_map[tensor_name] = shard_name

    if write_index:
        (snapshot / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": weight_map}, indent=2),
            encoding="utf-8",
        )
    return snapshot


def write_diffusers_style_cache_model(
    hub_root: Path,
    cache_dir_name: str,
    commit: str,
    components: dict[str, dict[str, dict[str, object]]],
    root_config: dict[str, object],
    component_configs: dict[str, dict[str, object]] | None = None,
) -> Path:
    model_root = hub_root / cache_dir_name
    snapshot = model_root / "snapshots" / commit
    snapshot.mkdir(parents=True, exist_ok=True)
    (model_root / "refs").mkdir(parents=True, exist_ok=True)
    (model_root / "refs" / "main").write_text(commit, encoding="utf-8")
    (snapshot / "model_index.json").write_text(json.dumps(root_config), encoding="utf-8")
    component_configs = component_configs or {}

    for component, tensor_map in components.items():
        component_dir = snapshot / component
        component_dir.mkdir(parents=True, exist_ok=True)
        write_fake_safetensors(component_dir / "model.safetensors", tensor_map)
        config_payload = component_configs.get(component, {"component": component})
        (component_dir / "config.json").write_text(json.dumps(config_payload), encoding="utf-8")

    return snapshot


class Stage1Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.hf_home = Path(self.tmpdir.name)
        self.hub_root = self.hf_home / "hub"
        self.metadata = load_model_inventory()
        self.cache_map = load_cache_alias_map()

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def create_full_mock_cache(self) -> None:
        foundation_tensors = {
            "model.mmdit.block1.weight": {"shape": [2, 2]},
            "model.mmdit.block2.weight": {"shape": [2, 2]},
            "text_encoder.layers.0.weight": {"shape": [2, 2]},
            "vae.encoder.conv_in.weight": {"shape": [8, 3, 3, 3]},
            "vae.decoder.conv_out.weight": {"shape": [3, 8, 3, 3]},
        }
        edit_tensors = {
            "model.mmdit.block1.weight": {"shape": [2, 2]},
            "model.mmdit.block2.weight": {"shape": [2, 2]},
            "text_encoder.layers.0.weight": {"shape": [2, 2]},
            "vae.encoder.conv_in.weight": {"shape": [8, 3, 3, 3]},
            "vae.decoder.conv_out.weight": {"shape": [3, 8, 3, 3]},
            "edit_heads.proj.weight": {"shape": [2, 2]},
        }
        base_tensors = {
            "model.mmdit.block1.weight": {"shape": [2, 2]},
            "model.mmdit.block2.weight": {"shape": [2, 2]},
            "text_encoder.layers.0.weight": {"shape": [2, 2]},
            "vae.encoder.conv_in.weight": {"shape": [8, 3, 3, 3]},
            "vae.decoder.conv_out.weight": {"shape": [3, 8, 3, 3]},
        }
        layered_tensors = {
            "model.mmdit.block1.weight": {"shape": [2, 2]},
            "model.mmdit.block2.weight": {"shape": [2, 2]},
            "text_encoder.layers.0.weight": {"shape": [2, 3]},
            "vae.encoder.conv_in.weight": {"shape": [8, 4, 3, 3]},
            "vae.decoder.conv_out.weight": {"shape": [4, 8, 3, 3]},
            "rope.layer3d.freqs": {"shape": [16]},
        }
        write_cache_model(
            self.hub_root,
            self.cache_map["qwen-image-2512"],
            "commit-2512",
            {
                "model-00001-of-00002.safetensors": {
                    "model.mmdit.block1.weight": foundation_tensors["model.mmdit.block1.weight"],
                    "model.mmdit.block2.weight": foundation_tensors["model.mmdit.block2.weight"],
                    "text_encoder.layers.0.weight": foundation_tensors["text_encoder.layers.0.weight"],
                },
                "model-00002-of-00002.safetensors": {
                    "vae.encoder.conv_in.weight": foundation_tensors["vae.encoder.conv_in.weight"],
                    "vae.decoder.conv_out.weight": foundation_tensors["vae.decoder.conv_out.weight"],
                },
            },
            {"rope_mode": "2D"},
        )
        write_cache_model(
            self.hub_root,
            self.cache_map["qwen-image-edit-2511"],
            "commit-edit",
            {"model.safetensors": edit_tensors},
            {"rope_mode": "2D"},
        )
        write_cache_model(
            self.hub_root,
            self.cache_map["qwen-image-base"],
            "commit-base",
            {"model.safetensors": base_tensors},
            {"rope_mode": "2D"},
        )
        write_cache_model(
            self.hub_root,
            self.cache_map["qwen-image-layered"],
            "commit-layered",
            {"model.safetensors": layered_tensors},
            {"rope_mode": "Layer3D"},
        )

    def test_resolves_hf_cache_snapshot(self) -> None:
        self.create_full_mock_cache()
        snapshot = resolve_cache_snapshot(self.hf_home, self.cache_map["qwen-image-2512"])
        self.assertEqual(snapshot["commit"], "commit-2512")
        self.assertTrue(str(snapshot["snapshot"]).endswith("commit-2512"))

    def test_analyze_produces_evidence_backed_matrix(self) -> None:
        self.create_full_mock_cache()
        result = analyze(dry_run=True, hf_home=self.hf_home)
        matrix = result["matrix"]
        self.assertEqual(matrix["inspection_mode"], "hf-cache-real-checkpoint")
        self.assertGreater(matrix["pairwise_comparisons"]["foundation_vs_edit"]["shared_key_count"], 0)
        self.assertEqual(matrix["summary"]["delta_merge"], 1)
        self.assertEqual(matrix["summary"]["adapter_only"], 2)
        self.assertEqual(matrix["summary"]["incompatible"], 1)
        self.assertIn("Pairwise comparison stats", result["report_preview"])

    def test_compare_state_dicts_uses_real_manifests(self) -> None:
        self.create_full_mock_cache()
        manifests = inspect_cache_models(self.hf_home, self.metadata, self.cache_map)
        comparison = compare_state_dicts(manifests)
        self.assertEqual(comparison["shared_key_count"], 5)
        self.assertEqual(comparison["missing_key_count"], 1)

    def test_missing_refs_main_raises(self) -> None:
        model_root = self.hub_root / self.cache_map["qwen-image-2512"]
        model_root.mkdir(parents=True, exist_ok=True)
        with self.assertRaises(Stage1AnalysisError):
            resolve_cache_snapshot(self.hf_home, self.cache_map["qwen-image-2512"])

    def test_missing_snapshot_raises(self) -> None:
        model_root = self.hub_root / self.cache_map["qwen-image-2512"]
        (model_root / "refs").mkdir(parents=True, exist_ok=True)
        (model_root / "refs" / "main").write_text("missing-commit", encoding="utf-8")
        with self.assertRaises(Stage1AnalysisError):
            resolve_cache_snapshot(self.hf_home, self.cache_map["qwen-image-2512"])

    def test_missing_index_for_multiple_shards_is_error(self) -> None:
        model_root = self.hub_root / self.cache_map["qwen-image-2512"]
        snapshot = model_root / "snapshots" / "commit-2512"
        snapshot.mkdir(parents=True, exist_ok=True)
        (model_root / "refs").mkdir(parents=True, exist_ok=True)
        (model_root / "refs" / "main").write_text("commit-2512", encoding="utf-8")
        (snapshot / "config.json").write_text(json.dumps({"rope_mode": "2D"}), encoding="utf-8")
        write_fake_safetensors(snapshot / "part-1.safetensors", {"a.weight": {"shape": [1]}})
        write_fake_safetensors(snapshot / "part-2.safetensors", {"b.weight": {"shape": [1]}})
        with self.assertRaises(Stage1AnalysisError):
            inspect_cache_models(self.hf_home, self.metadata, self.cache_map)

    def test_non_sharded_single_file_is_supported(self) -> None:
        write_cache_model(
            self.hub_root,
            self.cache_map["qwen-image-2512"],
            "commit-2512",
            {"model.safetensors": {"model.mmdit.block1.weight": {"shape": [2, 2]}}},
            {"rope_mode": "2D"},
        )
        write_cache_model(
            self.hub_root,
            self.cache_map["qwen-image-edit-2511"],
            "commit-edit",
            {"model.safetensors": {"model.mmdit.block1.weight": {"shape": [2, 2]}}},
            {"rope_mode": "2D"},
        )
        write_cache_model(
            self.hub_root,
            self.cache_map["qwen-image-base"],
            "commit-base",
            {"model.safetensors": {"vae.encoder.conv_in.weight": {"shape": [8, 3, 3, 3]}}},
            {"rope_mode": "2D"},
        )
        write_cache_model(
            self.hub_root,
            self.cache_map["qwen-image-layered"],
            "commit-layered",
            {"model.safetensors": {"vae.encoder.conv_in.weight": {"shape": [8, 4, 3, 3]}}},
            {"rope_mode": "Layer3D"},
        )
        manifests = inspect_cache_models(self.hf_home, self.metadata, self.cache_map)
        self.assertEqual(manifests["qwen-image-2512"]["shard_count"], 1)

    def test_diffusers_component_layout_is_supported(self) -> None:
        write_cache_model(
            self.hub_root,
            self.cache_map["qwen-image-2512"],
            "commit-2512",
            {"model.safetensors": {"transformer.block1.weight": {"shape": [2, 2]}}},
            {"rope_mode": "2D"},
        )
        write_cache_model(
            self.hub_root,
            self.cache_map["qwen-image-edit-2511"],
            "commit-edit",
            {"model.safetensors": {"transformer.block1.weight": {"shape": [2, 2]}}},
            {"rope_mode": "2D"},
        )
        write_diffusers_style_cache_model(
            self.hub_root,
            self.cache_map["qwen-image-base"],
            "commit-base",
            {
                "transformer": {"transformer.block1.weight": {"shape": [2, 2]}},
                "text_encoder": {"layers.0.weight": {"shape": [2, 2]}},
                "vae": {
                    "encoder.conv_in.weight": {"shape": [8, 3, 3, 3]},
                    "decoder.conv_out.weight": {"shape": [3, 8, 3, 3]},
                },
            },
            {"architectures": ["DiffusionPipeline"]},
            {
                "transformer": {"rope_mode": "2D"},
                "text_encoder": {"model_type": "qwen2_5_vl"},
                "vae": {"in_channels": 3, "out_channels": 3},
            },
        )
        write_diffusers_style_cache_model(
            self.hub_root,
            self.cache_map["qwen-image-layered"],
            "commit-layered",
            {
                "transformer": {"transformer.block1.weight": {"shape": [2, 2]}},
                "text_encoder": {"layers.0.weight": {"shape": [2, 3]}},
                "vae": {
                    "encoder.conv_in.weight": {"shape": [8, 4, 3, 3]},
                    "decoder.conv_out.weight": {"shape": [4, 8, 3, 3]},
                },
                "rope": {"layer3d.freqs": {"shape": [16]}},
            },
            {"architectures": ["DiffusionPipeline"]},
            {
                "transformer": {"rope_mode": "Layer3D"},
                "text_encoder": {"model_type": "qwen2_5_vl"},
                "vae": {"in_channels": 4, "out_channels": 4},
                "rope": {"name": "layer3d"},
            },
        )
        manifests = inspect_cache_models(self.hf_home, self.metadata, self.cache_map)
        self.assertEqual(manifests["qwen-image-base"]["layout"], "componentized")
        self.assertIn("vae", manifests["qwen-image-base"]["components"])
        self.assertGreater(manifests["qwen-image-base"]["component_tensor_counts"]["vae"], 0)

    def test_analyze_uses_hf_home_env_by_default(self) -> None:
        self.create_full_mock_cache()
        original_hf_home = os.environ.get("HF_HOME")
        os.environ["HF_HOME"] = str(self.hf_home)
        try:
            result = analyze(dry_run=True)
        finally:
            if original_hf_home is None:
                os.environ.pop("HF_HOME", None)
            else:
                os.environ["HF_HOME"] = original_hf_home
        self.assertEqual(result["hf_home"], str(self.hf_home))
        self.assertIn("Model snapshot inventory", result["report_preview"])


if __name__ == "__main__":
    unittest.main()
