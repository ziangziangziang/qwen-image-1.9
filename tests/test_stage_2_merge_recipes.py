from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from qwen_image_19.stage_2_fusion import Stage2FusionError, fuse


MODELS = {
    "qwen-image-base": {"alias": "qwen-image-base", "model_id": "Qwen/Qwen-Image"},
    "qwen-image-2512": {"alias": "qwen-image-2512", "model_id": "Qwen/Qwen-Image-2512"},
    "qwen-image-edit-2511": {"alias": "qwen-image-edit-2511", "model_id": "Qwen/Qwen-Image-Edit-2511"},
    "qwen-image-layered": {"alias": "qwen-image-layered", "model_id": "Qwen/Qwen-Image-Layered"},
}


STAGE1_MATRIX = {
    "summary": {"direct_merge": 0, "delta_merge": 1, "adapter_only": 1, "incompatible": 2},
    "model_summaries": {
        "qwen-image-base": {"vae": {"label": "RGB"}, "rope": {"label": "2D-or-rotary"}},
        "qwen-image-2512": {"vae": {"label": "RGB"}, "rope": {"label": "2D-or-rotary"}},
        "qwen-image-edit-2511": {"vae": {"label": "RGB"}, "rope": {"label": "2D-or-rotary"}},
        "qwen-image-layered": {"vae": {"label": "RGBA"}, "rope": {"label": "Layer3D"}},
    },
    "pairwise_comparisons": {
        "foundation_vs_layered": {
            "top_mismatching_prefixes": [
                {"prefix": "transformer.time_text_embed"},
                {"prefix": "vae.decoder"},
            ]
        }
    },
    "subsystems": [
        {
            "subsystem": "mmdit_backbone",
            "structural_compatibility": "direct-merge",
            "recommended_merge_strategy": "delta-merge",
            "evidence": {"shared_key_count": 1933, "missing_key_count": 0, "shape_mismatch_count": 0},
        },
        {
            "subsystem": "vae",
            "structural_compatibility": "incompatible",
            "recommended_merge_strategy": "incompatible",
            "evidence": {"shared_key_count": 194, "missing_key_count": 0, "shape_mismatch_count": 3},
        },
        {
            "subsystem": "rope",
            "structural_compatibility": "incompatible",
            "recommended_merge_strategy": "incompatible",
            "evidence": {"shared_key_count": 0, "missing_key_count": 0, "shape_mismatch_count": 0},
        },
    ],
}


STAGE1_WEIGHT = {
    "foundation_vs_edit": {
        "models": ["qwen-image-2512", "qwen-image-edit-2511"],
        "shared_key_count": 2856,
        "exact_equal_tensor_ratio": 0.3246,
        "top_divergent_blocks": [
            {"layer_id": "mmdit_backbone:transformer_blocks:49"},
            {"layer_id": "mmdit_backbone:transformer_blocks:48"},
            {"layer_id": "mmdit_backbone:transformer_blocks:47"},
        ],
        "by_subsystem": {
            "mmdit_backbone": {"mean_block_relative_l2_delta": 0.2219},
            "text_encoder": {
                "mean_exact_tensor_match_ratio": 1.0,
                "mean_block_relative_l2_delta": 0.0,
            },
            "vae": {
                "mean_exact_tensor_match_ratio": 1.0,
                "mean_block_relative_l2_delta": 0.0,
            },
        },
    },
    "base_vs_layered": {
        "models": ["qwen-image-base", "qwen-image-layered"],
        "shared_key_count": 2853,
        "exact_equal_tensor_ratio": 0.2611,
        "top_divergent_blocks": [
            {"layer_id": "vae:decoder.up_blocks:3"},
        ],
        "by_subsystem": {
            "mmdit_backbone": {"mean_block_relative_l2_delta": 0.0819},
            "text_encoder": {
                "mean_exact_tensor_match_ratio": 1.0,
                "mean_block_relative_l2_delta": 0.0,
            },
            "vae": {
                "mean_exact_tensor_match_ratio": 0.0511,
                "mean_block_relative_l2_delta": 0.6357,
            },
        },
    },
    "foundation_vs_layered": {
        "models": ["qwen-image-2512", "qwen-image-layered"],
        "shared_key_count": 2853,
        "exact_equal_tensor_ratio": 0.2597,
        "top_divergent_blocks": [
            {"layer_id": "mmdit_backbone:transformer_blocks:49"},
            {"layer_id": "mmdit_backbone:transformer_blocks:48"},
            {"layer_id": "vae:decoder.up_blocks:3"},
        ],
        "by_subsystem": {
            "mmdit_backbone": {"mean_block_relative_l2_delta": 0.1828},
            "text_encoder": {
                "mean_exact_tensor_match_ratio": 1.0,
                "mean_block_relative_l2_delta": 0.0,
            },
            "vae": {
                "mean_exact_tensor_match_ratio": 0.0511,
                "mean_block_relative_l2_delta": 0.6357,
            },
        },
    },
}


class Stage2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def write_json_file(self, path: Path, payload: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def write_stage2_configs(self) -> None:
        config_dir = self.root / "configs" / "merge"
        self.write_json_file(
            config_dir / "stage-2-delta-edit.yaml",
            {
                "recipe_name": "stage-2-core-delta",
                "foundation_model": "qwen-image-2512",
                "delta_source": "qwen-image-edit-2511",
                "delta_base_candidate": "qwen-image-base",
                "target_components": ["transformer"],
                "target_subsystems": ["mmdit_backbone"],
                "excluded_subsystems": ["text_encoder", "vae", "rope"],
                "coefficient_sweep": [0.2, 0.3, 0.35, 0.4],
                "selection_rule": "Prefer edit retention without visible generation regression on the smoke suite.",
            },
        )
        self.write_json_file(
            config_dir / "stage-2-layered-bridge.yaml",
            {
                "recipe_name": "stage-2-layered-bridge",
                "foundation_model": "qwen-image-2512",
                "donor_model": "qwen-image-layered",
                "strategy": "learnable-bridge",
                "target_component": "transformer",
                "target_subsystems": ["mmdit_backbone"],
                "bridge_block_window": {"start": 40, "end": 60},
                "extra_parameter_paths": ["transformer.time_text_embed.addition_t_embedding.weight"],
                "freeze_policy": ["text_encoder", "vae", "rope", "transformer_blocks.0-39"],
                "trainable_modules": ["bridge_adapter", "per_block_gates"],
                "distillation_target": "rgb-output",
                "layered_output_adapter": "alpha-composite-to-rgb",
            },
        )
        self.write_json_file(
            config_dir / "stage-2-synthetic-dataset.yaml",
            {
                "recipe_name": "stage-2-synthetic-teacher-dataset",
                "output_root": "stage-2/datasets/teacher-db",
                "flatten_layered_rgba": "alpha-composite-to-rgb",
                "record_schema": [
                    "sample_id",
                    "teacher_model_alias",
                    "teacher_model",
                    "prompt",
                    "source_prompt",
                    "edit_instruction",
                    "seed",
                    "generation_settings",
                    "asset_paths",
                    "output_metadata",
                ],
                "splits": {
                    "generation_teacher": {
                        "teacher_model": "qwen-image-2512",
                        "task": "text-to-image",
                        "planned_sample_count": 8,
                        "seed_schedule": [101, 202, 303, 404],
                        "generation_settings": {"resolution": "1024x1024", "steps": 30},
                        "prompt_bank": ["prompt a", "prompt b", "prompt c", "prompt d"],
                    },
                    "edit_teacher": {
                        "teacher_model": "qwen-image-edit-2511",
                        "task": "generate-then-edit",
                        "planned_sample_count": 8,
                        "seed_schedule": [111, 222, 333, 444],
                        "generation_settings": {"resolution": "1024x1024", "steps": 30},
                        "prompt_bank": [
                            {"source_prompt": "source a", "edit_instruction": "edit a"},
                            {"source_prompt": "source b", "edit_instruction": "edit b"},
                            {"source_prompt": "source c", "edit_instruction": "edit c"},
                            {"source_prompt": "source d", "edit_instruction": "edit d"},
                        ],
                    },
                    "layered_teacher": {
                        "teacher_model": "qwen-image-layered",
                        "task": "layer-aware-generation",
                        "planned_sample_count": 8,
                        "seed_schedule": [121, 242, 363, 484],
                        "generation_settings": {"resolution": "1024x1024", "steps": 30},
                        "prompt_bank": ["layered a", "layered b", "layered c", "layered d"],
                    },
                },
            },
        )
        self.write_json_file(
            config_dir / "stage-2-run-profiles.yaml",
                {
                    "profiles": {
                        "smoke": {
                            "resource_profile": {"num_gpus": 2, "vram_target_gb": 160},
                            "limits": {
                                "core_candidate_id": "core-delta-w035",
                                "dataset_samples_per_split": 2,
                                "bridge_train_steps": 64,
                                "bridge_batch_size": 1,
                                "eval_prompt_count": 6,
                                "poc_steps": 6,
                                "poc_side": 512,
                                "poc_true_cfg_scale": 4.0,
                                "poc_guidance_scale": 1.0,
                                "poc_negative_prompt": "low quality, blurry, distorted text",
                            },
                        },
                        "full": {
                            "resource_profile": {"num_gpus": 2, "vram_target_gb": 160},
                            "limits": {
                                "core_candidate_id": "core-delta-w035",
                                "dataset_samples_per_split": 8,
                                "bridge_train_steps": 500,
                                "bridge_batch_size": 2,
                                "eval_prompt_count": 24,
                                "poc_steps": 30,
                                "poc_side": 1024,
                                "poc_true_cfg_scale": 4.0,
                                "poc_guidance_scale": 1.0,
                                "poc_negative_prompt": "low quality, blurry, distorted text",
                            },
                        },
                        "quality": {
                            "resource_profile": {"num_gpus": 2, "vram_target_gb": 160},
                            "limits": {
                                "core_candidate_id": "core-delta-w035",
                                "dataset_samples_per_split": 8,
                                "bridge_train_steps": 500,
                                "bridge_batch_size": 2,
                                "eval_prompt_count": 24,
                                "poc_steps": 50,
                                "poc_side": 1328,
                                "poc_true_cfg_scale": 4.0,
                                "poc_guidance_scale": 1.0,
                                "poc_negative_prompt": "low quality, blurry, distorted text",
                            },
                        },
                    },
                },
        )

    def write_stage1_artifacts(self) -> None:
        stage1_dir = self.root / "reports" / "stage-1"
        self.write_json_file(stage1_dir / "compatibility-matrix.json", STAGE1_MATRIX)
        self.write_json_file(stage1_dir / "weight-analysis.json", STAGE1_WEIGHT)

    def test_fuse_requires_canonical_stage1_artifacts(self) -> None:
        self.write_stage2_configs()
        with patch("qwen_image_19.stage_2_fusion.repo_root", return_value=self.root):
            with patch("qwen_image_19.stage_2_fusion.load_model_inventory", return_value=MODELS):
                with patch(
                    "qwen_image_19.stage_2_fusion.default_remote_context",
                    return_value={
                        "name": "remote-test",
                        "python": "python3",
                        "workdir": "/mnt/private/workdir",
                        "artifact_dir": "/mnt/private/artifacts",
                        "cache_dir": "/mnt/private/cache",
                    },
                ):
                    with self.assertRaises(Stage2FusionError):
                        fuse(dry_run=True)

    def test_fuse_dry_run_builds_dual_track_manifest(self) -> None:
        self.write_stage2_configs()
        self.write_stage1_artifacts()
        with patch("qwen_image_19.stage_2_fusion.repo_root", return_value=self.root):
            with patch("qwen_image_19.stage_2_fusion.load_model_inventory", return_value=MODELS):
                with patch(
                    "qwen_image_19.stage_2_fusion.default_remote_context",
                    return_value={
                        "name": "remote-test",
                        "python": "python3",
                        "workdir": "/mnt/private/workdir",
                        "artifact_dir": "/mnt/private/artifacts",
                        "cache_dir": "/mnt/private/cache",
                    },
                ):
                    result = fuse(dry_run=True)
        manifest = result["manifest"]
        self.assertIn("stage1_evidence", manifest)
        self.assertIn("core_delta_recipe", manifest)
        self.assertIn("core_delta_candidates", manifest)
        self.assertIn("selected_core_candidate", manifest)
        self.assertIn("layered_bridge_recipe", manifest)
        self.assertIn("dataset", manifest)
        self.assertIn("remote_jobs", manifest)
        self.assertIn("artifacts", manifest)
        self.assertIn("exclusions", manifest)
        self.assertEqual(manifest["core_delta_recipe"]["target_components"], ["transformer"])
        self.assertEqual(manifest["core_delta_recipe"]["coefficient_sweep"], [0.2, 0.3, 0.35, 0.4])
        self.assertEqual(manifest["selected_core_candidate"]["blend_weight"], 0.35)
        self.assertEqual(manifest["run_profile"], "full")
        self.assertEqual(manifest["execution_enabled"], False)
        self.assertEqual(manifest["execution_policy"], "overwrite")
        self.assertEqual(manifest["cleanup_performed"], False)
        self.assertEqual(manifest["resource_profile"]["num_gpus"], 2)
        self.assertEqual(manifest["layered_bridge_recipe"]["strategy"], "learnable-bridge")
        self.assertEqual(
            manifest["layered_bridge_recipe"]["bridge_block_window"],
            {"start": 40, "end": 60},
        )
        manifest_text = json.dumps(manifest, indent=2)
        self.assertNotIn("/mnt/private", manifest_text)
        self.assertIn("text_encoder", manifest["exclusions"])
        self.assertIn("vae", manifest["exclusions"])
        self.assertIn("rope", manifest["exclusions"])
        self.assertTrue(all(not path.startswith("/") for path in result["dataset_manifest"]["splits"]["generation_teacher"]["asset_root"].splitlines()))
        self.assertIn("No-Go List", result["report_preview"])
        self.assertIn("Stable Core Track", result["report_preview"])
        self.assertIn("Experimental Layered Bridge Track", result["report_preview"])
        self.assertIn("Run Mode", result["report_preview"])
        self.assertNotIn("/mnt/private", result["report_preview"])

    def test_fuse_smoke_run_reduces_plan_scope(self) -> None:
        self.write_stage2_configs()
        self.write_stage1_artifacts()
        artifact_root = self.root / "reports" / "stage-2"
        with patch("qwen_image_19.stage_2_fusion.repo_root", return_value=self.root):
            with patch("qwen_image_19.stage_2_fusion.load_model_inventory", return_value=MODELS):
                with patch(
                    "qwen_image_19.stage_2_fusion.default_remote_context",
                    return_value={"name": "remote-test", "python": "python3"},
                ):
                    with patch("qwen_image_19.stage_2_fusion.run_subprocess_job", return_value=(0, 0.01)):
                        with patch("qwen_image_19.stage_2_fusion.ensure_outputs_exist", return_value=[]):
                            result = fuse(artifact_dir=artifact_root, smoke_run=True)
        manifest = result["manifest"]
        self.assertEqual(result["run_profile"], "smoke")
        self.assertEqual(result["execution_policy"], "overwrite")
        self.assertEqual(result["execution_enabled"], True)
        self.assertEqual(manifest["run_profile"], "smoke")
        self.assertEqual(manifest["execution_enabled"], True)
        self.assertEqual(manifest["resource_profile"]["num_gpus"], 2)
        self.assertEqual(len(manifest["core_delta_candidates"]), 1)
        self.assertEqual(manifest["core_delta_candidates"][0]["candidate_id"], "core-delta-w035")
        self.assertEqual(manifest["dataset"]["split_counts"]["generation_teacher"], 2)
        self.assertEqual(manifest["dataset"]["split_counts"]["edit_teacher"], 2)
        self.assertEqual(manifest["dataset"]["split_counts"]["layered_teacher"], 2)
        self.assertEqual(result["dataset_manifest"]["splits"]["generation_teacher"]["planned_sample_count"], 2)
        self.assertIn("run_status", result)

    def test_fuse_rejects_dry_and_smoke_together(self) -> None:
        self.write_stage2_configs()
        self.write_stage1_artifacts()
        with patch("qwen_image_19.stage_2_fusion.repo_root", return_value=self.root):
            with patch("qwen_image_19.stage_2_fusion.load_model_inventory", return_value=MODELS):
                with patch("qwen_image_19.stage_2_fusion.default_remote_context", return_value={"name": "remote-test", "python": "python3"}):
                    with self.assertRaises(Stage2FusionError):
                        fuse(dry_run=True, smoke_run=True)

    def test_fuse_write_creates_stage2_folder_and_shims(self) -> None:
        self.write_stage2_configs()
        self.write_stage1_artifacts()
        artifact_root = self.root / "reports" / "stage-2"
        with patch("qwen_image_19.stage_2_fusion.repo_root", return_value=self.root):
            with patch("qwen_image_19.stage_2_fusion.load_model_inventory", return_value=MODELS):
                with patch(
                    "qwen_image_19.stage_2_fusion.default_remote_context",
                    return_value={
                        "name": "remote-test",
                        "python": "python3",
                        "workdir": "/mnt/private/workdir",
                        "artifact_dir": "/mnt/private/artifacts",
                        "cache_dir": "/mnt/private/cache",
                    },
                ):
                    result = fuse(artifact_dir=artifact_root)
        self.assertTrue((artifact_root / "README.md").exists())
        self.assertTrue((artifact_root / "merge-manifest.json").exists())
        self.assertTrue((artifact_root / "dataset-manifest.json").exists())
        self.assertTrue((artifact_root.parent / "stage-2-fusion-report.md").exists())
        self.assertTrue((artifact_root.parent / "stage-2-merge-manifest.json").exists())
        report_text = (artifact_root / "README.md").read_text(encoding="utf-8")
        self.assertIn("No-Go List", report_text)
        self.assertIn("Stable Core Track", report_text)
        self.assertIn("Experimental Layered Bridge Track", report_text)
        self.assertNotIn("/mnt/private", report_text)
        manifest_text = (artifact_root / "merge-manifest.json").read_text(encoding="utf-8")
        self.assertNotIn("/mnt/private", manifest_text)
        dataset_text = (artifact_root / "dataset-manifest.json").read_text(encoding="utf-8")
        self.assertNotIn("/mnt/private", dataset_text)
        self.assertIn("written", result)
        self.assertNotIn("run_status", result)

    def test_fuse_execute_writes_run_status(self) -> None:
        self.write_stage2_configs()
        self.write_stage1_artifacts()
        artifact_root = self.root / "reports" / "stage-2"
        with patch("qwen_image_19.stage_2_fusion.repo_root", return_value=self.root):
            with patch("qwen_image_19.stage_2_fusion.load_model_inventory", return_value=MODELS):
                with patch("qwen_image_19.stage_2_fusion.default_remote_context", return_value={"name": "remote-test", "python": "python3"}):
                    with patch("qwen_image_19.stage_2_fusion.run_subprocess_job", return_value=(0, 0.01)):
                        with patch("qwen_image_19.stage_2_fusion.ensure_outputs_exist", return_value=[]):
                            result = fuse(artifact_dir=artifact_root, smoke_run=True, execute=True)
        run_status = self.root / "stage-2" / "run-status.json"
        self.assertTrue(run_status.exists())
        status_payload = json.loads(run_status.read_text(encoding="utf-8"))
        self.assertEqual(status_payload["execution_policy"], "overwrite")
        self.assertEqual(status_payload["summary"]["failed_job"], None)
        core_delta_cmd = status_payload["jobs"]["core_delta_sweep"]["commands"][0]
        self.assertIn("--required-gpus", core_delta_cmd)
        self.assertIn("--required-total-vram-gb", core_delta_cmd)
        self.assertIn("--true-cfg-scale", core_delta_cmd)
        self.assertIn("--guidance-scale", core_delta_cmd)
        self.assertIn("--negative-prompt", core_delta_cmd)
        self.assertEqual(core_delta_cmd[core_delta_cmd.index("--required-gpus") + 1], "2")
        self.assertEqual(core_delta_cmd[core_delta_cmd.index("--required-total-vram-gb") + 1], "160")
        self.assertEqual(core_delta_cmd[core_delta_cmd.index("--true-cfg-scale") + 1], "4")
        self.assertEqual(core_delta_cmd[core_delta_cmd.index("--guidance-scale") + 1], "1")
        self.assertEqual(core_delta_cmd[core_delta_cmd.index("--negative-prompt") + 1], "low quality, blurry, distorted text")
        for job_name in ("core_smoke_eval", "teacher_dataset_generation", "experimental_smoke_eval"):
            command = status_payload["jobs"][job_name]["command"]
            self.assertIn("--required-gpus", command)
            self.assertIn("--required-total-vram-gb", command)
            self.assertIn("--true-cfg-scale", command)
            self.assertIn("--guidance-scale", command)
            self.assertIn("--negative-prompt", command)
            self.assertEqual(command[command.index("--required-gpus") + 1], "2")
            self.assertEqual(command[command.index("--required-total-vram-gb") + 1], "160")
            self.assertEqual(command[command.index("--true-cfg-scale") + 1], "4")
            self.assertEqual(command[command.index("--guidance-scale") + 1], "1")
            self.assertEqual(command[command.index("--negative-prompt") + 1], "low quality, blurry, distorted text")
        bridge_command = status_payload["jobs"]["layered_bridge_train"]["command"]
        self.assertNotIn("--required-gpus", bridge_command)
        self.assertNotIn("--required-total-vram-gb", bridge_command)
        for job_name in (
            "core_delta_sweep",
            "core_smoke_eval",
            "teacher_dataset_generation",
            "layered_bridge_train",
            "experimental_smoke_eval",
        ):
            self.assertIn(job_name, status_payload["jobs"])
            self.assertIn(status_payload["jobs"][job_name]["status"], ("succeeded", "skipped"))
        self.assertIn("run_status", result)
        self.assertEqual(result["execution_policy"], "overwrite")

    def test_fuse_full_profile_without_execute_is_plan_only(self) -> None:
        self.write_stage2_configs()
        self.write_stage1_artifacts()
        artifact_root = self.root / "reports" / "stage-2"
        with patch("qwen_image_19.stage_2_fusion.repo_root", return_value=self.root):
            with patch("qwen_image_19.stage_2_fusion.load_model_inventory", return_value=MODELS):
                with patch("qwen_image_19.stage_2_fusion.default_remote_context", return_value={"name": "remote-test", "python": "python3"}):
                    result = fuse(artifact_dir=artifact_root, run_profile="full")
        self.assertEqual(result["run_profile"], "full")
        self.assertEqual(result["execution_enabled"], False)
        self.assertNotIn("run_status", result)

    def test_fuse_execute_failure_marks_failed_job(self) -> None:
        self.write_stage2_configs()
        self.write_stage1_artifacts()
        artifact_root = self.root / "reports" / "stage-2"
        with patch("qwen_image_19.stage_2_fusion.repo_root", return_value=self.root):
            with patch("qwen_image_19.stage_2_fusion.load_model_inventory", return_value=MODELS):
                with patch("qwen_image_19.stage_2_fusion.default_remote_context", return_value={"name": "remote-test", "python": "python3"}):
                    with patch("qwen_image_19.stage_2_fusion.ensure_outputs_exist", return_value=[]):
                        with patch("qwen_image_19.stage_2_fusion.run_subprocess_job", side_effect=[(1, 0.01)]):
                            with self.assertRaises(Stage2FusionError):
                                fuse(artifact_dir=artifact_root, smoke_run=True, execute=True)
        run_status = self.root / "stage-2" / "run-status.json"
        status_payload = json.loads(run_status.read_text(encoding="utf-8"))
        self.assertEqual(status_payload["summary"]["failed_job"], "core_delta_sweep")

    def test_fuse_execute_resume_skips_succeeded_jobs(self) -> None:
        self.write_stage2_configs()
        self.write_stage1_artifacts()
        artifact_root = self.root / "reports" / "stage-2"
        run_status = self.root / "stage-2" / "run-status.json"
        run_status.parent.mkdir(parents=True, exist_ok=True)
        run_status.write_text(
            json.dumps(
                {
                    "stage": "stage2",
                    "run_profile": "smoke",
                    "jobs": {
                        "core_delta_sweep": {
                            "status": "succeeded",
                            "outputs": ["stage-2/artifacts/core-candidates/core-delta-w035/qwen-image-1.9-core-bf16.safetensors"],
                        }
                    },
                    "summary": {"failed_job": None, "resume_hint": None},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        output = self.root / "stage-2" / "artifacts" / "core-candidates" / "core-delta-w035" / "qwen-image-1.9-core-bf16.safetensors"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("ok\n", encoding="utf-8")
        with patch("qwen_image_19.stage_2_fusion.repo_root", return_value=self.root):
            with patch("qwen_image_19.stage_2_fusion.load_model_inventory", return_value=MODELS):
                with patch("qwen_image_19.stage_2_fusion.default_remote_context", return_value={"name": "remote-test", "python": "python3"}):
                    with patch("qwen_image_19.stage_2_fusion.run_subprocess_job", return_value=(0, 0.01)):
                        with patch("qwen_image_19.stage_2_fusion.ensure_outputs_exist", return_value=[]):
                            result = fuse(artifact_dir=artifact_root, smoke_run=True, execute=True, resume=True)
        self.assertIn("run_status", result)
        status_payload = json.loads((self.root / "stage-2" / "run-status.json").read_text(encoding="utf-8"))
        self.assertEqual(status_payload["jobs"]["core_delta_sweep"]["status"], "skipped")
        self.assertEqual(status_payload["execution_policy"], "resume")

    def test_fuse_execute_default_overwrite_ignores_prior_success(self) -> None:
        self.write_stage2_configs()
        self.write_stage1_artifacts()
        artifact_root = self.root / "reports" / "stage-2"
        run_status = self.root / "stage-2" / "run-status.json"
        run_status.parent.mkdir(parents=True, exist_ok=True)
        run_status.write_text(
            json.dumps(
                {
                    "stage": "stage2",
                    "run_profile": "smoke",
                    "jobs": {
                        "core_delta_sweep": {"status": "succeeded"},
                    },
                    "summary": {"failed_job": None, "resume_hint": None},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        with patch("qwen_image_19.stage_2_fusion.repo_root", return_value=self.root):
            with patch("qwen_image_19.stage_2_fusion.load_model_inventory", return_value=MODELS):
                with patch("qwen_image_19.stage_2_fusion.default_remote_context", return_value={"name": "remote-test", "python": "python3"}):
                    with patch("qwen_image_19.stage_2_fusion.run_subprocess_job", return_value=(0, 0.01)):
                        with patch("qwen_image_19.stage_2_fusion.ensure_outputs_exist", return_value=[]):
                            result = fuse(artifact_dir=artifact_root, smoke_run=True, execute=True)
        status_payload = json.loads(run_status.read_text(encoding="utf-8"))
        self.assertEqual(result["execution_policy"], "overwrite")
        self.assertEqual(status_payload["execution_policy"], "overwrite")
        self.assertEqual(status_payload["jobs"]["core_delta_sweep"]["status"], "succeeded")
        self.assertNotEqual(status_payload["jobs"]["core_delta_sweep"].get("skip_reason"), "already_succeeded")


if __name__ == "__main__":
    unittest.main()
