from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    import torch
    from safetensors.torch import load_file, save_file
except Exception:  # pragma: no cover - optional test dependency
    torch = None  # type: ignore[assignment]
    load_file = None  # type: ignore[assignment]
    save_file = None  # type: ignore[assignment]

from qwen_image_19.config_io import repo_root
from qwen_image_19.workflow import run_abliterate_step, run_merge, run_quantize_step, run_report
from qwen_image_19.webserver import resolve_api_request


def _repo_tempdir() -> tempfile.TemporaryDirectory[str]:
    return tempfile.TemporaryDirectory(dir=repo_root())


class PipelineWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = _repo_tempdir()
        self.runs_root = Path(self.tmpdir.name)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_merge_writes_run_manifest_and_step_bundle(self) -> None:
        fake_merge_result = {
            "mode": "write",
            "run_profile": "full",
            "execution_enabled": False,
            "artifact_dir": "reports/stage-2",
            "manifest": {
                "selected_core_candidate": {"output_checkpoint": "s3://artifacts/merged-model"},
                "core_delta_candidates": [{"candidate_id": "core-delta-w035"}],
            },
            "dataset_manifest": {
                "output_root": "reports/stage-2/datasets/teacher-db",
                "splits": {"generation_teacher": {}},
            },
        }
        with patch("qwen_image_19.workflow.fuse", return_value=fake_merge_result):
            result = run_merge(run_id="run-merge", artifact_dir=str(self.runs_root))
        run_dir = self.runs_root / "run-merge"
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(result["run_id"], "run-merge")
        self.assertEqual(manifest["steps"]["merge"]["status"], "completed")
        self.assertEqual(manifest["steps"]["merge"]["output_checkpoint"], "s3://artifacts/merged-model")
        self.assertTrue((run_dir / "merge" / "step-result.json").exists())
        self.assertTrue((run_dir / "report-index.json").exists())
        self.assertFalse(manifest["steps"]["merge"]["step_result"].startswith("/"))

    def test_abliterate_uses_merge_output_checkpoint(self) -> None:
        fake_merge_result = {
            "mode": "write",
            "run_profile": "full",
            "execution_enabled": False,
            "artifact_dir": "reports/stage-2",
            "manifest": {
                "selected_core_candidate": {"output_checkpoint": "s3://artifacts/merged-model"},
                "core_delta_candidates": [{"candidate_id": "core-delta-w035"}],
            },
            "dataset_manifest": {
                "output_root": "reports/stage-2/datasets/teacher-db",
                "splits": {"generation_teacher": {}},
            },
        }
        with patch("qwen_image_19.workflow.fuse", return_value=fake_merge_result):
            run_merge(run_id="run-abliterate", artifact_dir=str(self.runs_root))
        result = run_abliterate_step(run_id="run-abliterate", artifact_dir=str(self.runs_root))
        step_result = result["step_result"]
        self.assertEqual(step_result["input_checkpoint"], "s3://artifacts/merged-model")
        self.assertEqual(step_result["status"], "planned")
        self.assertIn("abliterated-model", step_result["output_checkpoint"])

    @unittest.skipIf(torch is None or save_file is None or load_file is None, "torch+safetensors not installed")
    def test_abliterate_execute_runs_worker_and_writes_execution_outputs(self) -> None:
        checkpoint_path = self.runs_root / "input-model.safetensors"
        save_file(
            {
                "model.layers.0.self_attn.o_proj.weight": torch.eye(4, dtype=torch.float32),
                "model.layers.0.mlp.down_proj.weight": torch.eye(4, dtype=torch.float32),
            },
            str(checkpoint_path),
        )
        measurements_path = self.runs_root / "measurements.pt"
        torch.save(
            {
                "refusenorm_0": torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                "harmless_0": torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float32),
            },
            measurements_path,
        )
        recipe_path = self.runs_root / "abliterate.yaml"
        recipe_path.write_text(
            "\n".join(
                [
                    f"measurements: {measurements_path.name}",
                    "normpreserve: false",
                    "projected: false",
                    "ablate:",
                    "  - layer: 0",
                    "    measurement: 0",
                    "    scale: 1.0",
                    "    tensor_patterns:",
                    '      - ".layers.0.self_attn.o_proj.weight"',
                ]
            ),
            encoding="utf-8",
        )
        fake_merge_result = {
            "mode": "write",
            "run_profile": "full",
            "execution_enabled": False,
            "artifact_dir": "reports/stage-2",
            "manifest": {
                "selected_core_candidate": {"output_checkpoint": str(checkpoint_path)},
                "core_delta_candidates": [{"candidate_id": "core-delta-w035"}],
            },
            "dataset_manifest": {
                "output_root": "reports/stage-2/datasets/teacher-db",
                "splits": {"generation_teacher": {}},
            },
        }
        with patch("qwen_image_19.workflow.fuse", return_value=fake_merge_result):
            run_merge(run_id="run-abliterate-exec", artifact_dir=str(self.runs_root))
        result = run_abliterate_step(
            run_id="run-abliterate-exec",
            artifact_dir=str(self.runs_root),
            execute=True,
            recipe_config=str(recipe_path),
        )
        step_result = result["step_result"]
        self.assertEqual(step_result["status"], "succeeded")
        output_checkpoint = self.runs_root / "run-abliterate-exec" / "abliterate" / "execution" / "abliterated-model.safetensors"
        self.assertTrue(output_checkpoint.exists())
        self.assertTrue((self.runs_root / "run-abliterate-exec" / "abliterate" / "execution" / "execution-manifest.json").exists())
        self.assertEqual(step_result["remote_job"]["status"], "succeeded")
        output_weights = load_file(str(output_checkpoint))
        self.assertFalse(
            torch.equal(
                output_weights["model.layers.0.self_attn.o_proj.weight"],
                torch.eye(4, dtype=torch.float32),
            )
        )

    @unittest.skipIf(torch is None or save_file is None, "torch+safetensors not installed")
    def test_abliterate_execute_requires_recipe(self) -> None:
        fake_merge_result = {
            "mode": "write",
            "run_profile": "full",
            "execution_enabled": False,
            "artifact_dir": "reports/stage-2",
            "manifest": {
                "selected_core_candidate": {"output_checkpoint": str(self.runs_root / "missing.safetensors")},
                "core_delta_candidates": [{"candidate_id": "core-delta-w035"}],
            },
            "dataset_manifest": {
                "output_root": "reports/stage-2/datasets/teacher-db",
                "splits": {"generation_teacher": {}},
            },
        }
        missing_checkpoint = self.runs_root / "missing.safetensors"
        save_file({"model.layers.0.self_attn.o_proj.weight": torch.eye(4, dtype=torch.float32)}, str(missing_checkpoint))
        with patch("qwen_image_19.workflow.fuse", return_value=fake_merge_result):
            run_merge(run_id="run-abliterate-recipe", artifact_dir=str(self.runs_root))
        with self.assertRaises(ValueError):
            run_abliterate_step(
                run_id="run-abliterate-recipe",
                artifact_dir=str(self.runs_root),
                execute=True,
            )

    @unittest.skipIf(torch is None or save_file is None, "torch+safetensors not installed")
    def test_quantize_uses_abliterate_output_checkpoint(self) -> None:
        fake_merge_result = {
            "mode": "write",
            "run_profile": "full",
            "execution_enabled": False,
            "artifact_dir": "reports/stage-2",
            "manifest": {
                "selected_core_candidate": {"output_checkpoint": str(self.runs_root / "quant-input.safetensors")},
                "core_delta_candidates": [{"candidate_id": "core-delta-w035"}],
            },
            "dataset_manifest": {
                "output_root": "reports/stage-2/datasets/teacher-db",
                "splits": {"generation_teacher": {}},
            },
        }
        save_file({"model.layers.0.self_attn.o_proj.weight": torch.eye(4, dtype=torch.float32)}, str(self.runs_root / "quant-input.safetensors"))
        measurements_path = self.runs_root / "quant-measurements.pt"
        torch.save(
            {
                "refusenorm_0": torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                "harmless_0": torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float32),
            },
            measurements_path,
        )
        recipe_path = self.runs_root / "quant-abliterate.yaml"
        recipe_path.write_text(
            "\n".join(
                [
                    f"measurements: {measurements_path.name}",
                    "ablate:",
                    "  - layer: 0",
                    "    measurement: 0",
                    "    tensor_patterns:",
                    '      - ".layers.0.self_attn.o_proj.weight"',
                ]
            ),
            encoding="utf-8",
        )
        fake_quantize_result = {
            "stage": "stage4",
            "artifact_dir": "reports/stage-4",
            "profiles": {"gguf": {}, "exl2-gptq": {}},
        }
        with patch("qwen_image_19.workflow.fuse", return_value=fake_merge_result):
            run_merge(run_id="run-quantize", artifact_dir=str(self.runs_root))
        abliterate_result = run_abliterate_step(
            run_id="run-quantize",
            artifact_dir=str(self.runs_root),
            execute=True,
            recipe_config=str(recipe_path),
        )
        with patch("qwen_image_19.workflow.legacy_quantize", return_value=fake_quantize_result):
            result = run_quantize_step(run_id="run-quantize", artifact_dir=str(self.runs_root))
        self.assertEqual(result["step_result"]["input_checkpoint"], abliterate_result["step_result"]["output_checkpoint"])
        manifest = json.loads((self.runs_root / "run-quantize" / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["steps"]["quantize"]["status"], "completed")
        self.assertTrue((self.runs_root / "run-quantize" / "quantize" / "eval-summary.json").exists())

    def test_report_command_writes_dashboard_index(self) -> None:
        fake_merge_result = {
            "mode": "write",
            "run_profile": "full",
            "execution_enabled": False,
            "artifact_dir": "reports/stage-2",
            "manifest": {
                "selected_core_candidate": {"output_checkpoint": "s3://artifacts/merged-model"},
                "core_delta_candidates": [{"candidate_id": "core-delta-w035"}],
            },
            "dataset_manifest": {
                "output_root": "reports/stage-2/datasets/teacher-db",
                "splits": {"generation_teacher": {}},
            },
        }
        with patch("qwen_image_19.workflow.fuse", return_value=fake_merge_result):
            run_merge(run_id="run-report", artifact_dir=str(self.runs_root))
        result = run_report(artifact_dir=str(self.runs_root))
        self.assertEqual(result["run_count"], 1)
        self.assertTrue((self.runs_root / "dashboard-index.json").exists())

    def test_results_server_contract_resolves_run_endpoints(self) -> None:
        fake_merge_result = {
            "mode": "write",
            "run_profile": "full",
            "execution_enabled": False,
            "artifact_dir": "reports/stage-2",
            "manifest": {
                "selected_core_candidate": {"output_checkpoint": "s3://artifacts/merged-model"},
                "core_delta_candidates": [{"candidate_id": "core-delta-w035"}],
            },
            "dataset_manifest": {
                "output_root": "reports/stage-2/datasets/teacher-db",
                "splits": {"generation_teacher": {}},
            },
        }
        with patch("qwen_image_19.workflow.fuse", return_value=fake_merge_result):
            run_merge(run_id="run-server", artifact_dir=str(self.runs_root))
        runs_status, runs, runs_content_type = resolve_api_request("/api/runs", self.runs_root)
        manifest_status, manifest, manifest_content_type = resolve_api_request("/api/runs/run-server", self.runs_root)
        step_status, step_result, step_content_type = resolve_api_request("/api/runs/run-server/steps/merge", self.runs_root)
        self.assertEqual(runs_status, 200)
        self.assertEqual(manifest_status, 200)
        self.assertEqual(step_status, 200)
        self.assertEqual(runs_content_type, "application/json")
        self.assertEqual(manifest_content_type, "application/json")
        self.assertEqual(step_content_type, "application/json")
        self.assertEqual(runs[0]["run_id"], "run-server")  # type: ignore[index]
        self.assertEqual(manifest["run_id"], "run-server")
        self.assertEqual(step_result["step"], "merge")


if __name__ == "__main__":
    unittest.main()
