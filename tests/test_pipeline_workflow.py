from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from qwen_image_19.config_io import repo_root
from qwen_image_19.workflow_v2 import (
    run_abliterate,
    run_merge,
    run_post_merge_train,
    run_quantize,
    run_report,
)
from qwen_image_19.server import resolve_api


def _repo_tempdir() -> tempfile.TemporaryDirectory[str]:
    return tempfile.TemporaryDirectory(dir=repo_root())


class PipelineWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = _repo_tempdir()
        self.runs_root = Path(self.tmpdir.name)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_merge_writes_run_manifest_and_step_bundle(self) -> None:
        result = run_merge(run_id="run-merge", artifact_dir=str(self.runs_root))
        run_dir = self.runs_root / "run-merge"
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(result["run_id"], "run-merge")
        self.assertIn("merge", manifest["steps"])
        self.assertTrue((run_dir / "merge" / "step-result.json").exists())
        self.assertTrue((run_dir / "report-index.json").exists())

    def test_abliterate_uses_merge_output_checkpoint(self) -> None:
        run_merge(run_id="run-abliterate", artifact_dir=str(self.runs_root))
        result = run_abliterate(run_id="run-abliterate", artifact_dir=str(self.runs_root))
        step_result = result["step_result"]
        self.assertIsNotNone(step_result["input_checkpoint"])
        self.assertEqual(step_result["status"], "planned")
        self.assertIn("abliterated", step_result["output_checkpoint"])

    def test_post_merge_train_chains_from_merge(self) -> None:
        run_merge(run_id="run-train", artifact_dir=str(self.runs_root))
        result = run_post_merge_train(run_id="run-train", artifact_dir=str(self.runs_root))
        step_result = result["step_result"]
        self.assertEqual(step_result["step"], "post_merge_train")
        self.assertIsNotNone(step_result["input_checkpoint"])
        self.assertTrue((self.runs_root / "run-train" / "post_merge_train" / "step-result.json").exists())

    def test_quantize_chains_from_abliterate(self) -> None:
        run_merge(run_id="run-quant", artifact_dir=str(self.runs_root))
        run_abliterate(run_id="run-quant", artifact_dir=str(self.runs_root))
        result = run_quantize(run_id="run-quant", artifact_dir=str(self.runs_root))
        step_result = result["step_result"]
        self.assertEqual(step_result["step"], "quantize")
        self.assertIsNotNone(step_result["input_checkpoint"])
        manifest = json.loads(
            (self.runs_root / "run-quant" / "manifest.json").read_text(encoding="utf-8")
        )
        self.assertIn("quantize", manifest["steps"])

    def test_report_command_writes_dashboard_index(self) -> None:
        run_merge(run_id="run-report", artifact_dir=str(self.runs_root))
        result = run_report(artifact_dir=str(self.runs_root))
        self.assertEqual(result["run_count"], 1)
        self.assertTrue((self.runs_root / "dashboard-index.json").exists())

    def test_results_server_contract_resolves_run_endpoints(self) -> None:
        run_merge(run_id="run-server", artifact_dir=str(self.runs_root))
        runs_status, runs, runs_content_type = resolve_api("/api/runs", self.runs_root)
        manifest_status, manifest, manifest_content_type = resolve_api("/api/runs/run-server", self.runs_root)
        step_status, step_result, step_content_type = resolve_api("/api/runs/run-server/steps/merge", self.runs_root)
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
