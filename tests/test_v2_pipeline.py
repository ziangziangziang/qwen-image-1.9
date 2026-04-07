"""Tests for the new v2 pipeline workflow."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from qwen_image_19.config_io import repo_root
from qwen_image_19.contracts import PIPELINE_STEPS
from qwen_image_19.workflow_v2 import (
    run_merge,
    run_post_merge_train,
    run_abliterate,
    run_post_abliterate_train,
    run_quantize,
    run_post_quantize_eval,
    run_report,
)


def _tmpdir() -> tempfile.TemporaryDirectory[str]:
    return tempfile.TemporaryDirectory(dir=repo_root())


class TestPipelineSteps(unittest.TestCase):
    """New 6-step pipeline: merge → train → abliterate → train → quantize → eval."""

    def setUp(self) -> None:
        self.tmpdir = _tmpdir()
        self.root = Path(self.tmpdir.name)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    # ── PIPELINE_STEPS definition ─────────────────────────────────

    def test_pipeline_has_six_steps(self) -> None:
        self.assertEqual(len(PIPELINE_STEPS), 6)
        self.assertEqual(
            PIPELINE_STEPS,
            ("merge", "post_merge_train", "abliterate",
             "post_abliterate_train", "quantize", "post_quantize_eval"),
        )

    # ── Merge ─────────────────────────────────────────────────────

    def test_merge_dry_run(self) -> None:
        result = run_merge(run_id="test-merge", artifact_dir=str(self.root), dry_run=True)
        self.assertEqual(result["run_id"], "test-merge")
        self.assertIn("step_result", result)
        self.assertEqual(result["step_result"]["step"], "merge")
        self.assertEqual(result["step_result"]["status"], "planned")

    def test_merge_writes_manifest(self) -> None:
        result = run_merge(run_id="test-merge-w", artifact_dir=str(self.root))
        run_dir = self.root / "test-merge-w"
        self.assertTrue((run_dir / "manifest.json").exists())
        manifest = json.loads((run_dir / "manifest.json").read_text("utf-8"))
        self.assertEqual(manifest["run_id"], "test-merge-w")
        self.assertIn("merge", manifest["steps"])
        # All 6 steps should be initialized
        for step in PIPELINE_STEPS:
            self.assertIn(step, manifest["steps"])

    # ── Post-merge train ──────────────────────────────────────────

    def test_post_merge_train_dry_run(self) -> None:
        # First run merge to create the manifest
        run_merge(run_id="test-train", artifact_dir=str(self.root))
        result = run_post_merge_train(
            run_id="test-train", artifact_dir=str(self.root), dry_run=True,
        )
        self.assertEqual(result["step_result"]["step"], "post_merge_train")
        self.assertEqual(result["step_result"]["status"], "planned")
        # Input should come from merge output
        self.assertIsNotNone(result["step_result"]["input_checkpoint"])

    # ── Abliterate ────────────────────────────────────────────────

    def test_abliterate_dry_run(self) -> None:
        run_merge(run_id="test-abl", artifact_dir=str(self.root))
        result = run_abliterate(
            run_id="test-abl", artifact_dir=str(self.root), dry_run=True,
        )
        self.assertEqual(result["step_result"]["step"], "abliterate")
        self.assertEqual(result["step_result"]["status"], "planned")

    def test_abliterate_requires_recipe_for_execute(self) -> None:
        run_merge(run_id="test-abl-exec", artifact_dir=str(self.root))
        with self.assertRaises(ValueError):
            run_abliterate(
                run_id="test-abl-exec", artifact_dir=str(self.root),
                execute=True,
            )

    # ── Post-abliterate train ─────────────────────────────────────

    def test_post_abliterate_train_dry_run(self) -> None:
        run_merge(run_id="test-pat", artifact_dir=str(self.root))
        run_abliterate(run_id="test-pat", artifact_dir=str(self.root), dry_run=True)
        # Write the abliterate step so the manifest has the output checkpoint
        run_abliterate(run_id="test-pat", artifact_dir=str(self.root))
        result = run_post_abliterate_train(
            run_id="test-pat", artifact_dir=str(self.root), dry_run=True,
        )
        self.assertEqual(result["step_result"]["step"], "post_abliterate_train")

    # ── Quantize ──────────────────────────────────────────────────

    def test_quantize_dry_run(self) -> None:
        run_merge(run_id="test-quant", artifact_dir=str(self.root))
        result = run_quantize(
            run_id="test-quant", artifact_dir=str(self.root), dry_run=True,
        )
        self.assertEqual(result["step_result"]["step"], "quantize")
        self.assertIn("quant_method", result["step_result"]["metrics"])

    # ── Post-quantize eval ────────────────────────────────────────

    def test_post_quantize_eval_dry_run(self) -> None:
        run_merge(run_id="test-eval", artifact_dir=str(self.root))
        run_quantize(run_id="test-eval", artifact_dir=str(self.root))
        result = run_post_quantize_eval(
            run_id="test-eval", artifact_dir=str(self.root), dry_run=True,
        )
        self.assertEqual(result["step_result"]["step"], "post_quantize_eval")

    # ── Full pipeline dry-run ─────────────────────────────────────

    def test_full_pipeline_dry_run(self) -> None:
        rid = "test-full"
        run_merge(run_id=rid, artifact_dir=str(self.root))
        run_post_merge_train(run_id=rid, artifact_dir=str(self.root))
        run_abliterate(run_id=rid, artifact_dir=str(self.root))
        run_post_abliterate_train(run_id=rid, artifact_dir=str(self.root))
        run_quantize(run_id=rid, artifact_dir=str(self.root))
        run_post_quantize_eval(run_id=rid, artifact_dir=str(self.root))

        manifest = json.loads(
            (self.root / rid / "manifest.json").read_text("utf-8")
        )
        for step in PIPELINE_STEPS:
            self.assertIn(step, manifest["steps"])
            self.assertIsNotNone(manifest["steps"][step]["step_result"],
                                  f"{step} should have step_result")

    # ── Report ────────────────────────────────────────────────────

    def test_report_generates_index(self) -> None:
        run_merge(run_id="test-report", artifact_dir=str(self.root))
        result = run_report(artifact_dir=str(self.root))
        self.assertIn("run_count", result)
        self.assertEqual(result["run_count"], 1)


class TestCLIParsing(unittest.TestCase):
    """Test that CLI parses new commands correctly."""

    def test_all_commands_parse(self) -> None:
        from qwen_image_19.cli import build_parser
        parser = build_parser()

        # merge
        args = parser.parse_args(["merge", "--dry-run"])
        self.assertEqual(args.command, "merge")
        self.assertTrue(args.dry_run)

        # post-merge-train
        args = parser.parse_args(["post-merge-train", "--run-id", "r1", "--dry-run"])
        self.assertEqual(args.command, "post-merge-train")
        self.assertEqual(args.run_id, "r1")

        # abliterate
        args = parser.parse_args(["abliterate", "--run-id", "r1", "--dry-run"])
        self.assertEqual(args.command, "abliterate")

        # post-abliterate-train
        args = parser.parse_args(["post-abliterate-train", "--run-id", "r1", "--dry-run"])
        self.assertEqual(args.command, "post-abliterate-train")

        # quantize
        args = parser.parse_args(["quantize", "--run-id", "r1", "--dry-run", "--method", "exl2", "--bits", "3"])
        self.assertEqual(args.command, "quantize")
        self.assertEqual(args.method, "exl2")
        self.assertEqual(args.bits, 3)

        # eval
        args = parser.parse_args(["eval", "--run-id", "r1", "--prompts", "4"])
        self.assertEqual(args.command, "eval")
        self.assertEqual(args.prompts, 4)

        # report
        args = parser.parse_args(["report", "--serve", "--port", "9000"])
        self.assertEqual(args.command, "report")
        self.assertTrue(args.serve)
        self.assertEqual(args.port, 9000)


class TestContracts(unittest.TestCase):
    """Test contract helpers."""

    def test_public_path_relative(self) -> None:
        from qwen_image_19.contracts import public_path
        p = public_path(repo_root() / "reports" / "runs" / "test")
        self.assertEqual(p, "reports/runs/test")
        self.assertFalse(p.startswith("/"))

    def test_public_path_remote_passthrough(self) -> None:
        from qwen_image_19.contracts import public_path
        self.assertEqual(public_path("hf://Qwen/test"), "hf://Qwen/test")
        self.assertEqual(public_path("s3://bucket/key"), "s3://bucket/key")

    def test_ensure_run_manifest_backfills_steps(self) -> None:
        from qwen_image_19.contracts import ensure_run_manifest
        with _tmpdir() as td:
            root = Path(td)
            m, _ = ensure_run_manifest(
                run_id="t", runs_root=root, source_models={},
            )
            for step in PIPELINE_STEPS:
                self.assertIn(step, m["steps"])
                self.assertEqual(m["steps"][step]["status"], "pending")


class TestServer(unittest.TestCase):
    """Test the server API routing logic."""

    def test_api_runs_returns_list(self) -> None:
        from qwen_image_19.server import resolve_api
        with _tmpdir() as td:
            root = Path(td)
            result = resolve_api("/api/runs", root)
            self.assertIsNotNone(result)
            status, body, ct = result  # type: ignore
            self.assertEqual(status, 200)
            self.assertEqual(ct, "application/json")
            self.assertIsInstance(body, list)

    def test_api_run_not_found(self) -> None:
        from qwen_image_19.server import resolve_api
        with _tmpdir() as td:
            root = Path(td)
            result = resolve_api("/api/runs/nonexistent", root)
            self.assertIsNotNone(result)
            status, _, _ = result  # type: ignore
            self.assertEqual(status, 404)

    def test_dashboard_html(self) -> None:
        from qwen_image_19.server import _dashboard_html
        html = _dashboard_html()
        self.assertIn("Qwen-Image 1.9 Dashboard", html)
        self.assertIn("post_merge_train", html)


if __name__ == "__main__":
    unittest.main()
