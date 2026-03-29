from __future__ import annotations

from pathlib import Path
import unittest

from qwen_image_19.stage_3_eval import build_eval_registry, build_eval_summary, evaluate


class Stage3Tests(unittest.TestCase):
    def test_registry_contains_pipeline_steps(self) -> None:
        registry = build_eval_registry()
        self.assertIn("merge", registry)
        self.assertIn("abliterate", registry)
        self.assertIn("quantize", registry)
        self.assertIn("merge-generation-fidelity", registry["merge"])

    def test_build_eval_summary_uses_shared_schema(self) -> None:
        summary = build_eval_summary(
            run_id="run-001",
            step="quantize",
            checkpoint_ref="s3://bucket/quantized",
            sample_root=Path("reports/runs/run-001/quantize/samples"),
        )
        self.assertEqual(summary["step"], "quantize")
        self.assertTrue(summary["suites"])
        self.assertEqual(summary["suites"][0]["checkpoint_ref"], "s3://bucket/quantized")

    def test_legacy_evaluate_still_supports_dry_run(self) -> None:
        result = evaluate(dry_run=True)
        self.assertEqual(result["stage"], "stage3")
        self.assertIn("registry", result)


if __name__ == "__main__":
    unittest.main()
