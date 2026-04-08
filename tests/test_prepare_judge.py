"""Tests for the `prepare-judge` command and judge comparison report."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from qwen_image_19.config_io import repo_root
from qwen_image_19.workflow_v2 import run_prepare_judge, _run_judge_comparison


def _repo_tempdir() -> tempfile.TemporaryDirectory[str]:
    return tempfile.TemporaryDirectory(dir=repo_root())


class PreJudgeDryRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = _repo_tempdir()
        self.out_path = Path(self.tmpdir.name) / "judge-out"

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_dry_run_returns_expected_keys(self) -> None:
        result = run_prepare_judge(
            output_path=str(self.out_path),
            dry_run=True,
        )
        self.assertIn("judge_model", result)
        self.assertIn("output_path", result)
        self.assertIn("plan", result)
        self.assertTrue(result["dry_run"])
        self.assertEqual(result["status"], "planned")

    def test_dry_run_writes_no_files(self) -> None:
        run_prepare_judge(
            output_path=str(self.out_path),
            dry_run=True,
        )
        # The output directory should not have been created
        self.assertFalse(self.out_path.exists())

    def test_no_execute_no_execute_flag_returns_planned(self) -> None:
        result = run_prepare_judge(
            output_path=str(self.out_path),
            dry_run=False,
            execute=False,
        )
        self.assertEqual(result["status"], "planned")

    def test_judge_model_default_is_constant(self) -> None:
        from qwen_image_19.quality_judge import JUDGE_MODEL_ID
        result = run_prepare_judge(
            output_path=str(self.out_path),
            dry_run=True,
        )
        self.assertEqual(result["judge_model"], JUDGE_MODEL_ID)

    def test_judge_model_override(self) -> None:
        result = run_prepare_judge(
            judge_model="my-org/custom-judge",
            output_path=str(self.out_path),
            dry_run=True,
        )
        self.assertEqual(result["judge_model"], "my-org/custom-judge")


class PreJudgeExecuteValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = _repo_tempdir()
        self.out_path = Path(self.tmpdir.name) / "judge-out"

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_execute_without_recipe_raises_when_default_missing(self) -> None:
        """--execute without any recipe config raises ValueError."""
        # Patch out the default recipe so it appears absent
        with patch("qwen_image_19.workflow_v2.repo_root") as mock_root:
            mock_root.return_value = Path(self.tmpdir.name)
            with self.assertRaises((ValueError, RuntimeError)):
                run_prepare_judge(
                    output_path=str(self.out_path),
                    execute=True,
                    dry_run=False,
                )

    def test_execute_uses_default_recipe_when_present(self) -> None:
        """When the bundled recipe exists, --execute uses it without --recipe-config."""
        result = run_prepare_judge(
            output_path=str(self.out_path),
            execute=False,  # don't actually run the worker
            dry_run=False,
        )
        # The bundled recipe should have been resolved
        bundled = repo_root() / "configs" / "abliterate" / "judge-abliteration.yaml"
        if bundled.exists():
            self.assertEqual(result["recipe_config"], str(bundled))


class PreJudgeCLISubparserTests(unittest.TestCase):
    def _build_parser(self):
        from qwen_image_19.cli import build_parser
        return build_parser()

    def test_prepare_judge_subcommand_exists(self) -> None:
        parser = self._build_parser()
        args = parser.parse_args(["prepare-judge"])
        self.assertEqual(args.command, "prepare-judge")

    def test_default_output_path(self) -> None:
        parser = self._build_parser()
        args = parser.parse_args(["prepare-judge"])
        self.assertEqual(args.output_path, "/scratch/qwen-judge-abliterated")

    def test_custom_output_path(self) -> None:
        parser = self._build_parser()
        args = parser.parse_args(["prepare-judge", "--output-path", "/tmp/my-judge"])
        self.assertEqual(args.output_path, "/tmp/my-judge")

    def test_judge_model_flag(self) -> None:
        parser = self._build_parser()
        args = parser.parse_args(["prepare-judge", "--judge-model", "org/model"])
        self.assertEqual(args.judge_model, "org/model")

    def test_skip_comparison_flag(self) -> None:
        parser = self._build_parser()
        args = parser.parse_args(["prepare-judge", "--skip-comparison"])
        self.assertTrue(args.skip_comparison)

    def test_skip_comparison_default_false(self) -> None:
        parser = self._build_parser()
        args = parser.parse_args(["prepare-judge"])
        self.assertFalse(args.skip_comparison)

    def test_measure_pairs_default(self) -> None:
        parser = self._build_parser()
        args = parser.parse_args(["prepare-judge"])
        self.assertEqual(args.measure_pairs, 64)

    def test_measure_pairs_flag(self) -> None:
        parser = self._build_parser()
        args = parser.parse_args(["prepare-judge", "--measure-pairs", "128"])
        self.assertEqual(args.measure_pairs, 128)

    def test_recipe_config_flag(self) -> None:
        parser = self._build_parser()
        args = parser.parse_args([
            "prepare-judge",
            "--recipe-config", "configs/abliterate/judge-abliteration.yaml",
        ])
        self.assertEqual(args.recipe_config, "configs/abliterate/judge-abliteration.yaml")

    def test_measurements_flag(self) -> None:
        parser = self._build_parser()
        args = parser.parse_args(["prepare-judge", "--measurements", "/tmp/measures.pt"])
        self.assertEqual(args.measurements, "/tmp/measures.pt")

    def test_dry_run_flag(self) -> None:
        parser = self._build_parser()
        args = parser.parse_args(["prepare-judge", "--dry-run"])
        self.assertTrue(args.dry_run)

    def test_execute_flag(self) -> None:
        parser = self._build_parser()
        args = parser.parse_args(["prepare-judge", "--execute"])
        self.assertTrue(args.execute)

    def test_dispatch_calls_run_prepare_judge(self) -> None:
        from qwen_image_19 import cli
        parser = self._build_parser()
        args = parser.parse_args(["prepare-judge", "--dry-run", "--output-path", "/tmp/j"])
        with patch.object(cli, "run_prepare_judge", return_value={"status": "ok"}) as mock_fn:
            result = cli.dispatch(args)
        mock_fn.assert_called_once()
        call_kwargs = mock_fn.call_args.kwargs
        self.assertEqual(call_kwargs["output_path"], "/tmp/j")
        self.assertTrue(call_kwargs["dry_run"])
        self.assertIn("measure_pairs", call_kwargs)
        self.assertEqual(result, {"status": "ok"})


class JudgeComparisonReportTests(unittest.TestCase):
    """Unit tests for _run_judge_comparison — both judges are mocked."""

    def _make_score(self, overall: float, refused: bool = False) -> dict:
        if refused:
            raise ValueError("I cannot score this content.")
        return {
            "prompt_adherence": overall,
            "visual_quality": overall,
            "aesthetic_score": overall,
            "overall": overall,
            "reasoning": "test",
            "passed": overall >= 6.5,
            "scored_at": "2026-04-08T00:00:00Z",
            "task": "generation",
        }

    def test_comparison_writes_json_and_md(self) -> None:
        with _repo_tempdir() as tmp:
            out_dir = Path(tmp) / "comparison"

            call_count = [0]

            def judge_factory(model_id="", **kwargs):
                call_count[0] += 1
                m = MagicMock()
                score_val = 7.0 if call_count[0] == 1 else 8.5
                m.score_generation.return_value = {
                    "overall": score_val,
                    "passed": score_val >= 6.5,
                    "refused": False,
                    "scored_at": "2026-04-08T00:00:00Z",
                }
                m.unload = MagicMock()
                return m

            # Prevent any network calls — load_dataset raises so function falls
            # through to hardcoded examples (PIL required for the fallback)
            def _no_network(*args, **kwargs):
                raise RuntimeError("no network in tests")

            with patch("qwen_image_19.quality_judge.QualityJudge", side_effect=judge_factory), \
                 patch("datasets.load_dataset", side_effect=_no_network):
                result = _run_judge_comparison(
                    original_model_id="Qwen/Qwen3.5-35B-A3B",
                    abliterated_path="/tmp/fake-abliterated",
                    output_dir=out_dir,
                )

            self.assertTrue((out_dir / "comparison.json").exists())
            self.assertTrue((out_dir / "comparison.md").exists())

            records = json.loads((out_dir / "comparison.json").read_text())
            self.assertIsInstance(records, list)
            for rec in records:
                self.assertIn("prompt", rec)
                self.assertIn("input_type", rec)
                self.assertIn("before", rec)
                self.assertIn("after", rec)
                self.assertIn("delta_overall", rec)
                self.assertIn("before_refused", rec)

            md = (out_dir / "comparison.md").read_text()
            self.assertIn("## Image Input Examples", md)
            self.assertIn("## Text Input Examples", md)
            self.assertIn("Before overall", md)
            self.assertIn("After overall", md)

    def test_comparison_result_structure(self) -> None:
        """Lighter test: verify _run_judge_comparison returns required keys."""
        with _repo_tempdir() as tmp:
            out_dir = Path(tmp) / "cmp"

            call_count = [0]

            def judge_factory(model_id="", **kwargs):
                call_count[0] += 1
                m = MagicMock()
                m.score_generation.return_value = {
                    "overall": 7.0 if call_count[0] == 1 else 8.5,
                    "passed": True,
                    "refused": False,
                    "scored_at": "2026-04-08T00:00:00Z",
                }
                m.unload = MagicMock()
                return m

            # Prevent any network calls — load_dataset raises so function falls
            # through to hardcoded examples
            def _no_network(*args, **kwargs):
                raise RuntimeError("no network in tests")

            with patch("qwen_image_19.quality_judge.QualityJudge", side_effect=judge_factory), \
                 patch("datasets.load_dataset", side_effect=_no_network):
                result = _run_judge_comparison(
                    original_model_id="orig-model",
                    abliterated_path="/tmp/abl",
                    output_dir=out_dir,
                )

            self.assertIn("records", result)
            self.assertIn("refused_count", result)
            self.assertIn("mean_delta", result)
            self.assertIn("comparison_json", result)
            self.assertIn("comparison_md", result)

            md = (out_dir / "comparison.md").read_text()
            self.assertIn("## Image Input Examples", md)
            self.assertIn("## Text Input Examples", md)


class ResolveJudgeModelIdTests(unittest.TestCase):
    def test_returns_default_when_checkpoint_absent(self) -> None:
        from qwen_image_19.stage_3_eval.eval_worker import _resolve_judge_model_id
        from qwen_image_19.quality_judge import JUDGE_MODEL_ID
        with patch(
            "qwen_image_19.stage_3_eval.eval_worker._load_eval_config",
            return_value={"judge": {"model_id": JUDGE_MODEL_ID, "checkpoint": "/nonexistent/path"}},
        ):
            result = _resolve_judge_model_id()
        self.assertEqual(result, JUDGE_MODEL_ID)

    def test_returns_checkpoint_when_path_exists(self) -> None:
        from qwen_image_19.stage_3_eval.eval_worker import _resolve_judge_model_id
        with _repo_tempdir() as tmp:
            ckpt_path = tmp
            with patch(
                "qwen_image_19.stage_3_eval.eval_worker._load_eval_config",
                return_value={"judge": {"model_id": "orig-model", "checkpoint": ckpt_path}},
            ):
                result = _resolve_judge_model_id()
        self.assertEqual(result, ckpt_path)

    def test_returns_model_id_when_no_checkpoint_key(self) -> None:
        from qwen_image_19.stage_3_eval.eval_worker import _resolve_judge_model_id
        with patch(
            "qwen_image_19.stage_3_eval.eval_worker._load_eval_config",
            return_value={"judge": {"model_id": "org/model"}},
        ):
            result = _resolve_judge_model_id()
        self.assertEqual(result, "org/model")


if __name__ == "__main__":
    unittest.main()
