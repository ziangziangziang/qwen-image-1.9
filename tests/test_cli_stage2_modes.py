from __future__ import annotations

import unittest
from unittest.mock import patch

from qwen_image_19.cli import build_parser, dispatch


class PipelineCliTests(unittest.TestCase):
    def test_parser_accepts_merge_smoke_run_without_execute(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["merge", "--smoke-run"])
        self.assertTrue(args.smoke_run)
        self.assertFalse(args.execute)

    def test_parser_accepts_merge_quality_profile(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["merge", "--run-profile", "quality"])
        self.assertEqual(args.run_profile, "quality")
        self.assertFalse(args.smoke_run)

    def test_dispatch_forwards_merge_flags(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["merge", "--smoke-run", "--execute", "--tag", "nightly"])
        with patch("qwen_image_19.cli.run_merge", return_value={"ok": True}) as mocked_merge:
            dispatch(args)
        kwargs = mocked_merge.call_args.kwargs
        self.assertEqual(kwargs["smoke_run"], True)
        self.assertEqual(kwargs["execute"], True)
        self.assertEqual(kwargs["tags"], ["nightly"])
        self.assertEqual(kwargs["run_profile"], None)

    def test_parser_requires_run_id_for_abliterate(self) -> None:
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["abliterate"])

    def test_legacy_stage_commands_are_rejected(self) -> None:
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["stage2", "fuse", "--smoke-run", "--execute"])


if __name__ == "__main__":
    unittest.main()
