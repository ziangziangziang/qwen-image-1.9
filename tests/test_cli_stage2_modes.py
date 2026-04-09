from __future__ import annotations

import unittest
from unittest.mock import patch

from qwen_image_19.cli import build_parser, dispatch


class PipelineCliTests(unittest.TestCase):
    def test_parser_accepts_merge_dry_run(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["merge", "--dry-run"])
        self.assertTrue(args.dry_run)
        self.assertFalse(args.execute)

    def test_parser_accepts_merge_method(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["merge", "--method", "ties"])
        self.assertEqual(args.method, "ties")

    def test_dispatch_forwards_merge_flags(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["merge", "--execute", "--tag", "nightly"])
        with patch("qwen_image_19.cli.run_merge", return_value={"ok": True}) as mocked_merge:
            dispatch(args)
        kwargs = mocked_merge.call_args.kwargs
        self.assertEqual(kwargs["execute"], True)
        self.assertEqual(kwargs["tags"], ["nightly"])
        self.assertEqual(kwargs["merge_method"], "slerp")

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
