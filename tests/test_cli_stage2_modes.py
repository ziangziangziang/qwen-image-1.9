from __future__ import annotations

import unittest
from unittest.mock import patch

from qwen_image_19.cli import build_parser, dispatch


class Stage2CliModeTests(unittest.TestCase):
    def test_parser_accepts_smoke_run_without_execute(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["stage2", "fuse", "--smoke-run"])
        self.assertTrue(args.smoke_run)
        self.assertFalse(args.execute)

    def test_parser_accepts_smoke_profile_and_execute(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "stage2",
                "fuse",
                "--smoke-run",
                "--run-profile",
                "smoke",
                "--execute",
                "--resume",
            ]
        )
        self.assertTrue(args.smoke_run)
        self.assertEqual(args.run_profile, "smoke")
        self.assertTrue(args.execute)
        self.assertTrue(args.resume)

    def test_dispatch_forwards_stage2_flags(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["stage2", "fuse", "--smoke-run", "--execute"])
        with patch("qwen_image_19.cli.fuse", return_value={"ok": True}) as mocked_fuse:
            dispatch(args)
        kwargs = mocked_fuse.call_args.kwargs
        self.assertEqual(kwargs["smoke_run"], True)
        self.assertEqual(kwargs["execute"], True)
        self.assertEqual(kwargs["resume"], False)
        self.assertEqual(kwargs["run_profile"], None)
        self.assertEqual(kwargs["dry_run"], False)


if __name__ == "__main__":
    unittest.main()
