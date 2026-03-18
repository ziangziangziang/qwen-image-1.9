from __future__ import annotations

import unittest

from qwen_image_19.stage_2_fusion import fuse


class Stage2Tests(unittest.TestCase):
    def test_fuse_dry_run_records_merge_inputs(self) -> None:
        manifest = fuse(dry_run=True)["manifest"]
        self.assertEqual(manifest["recipes"]["edit_delta"]["blend_weight"], 0.35)
        self.assertIn("output_checkpoint", manifest["remote"])


if __name__ == "__main__":
    unittest.main()

