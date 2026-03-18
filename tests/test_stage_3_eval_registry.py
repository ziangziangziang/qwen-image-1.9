from __future__ import annotations

import unittest

from qwen_image_19.stage_3_eval import build_eval_registry


class Stage3Tests(unittest.TestCase):
    def test_registry_contains_capability_and_safety(self) -> None:
        registry = build_eval_registry()
        self.assertTrue(registry["capability_suites"])
        self.assertTrue(registry["safety_suites"])
        self.assertIn("layered decomposition reliability", registry["capability_suites"])


if __name__ == "__main__":
    unittest.main()

