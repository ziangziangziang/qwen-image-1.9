from __future__ import annotations

import unittest

from qwen_image_19.stage_4_quant import load_quant_profiles, validate_quant_profiles


class Stage4Tests(unittest.TestCase):
    def test_gguf_profile_requires_imatrix(self) -> None:
        profiles = load_quant_profiles()
        validate_quant_profiles(profiles)
        self.assertTrue(profiles["gguf"]["imatrix"]["required"])


if __name__ == "__main__":
    unittest.main()

