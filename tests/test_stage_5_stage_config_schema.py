from __future__ import annotations

import unittest

from qwen_image_19.remote import default_remote_context
from qwen_image_19.stage_5_deploy import generate_stage_config, load_stage_template, validate_stage_config


class Stage5Tests(unittest.TestCase):
    def test_generated_stage_config_is_valid(self) -> None:
        config = generate_stage_config(load_stage_template(), default_remote_context())
        self.assertEqual(validate_stage_config(config), [])
        self.assertEqual(config["runtime"]["backend"], "vllm-omni")


if __name__ == "__main__":
    unittest.main()

