#!/usr/bin/env python3
from __future__ import annotations

import json

from qwen_image_19.config_io import repo_root
from qwen_image_19.stage_1_analysis import load_model_inventory
from qwen_image_19.stage_2_fusion import build_text_encoder_merge_recipe


if __name__ == "__main__":
    print(json.dumps(build_text_encoder_merge_recipe(load_model_inventory(), repo_root() / "configs" / "merge"), indent=2))

