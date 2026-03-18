#!/usr/bin/env python3
from __future__ import annotations

import json

from qwen_image_19.stage_3_eval import build_eval_registry


if __name__ == "__main__":
    print(json.dumps({"safety_suites": build_eval_registry()["safety_suites"]}, indent=2))

