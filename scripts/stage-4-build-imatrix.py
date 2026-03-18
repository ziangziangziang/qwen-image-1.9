#!/usr/bin/env python3
from __future__ import annotations

import json

from qwen_image_19.stage_4_quant import load_quant_profiles


if __name__ == "__main__":
    print(json.dumps(load_quant_profiles()["gguf"]["imatrix"], indent=2))

