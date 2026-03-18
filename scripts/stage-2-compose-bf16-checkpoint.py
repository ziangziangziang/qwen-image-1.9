#!/usr/bin/env python3
from __future__ import annotations

import json

from qwen_image_19.stage_2_fusion import fuse


if __name__ == "__main__":
    print(json.dumps(fuse(dry_run=True), indent=2))

