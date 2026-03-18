#!/usr/bin/env python3
from __future__ import annotations

import json

from qwen_image_19.stage_5_deploy import deploy


if __name__ == "__main__":
    print(json.dumps(deploy(dry_run=True)["config"], indent=2))

