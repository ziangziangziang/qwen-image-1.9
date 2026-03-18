#!/usr/bin/env bash
set -euo pipefail
PYTHONPATH=src python3 -m qwen_image_19.cli stage4 quantize "$@"

