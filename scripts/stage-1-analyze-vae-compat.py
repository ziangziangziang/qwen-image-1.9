#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from qwen_image_19.stage_1_analysis import analyze_vae_compatibility, inspect_cache_models, load_cache_alias_map, load_model_inventory


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-home", required=True)
    parser.add_argument("--cache-map-config")
    args = parser.parse_args()
    manifests = inspect_cache_models(args.hf_home, load_model_inventory(), load_cache_alias_map(args.cache_map_config))
    print(json.dumps(analyze_vae_compatibility(manifests), indent=2))
