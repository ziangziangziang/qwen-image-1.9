#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from qwen_image_19.stage_1_analysis import build_compatibility_matrix, inspect_cache_models, load_cache_alias_map, load_model_inventory, render_similarity_visualization


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-home", required=True)
    parser.add_argument("--cache-map-config")
    args = parser.parse_args()
    manifests = inspect_cache_models(args.hf_home, load_model_inventory(), load_cache_alias_map(args.cache_map_config))
    matrix = build_compatibility_matrix(manifests)
    print(json.dumps(render_similarity_visualization(matrix), indent=2))
