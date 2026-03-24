#!/usr/bin/env python3
"""Stage 1 developer probe entrypoints.

Each probe calls an existing public analysis function and prints the result as
JSON to stdout. Invoke via::

    python -m qwen_image_19.stage_1_analysis._probes --probe vae --hf-home /path/to/hf
    python -m qwen_image_19.stage_1_analysis._probes --probe rope --hf-home /path/to/hf
    python -m qwen_image_19.stage_1_analysis._probes --probe compare --hf-home /path/to/hf
    python -m qwen_image_19.stage_1_analysis._probes --probe render-viz --hf-home /path/to/hf
"""
from __future__ import annotations

import argparse
import json

from qwen_image_19.stage_1_analysis import (
    analyze_vae_compatibility,
    build_compatibility_matrix,
    compare_state_dicts,
    inspect_cache_models,
    load_cache_alias_map,
    load_model_inventory,
    probe_rope_compatibility,
    render_similarity_visualization,
)


def _build_manifests(hf_home: str, cache_map_config: str | None):
    return inspect_cache_models(
        hf_home,
        load_model_inventory(),
        load_cache_alias_map(cache_map_config),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 1 developer probe — prints analysis result as JSON."
    )
    parser.add_argument(
        "--probe",
        choices=["vae", "rope", "compare", "render-viz"],
        required=True,
        help=(
            "vae: VAE compatibility report; "
            "rope: RoPE/text-encoder compatibility report; "
            "compare: state-dict comparison; "
            "render-viz: similarity visualization."
        ),
    )
    parser.add_argument("--hf-home", required=True, help="Path to HuggingFace cache root.")
    parser.add_argument("--cache-map-config", default=None, help="Optional cache alias map YAML.")
    args = parser.parse_args()

    manifests = _build_manifests(args.hf_home, args.cache_map_config)

    if args.probe == "vae":
        print(json.dumps(analyze_vae_compatibility(manifests), indent=2))
    elif args.probe == "rope":
        print(json.dumps(probe_rope_compatibility(manifests), indent=2))
    elif args.probe == "compare":
        print(json.dumps(compare_state_dicts(manifests), indent=2))
    elif args.probe == "render-viz":
        matrix = build_compatibility_matrix(manifests)
        print(json.dumps(render_similarity_visualization(matrix), indent=2))
