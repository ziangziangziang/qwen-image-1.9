#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from qwen_image_19.stage_2_fusion import fuse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--candidate-id", default="core-delta-w035")
    parser.add_argument("--output-checkpoint", help="Relative output checkpoint path.")
    return parser.parse_args()


def write_artifact(candidate_id: str, output_checkpoint: str) -> dict[str, object]:
    path = Path(output_checkpoint)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "candidate_id": candidate_id,
        "artifact_type": "bf16-core-checkpoint-placeholder",
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    args = parse_args()
    if not args.execute:
        print(json.dumps(fuse(dry_run=True)["manifest"]["core_delta_recipe"], indent=2))
        raise SystemExit(0)
    if not args.output_checkpoint:
        raise SystemExit("--output-checkpoint is required with --execute")
    artifact = write_artifact(args.candidate_id, args.output_checkpoint)
    print(json.dumps({"status": "ok", "output_checkpoint": args.output_checkpoint, "metadata": artifact}, indent=2))
