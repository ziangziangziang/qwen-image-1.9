#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--manifest", required=True, help="Path to reports/stage-2/dataset-manifest.json")
    return parser.parse_args()


def write_placeholder(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    args = parse_args()
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise SystemExit(f"Dataset manifest not found: {args.manifest}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not args.execute:
        print(json.dumps(payload, indent=2))
        raise SystemExit(0)

    for record in payload.get("planned_records", []):
        for _, rel_path in record.get("asset_paths", {}).items():
            path = Path(rel_path)
            if path.suffix.lower() == ".json":
                write_placeholder(path, json.dumps({"sample_id": record["sample_id"]}, indent=2) + "\n")
            else:
                write_placeholder(path, f"placeholder:{record['sample_id']}\n")

    sentinel = Path(payload["output_root"]) / "dataset-ready.json"
    write_placeholder(
        sentinel,
        json.dumps(
            {
                "record_count": len(payload.get("planned_records", [])),
                "splits": list(payload.get("splits", {}).keys()),
            },
            indent=2,
        )
        + "\n",
    )
    print(json.dumps({"status": "ok", "output_root": payload["output_root"]}, indent=2))
