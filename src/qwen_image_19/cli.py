from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Callable

from qwen_image_19.stage_1_analysis import analyze
from qwen_image_19.stage_2_fusion import fuse
from qwen_image_19.stage_3_eval import evaluate
from qwen_image_19.stage_4_quant import quantize
from qwen_image_19.stage_5_deploy import deploy


StageHandler = Callable[..., dict[str, Any]]


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--remote-config", help="Path to remote launcher or env config.")
    parser.add_argument("--artifact-dir", help="Directory for generated reports and manifests.")
    parser.add_argument("--cache-dir", help="Optional cache dir override for remote-first dry runs.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve configs and print outputs without writing.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="q19", description="Qwen-Image 1.9 remote-first scaffold CLI.")
    subparsers = parser.add_subparsers(dest="stage", required=True)

    stage1 = subparsers.add_parser("stage1", help="Stage 1 structural fusion analysis.")
    stage1_sub = stage1.add_subparsers(dest="action", required=True)
    stage1_analyze = stage1_sub.add_parser("analyze", help="Build Stage 1 compatibility artifacts.")
    add_common_args(stage1_analyze)
    stage1_analyze.add_argument("--hf-home", help="Path to HF_HOME or its hub directory on the remote machine.")
    stage1_analyze.add_argument(
        "--cache-map-config",
        help="Optional JSON/YAML mapping from model aliases to HF cache directory names.",
    )
    stage1_analyze.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="Print the full machine-readable Stage 1 payload to stdout.",
    )

    stage2 = subparsers.add_parser("stage2", help="Stage 2 fusion planning.")
    stage2_sub = stage2.add_subparsers(dest="action", required=True)
    stage2_fuse = stage2_sub.add_parser("fuse", help="Build Stage 2 merge artifacts.")
    add_common_args(stage2_fuse)

    stage3 = subparsers.add_parser("stage3", help="Stage 3 evaluation.")
    stage3_sub = stage3.add_subparsers(dest="action", required=True)
    stage3_eval = stage3_sub.add_parser("eval", help="Build Stage 3 evaluation artifacts.")
    add_common_args(stage3_eval)

    stage4 = subparsers.add_parser("stage4", help="Stage 4 quantization planning.")
    stage4_sub = stage4.add_subparsers(dest="action", required=True)
    stage4_quantize = stage4_sub.add_parser("quantize", help="Build Stage 4 quantization artifacts.")
    add_common_args(stage4_quantize)

    stage5 = subparsers.add_parser("stage5", help="Stage 5 deployment planning.")
    stage5_sub = stage5.add_subparsers(dest="action", required=True)
    stage5_deploy = stage5_sub.add_parser("deploy", help="Build Stage 5 deployment artifacts.")
    add_common_args(stage5_deploy)

    return parser


def dispatch(args: argparse.Namespace) -> dict[str, Any]:
    handlers: dict[tuple[str, str], StageHandler] = {
        ("stage1", "analyze"): analyze,
        ("stage2", "fuse"): fuse,
        ("stage3", "eval"): evaluate,
        ("stage4", "quantize"): quantize,
        ("stage5", "deploy"): deploy,
    }
    handler = handlers[(args.stage, args.action)]
    kwargs = {
        "artifact_dir": args.artifact_dir,
        "remote_config": args.remote_config,
        "dry_run": args.dry_run,
    }
    if hasattr(args, "cache_dir"):
        kwargs["cache_dir"] = args.cache_dir
    if hasattr(args, "hf_home"):
        kwargs["hf_home"] = args.hf_home
    if hasattr(args, "cache_map_config"):
        kwargs["cache_map_config"] = args.cache_map_config
    return handler(**kwargs)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = dispatch(args)
    except (RuntimeError, ValueError) as exc:
        print(json.dumps({"stage": getattr(args, "stage", None), "error": str(exc)}, indent=2), file=sys.stderr)
        return 1
    if (
        getattr(args, "stage", None) == "stage1"
        and getattr(args, "action", None) == "analyze"
        and not getattr(args, "json_output", False)
    ):
        print(result["terminal_summary"])
        return 0
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
