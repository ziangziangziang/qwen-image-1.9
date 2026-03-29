from __future__ import annotations

import argparse
import json
from typing import Any

from qwen_image_19.logging_utils import console
from qwen_image_19.workflow import run_abliterate_step, run_merge, run_preflight, run_quantize_step, run_report


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--remote-config", help="Path to remote launcher or env config.")
    parser.add_argument("--artifact-dir", help="Base artifact directory. Defaults to reports/runs for the new pipeline.")
    parser.add_argument("--cache-dir", help="Optional cache dir override for remote-first dry runs.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve configs and print outputs without writing.")
    parser.add_argument("--smoke-run", action="store_true", help="Run a minimal quick pass to prove the pipeline wiring.")
    parser.add_argument("--execute", action="store_true", help="Execute the full workload. Can be resource-intensive.")
    parser.add_argument("--resume", action="store_true", help="Resume from prior outputs instead of overwriting.")


def _add_run_args(parser: argparse.ArgumentParser, *, require_run_id: bool) -> None:
    parser.add_argument("--run-id", required=require_run_id, help="Stable run identifier under reports/runs/<run_id>.")
    parser.add_argument("--tag", dest="tags", action="append", default=[], help="Tag to store in the run manifest.")
    parser.add_argument("--notes", help="Free-form notes stored in the run manifest.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="q19", description="Qwen-Image 1.9 3-step pipeline CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight", help="Build preflight evidence from source checkpoints.")
    add_common_args(preflight)
    preflight.add_argument("--hf-home", help="Path to HF_HOME or its hub directory on the remote machine.")
    preflight.add_argument("--cache-map-config", help="Optional JSON/YAML mapping from model aliases to HF cache directory names.")
    preflight.add_argument("--json", dest="json_output", action="store_true", help="Print full machine-readable preflight payload.")

    merge = subparsers.add_parser("merge", help="Run the merge step and emit run-scoped artifacts.")
    add_common_args(merge)
    _add_run_args(merge, require_run_id=False)
    merge.add_argument(
        "--run-profile",
        choices=("smoke", "full", "quality"),
        help="Merge execution profile. Defaults to smoke when --smoke-run is set, otherwise full.",
    )

    abliterate = subparsers.add_parser("abliterate", help="Run the refusal-direction removal step.")
    add_common_args(abliterate)
    _add_run_args(abliterate, require_run_id=True)
    abliterate.add_argument("--input-checkpoint", help="Explicit input checkpoint. Defaults to the run's merge output.")

    quant = subparsers.add_parser("quantize", help="Run the quantization step and emit run-scoped artifacts.")
    add_common_args(quant)
    _add_run_args(quant, require_run_id=True)
    quant.add_argument("--input-checkpoint", help="Explicit input checkpoint. Defaults to the run's abliterate output.")

    report = subparsers.add_parser("report", help="Generate the shared results index and optional internal API server.")
    report.add_argument("--artifact-dir", help="Runs root. Defaults to reports/runs.")
    report.add_argument("--run-id", help="Optional run id to validate while building report indexes.")
    report.add_argument("--serve", action="store_true", help="Start the lightweight internal results server.")
    report.add_argument("--host", default="127.0.0.1", help="Server bind host for --serve.")
    report.add_argument("--port", type=int, default=8000, help="Server bind port for --serve.")

    return parser


def dispatch(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "preflight":
        return run_preflight(
            artifact_dir=args.artifact_dir,
            remote_config=args.remote_config,
            cache_dir=args.cache_dir,
            dry_run=args.dry_run,
            smoke_run=args.smoke_run,
            execute=args.execute,
            hf_home=args.hf_home,
            cache_map_config=args.cache_map_config,
        )
    if args.command == "merge":
        return run_merge(
            run_id=args.run_id,
            artifact_dir=args.artifact_dir,
            remote_config=args.remote_config,
            cache_dir=args.cache_dir,
            run_profile=args.run_profile,
            dry_run=args.dry_run,
            smoke_run=args.smoke_run,
            execute=args.execute,
            resume=args.resume,
            tags=args.tags,
            notes=args.notes,
        )
    if args.command == "abliterate":
        return run_abliterate_step(
            run_id=args.run_id,
            artifact_dir=args.artifact_dir,
            remote_config=args.remote_config,
            input_checkpoint=args.input_checkpoint,
            dry_run=args.dry_run,
            execute=args.execute,
        )
    if args.command == "quantize":
        return run_quantize_step(
            run_id=args.run_id,
            artifact_dir=args.artifact_dir,
            remote_config=args.remote_config,
            cache_dir=args.cache_dir,
            input_checkpoint=args.input_checkpoint,
            dry_run=args.dry_run,
            smoke_run=args.smoke_run,
            execute=args.execute,
            resume=args.resume,
        )
    if args.command == "report":
        return run_report(
            artifact_dir=args.artifact_dir,
            run_id=args.run_id,
            serve=args.serve,
            host=args.host,
            port=args.port,
        )
    raise ValueError(f"Unsupported command `{args.command}`.")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = dispatch(args)
    except (RuntimeError, ValueError) as exc:
        console.print_json(data=json.dumps({"command": getattr(args, "command", None), "error": str(exc)}, indent=2))
        return 1
    if args.command == "preflight" and not getattr(args, "json_output", False):
        if "terminal_summary" in result:
            print(result["terminal_summary"])
            return 0
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
