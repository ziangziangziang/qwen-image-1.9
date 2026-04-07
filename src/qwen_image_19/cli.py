"""Qwen-Image 1.9 CLI.

Pipeline: merge → post-merge-train → abliterate → post-abliterate-train
          → quantize → post-quantize-eval

All models sourced from HuggingFace.
"""
from __future__ import annotations

import argparse
import json
from typing import Any

from qwen_image_19.logging_utils import console
from qwen_image_19.workflow_v2 import (
    run_abliterate,
    run_merge,
    run_post_abliterate_train,
    run_post_merge_train,
    run_post_quantize_eval,
    run_quantize,
    run_report,
)


# ── Shared argument groups ──────────────────────────────────────────

def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--remote-config", help="Remote launcher config (YAML/.env).")
    p.add_argument("--artifact-dir", help="Runs root. Defaults to reports/runs.")
    p.add_argument("--dry-run", action="store_true", help="Print plan without writing files.")
    p.add_argument("--execute", action="store_true", help="Execute the full GPU workload.")
    p.add_argument("--resume", action="store_true", help="Resume from prior outputs.")


def _run_args(p: argparse.ArgumentParser, *, require_id: bool) -> None:
    p.add_argument("--run-id", required=require_id, help="Run identifier.")
    p.add_argument("--tag", dest="tags", action="append", default=[], help="Tag for the manifest.")
    p.add_argument("--notes", help="Free-form notes for the manifest.")


# ── Parser ──────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="q19",
        description="Qwen-Image 1.9 checkpoint pipeline (HuggingFace-based).",
    )
    sub = root.add_subparsers(dest="command", required=True)

    # merge
    m = sub.add_parser("merge", help="Merge HuggingFace source models.")
    _common(m); _run_args(m, require_id=False)
    m.add_argument("--method", default="slerp", help="Merge method (slerp, ties, dare).")
    m.add_argument("--model-id", dest="model_ids", action="append", default=[],
                   help="HuggingFace model IDs to merge. Repeatable.")

    # post-merge-train
    pmt = sub.add_parser("post-merge-train", help="Fine-tune after merge to verify quality.")
    _common(pmt); _run_args(pmt, require_id=True)
    pmt.add_argument("--input-checkpoint", help="Override input checkpoint.")
    pmt.add_argument("--training-config", help="Training config YAML.")

    # abliterate
    a = sub.add_parser("abliterate", help="Remove refusal directions.")
    _common(a); _run_args(a, require_id=True)
    a.add_argument("--input-checkpoint", help="Override input checkpoint.")
    a.add_argument("--recipe-config", help="Abliteration recipe YAML (required for --execute).")

    # post-abliterate-train
    pat = sub.add_parser("post-abliterate-train", help="Fine-tune after abliteration.")
    _common(pat); _run_args(pat, require_id=True)
    pat.add_argument("--input-checkpoint", help="Override input checkpoint.")
    pat.add_argument("--training-config", help="Training config YAML.")

    # quantize
    q = sub.add_parser("quantize", help="Quantize the checkpoint.")
    _common(q); _run_args(q, require_id=True)
    q.add_argument("--input-checkpoint", help="Override input checkpoint.")
    q.add_argument("--method", default="gguf", help="Quantization method (gguf, exl2, gptq).")
    q.add_argument("--bits", type=int, default=4, help="Quantization bits.")

    # eval
    e = sub.add_parser("eval", help="Run post-quantize evaluation.")
    _common(e); _run_args(e, require_id=True)
    e.add_argument("--input-checkpoint", help="Override input checkpoint.")
    e.add_argument("--prompts", type=int, default=8, help="Number of eval prompts.")

    # report
    r = sub.add_parser("report", help="Generate dashboard and optionally serve it.")
    r.add_argument("--artifact-dir", help="Runs root.")
    r.add_argument("--run-id", help="Validate a specific run.")
    r.add_argument("--serve", action="store_true", help="Start the dashboard server.")
    r.add_argument("--host", default="127.0.0.1", help="Server host.")
    r.add_argument("--port", type=int, default=8000, help="Server port.")

    return root


# ── Dispatch ────────────────────────────────────────────────────────

def dispatch(args: argparse.Namespace) -> dict[str, Any]:
    cmd = args.command

    if cmd == "merge":
        return run_merge(
            run_id=args.run_id, artifact_dir=args.artifact_dir,
            remote_config=args.remote_config,
            model_ids=args.model_ids or None,
            merge_method=args.method,
            dry_run=args.dry_run, execute=args.execute, resume=args.resume,
            tags=args.tags, notes=args.notes,
        )

    if cmd == "post-merge-train":
        return run_post_merge_train(
            run_id=args.run_id, artifact_dir=args.artifact_dir,
            remote_config=args.remote_config,
            input_checkpoint=args.input_checkpoint,
            training_config_path=args.training_config,
            dry_run=args.dry_run, execute=args.execute, resume=args.resume,
        )

    if cmd == "abliterate":
        return run_abliterate(
            run_id=args.run_id, artifact_dir=args.artifact_dir,
            remote_config=args.remote_config,
            input_checkpoint=args.input_checkpoint,
            recipe_config=args.recipe_config,
            dry_run=args.dry_run, execute=args.execute,
        )

    if cmd == "post-abliterate-train":
        return run_post_abliterate_train(
            run_id=args.run_id, artifact_dir=args.artifact_dir,
            remote_config=args.remote_config,
            input_checkpoint=args.input_checkpoint,
            training_config_path=args.training_config,
            dry_run=args.dry_run, execute=args.execute, resume=args.resume,
        )

    if cmd == "quantize":
        return run_quantize(
            run_id=args.run_id, artifact_dir=args.artifact_dir,
            remote_config=args.remote_config,
            input_checkpoint=args.input_checkpoint,
            quant_method=args.method, quant_bits=args.bits,
            dry_run=args.dry_run, execute=args.execute, resume=args.resume,
        )

    if cmd == "eval":
        return run_post_quantize_eval(
            run_id=args.run_id, artifact_dir=args.artifact_dir,
            remote_config=args.remote_config,
            input_checkpoint=args.input_checkpoint,
            num_prompts=args.prompts,
            dry_run=args.dry_run, execute=args.execute,
        )

    if cmd == "report":
        return run_report(
            artifact_dir=args.artifact_dir, run_id=args.run_id,
            serve=args.serve, host=args.host, port=args.port,
        )

    raise ValueError(f"Unknown command: {cmd}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = dispatch(args)
    except (RuntimeError, ValueError) as exc:
        console.print_json(
            data=json.dumps({"command": getattr(args, "command", None), "error": str(exc)}, indent=2)
        )
        return 1
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


if __name__ == "__main__":
    raise SystemExit(main())
