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
from qwen_image_19.stage_1_analysis.benchmark import run_gpu_stress_test
from qwen_image_19.workflow_v2 import (
    run_abliterate,
    run_merge,
    run_post_abliterate_train,
    run_post_merge_train,
    run_post_quantize_eval,
    run_preflight,
    run_prepare_judge,
    run_publish,
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

    # preflight
    pf = sub.add_parser("preflight", help="Inspect checkpoints and benchmark the device.")
    _common(pf); _run_args(pf, require_id=False)
    pf.add_argument("--skip-benchmark", action="store_true",
                    help="Skip the ~5-min GPU performance benchmark.")
    pf.add_argument("--benchmark-seconds", type=int, default=300,
                    help="Target benchmark duration in seconds (default 300).")
    pf.add_argument("--stress-test", action="store_true",
                    help="Run continuous GPU stress test at 100%% utilization for benchmark-seconds.")
    pf.add_argument("--stress-test-seconds", type=int, default=300,
                    help="Duration for stress test in seconds (default 300).")

    # merge
    m = sub.add_parser("merge", help="Merge HuggingFace source models.")
    _common(m); _run_args(m, require_id=False)
    m.add_argument("--method", default="slerp", help="Merge method (slerp, ties, dare).")
    m.add_argument("--recipe", default="tri-capability",
                   choices=["tri-capability", "delta-edit", "slerp-selective"],
                   help="Merge recipe. 'tri-capability' (default): generation + editing + layering. "
                        "'delta-edit': generation + editing only (legacy). "
                        "'slerp-selective': direct per-block SLERP between gen and edit models (recommended).")
    m.add_argument("--model-id", dest="model_ids", action="append", default=[],
                   help="HuggingFace model IDs to merge. Repeatable.")
    m.add_argument("--edit-coefficient", type=float, default=0.35,
                   help="Delta coefficient for edit capability blend (default: 0.35).")
    m.add_argument("--layer-coefficient", type=float, default=0.25,
                   help="Delta coefficient for layering capability blend (default: 0.25).")

    # post-merge-train
    pmt = sub.add_parser("post-merge-train", help="Fine-tune after merge to verify quality.")
    _common(pmt); _run_args(pmt, require_id=True)
    pmt.add_argument("--input-checkpoint", help="Override input checkpoint.")
    pmt.add_argument("--training-config", help="Training config YAML.")

    # prepare-judge
    pj = sub.add_parser(
        "prepare-judge",
        help="Abliterate the quality judge (Qwen3.5-35B-A3B) before eval.",
    )
    _common(pj)
    pj.add_argument(
        "--judge-model",
        default=None,
        help="HuggingFace model ID or local path for the judge (default: Qwen/Qwen3.5-35B-A3B).",
    )
    pj.add_argument(
        "--recipe-config",
        default=None,
        help="Abliteration recipe YAML. Defaults to configs/abliterate/judge-abliteration.yaml.",
    )
    pj.add_argument(
        "--output-path",
        default="/scratch/qwen-judge-abliterated",
        help="Directory to write the abliterated judge checkpoint (default: /scratch/qwen-judge-abliterated).",
    )
    pj.add_argument(
        "--measurements",
        default=None,
        help="Override path to pre-computed measurements .pt file.",
    )
    pj.add_argument(
        "--skip-comparison",
        action="store_true",
        help="Skip the before/after comparison report after execution.",
    )
    pj.add_argument(
        "--measure-pairs",
        type=int,
        default=64,
        help="Number of harmful/harmless prompt pairs used to compute judge directions (default: 64).",
    )

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
    q.add_argument("--method", default="all",
                   choices=["all", "gguf", "gptq", "exl2"],
                   help="Quantization format(s) to produce. "
                        "'all' (default): GGUF + GPTQ (vllm-omni) + EXL2. "
                        "'gptq': GPTQ marlin only — primary vllm-omni format. "
                        "'gguf': GGUF only (llama.cpp/ollama). "
                        "'exl2': EXL2 only (exllamav2/TabbyAPI).")
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

    # publish
    pub = sub.add_parser("publish", help="Upload artifacts and model card to HuggingFace.")
    _common(pub); _run_args(pub, require_id=True)
    pub.add_argument("--repo-id", default="ThirdMiddle/Qwen-Image-1.9",
                     help="HuggingFace repo ID (user/name).")
    pub.add_argument("--private", action="store_true", help="Create repo as private.")
    pub.add_argument("--hf-token", help="HuggingFace token override (defaults to .env).")

    return root


# ── Dispatch ────────────────────────────────────────────────────────

def dispatch(args: argparse.Namespace) -> dict[str, Any]:
    cmd = args.command

    if cmd == "preflight":
        if args.stress_test:
            from pathlib import Path
            from qwen_image_19.workflow_v2 import default_run_id, runs_root
            rid = args.run_id or default_run_id("preflight")
            root = Path(args.artifact_dir) if args.artifact_dir else None
            if root:
                run_dir = root / rid / "preflight"
            else:
                run_dir = runs_root() / rid / "preflight"
            return run_gpu_stress_test(
                run_id=rid,
                output_dir=run_dir,
                target_seconds=args.stress_test_seconds,
            )
        return run_preflight(
            run_id=args.run_id, artifact_dir=args.artifact_dir,
            dry_run=args.dry_run, execute=args.execute,
            skip_benchmark=args.skip_benchmark,
            benchmark_seconds=args.benchmark_seconds,
            tags=args.tags, notes=args.notes,
        )

    if cmd == "merge":
        return run_merge(
            run_id=args.run_id, artifact_dir=args.artifact_dir,
            remote_config=args.remote_config,
            model_ids=args.model_ids or None,
            merge_method=args.method,
            recipe=args.recipe,
            edit_coefficient=args.edit_coefficient,
            layer_coefficient=args.layer_coefficient,
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

    if cmd == "prepare-judge":
        return run_prepare_judge(
            judge_model=args.judge_model,
            output_path=args.output_path,
            recipe_config=args.recipe_config,
            measurements=args.measurements,
            measure_pairs=args.measure_pairs,
            dry_run=args.dry_run,
            execute=args.execute,
            remote_config=args.remote_config,
            skip_comparison=args.skip_comparison,
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

    if cmd == "publish":
        return run_publish(
            run_id=args.run_id, artifact_dir=args.artifact_dir,
            repo_id=args.repo_id,
            private=args.private,
            hf_token=args.hf_token,
            dry_run=args.dry_run, execute=args.execute,
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
