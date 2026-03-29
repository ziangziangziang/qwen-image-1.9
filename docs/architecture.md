# Architecture Notes

## Core Principle
The workflow is the product, but the public workflow is now a 3-step operational pipeline:

1. `merge`
2. `abliterate`
3. `quantize`

Each step writes a run-scoped JSON contract first and a Markdown report second. The shared results server reads the JSON contracts, not the Markdown.

## Run-Scoped Contracts
Every pipeline run has one canonical manifest at `reports/runs/<run_id>/manifest.json`. That manifest points to:

- `merge/step-result.json`
- `abliterate/step-result.json`
- `quantize/step-result.json`
- per-step `eval-summary.json`
- `report-index.json` for the shared dashboard

This keeps lineage explicit: source models, step inputs, step outputs, remote jobs, metrics, reports, and artifact URIs all live in one place.

## Local vs Remote
- Local: CLI, config validation, contract generation, report generation, test coverage, results server.
- Remote: model materialization, tensor scans, merge execution, refusal-direction removal, quantization, and heavyweight evals.

## Eval Model
Eval is no longer a standalone stage. It is a required post-step artifact after `merge`, `abliterate`, and `quantize`. Each eval summary includes:

- `eval_suite_id`
- `checkpoint_ref`
- `task_type`
- `samples`
- `aggregate_metrics`
- `failures`
- `judge` metadata

## Layered Research Track
`Qwen-Image-Layered` remains important, but it is not in the mainline release path. Its bridge work stays as a sidecar research track until the core 3-step pipeline is stable.

## Governance Boundary
`abliterate` in this repo means refusal-direction removal on the merged checkpoint. That step is internal-only and policy-sensitive. The architecture therefore requires:

- an explicit `abliterate` step record in the run manifest
- mandatory post-step evaluation
- result visibility through the shared dashboard
- no large binary storage in git
