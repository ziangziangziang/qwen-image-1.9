# Qwen-Image 1.9

This package now exposes a 3-step checkpoint pipeline:

1. `merge`
2. `abliterate`
3. `quantize`

Every step emits a run-scoped result bundle, a compact eval summary, and a Markdown report. A single internal results server reads the JSON artifacts directly from `reports/runs/<run_id>/...`; it does not scrape Markdown.

## Operating Model
- `merge` builds the merged checkpoint lineage and records the merge recipe/evidence.
- `abliterate` applies refusal-direction removal to the merged checkpoint and records the policy-sensitive deltas that must be reviewed.
- `quantize` produces smaller deployment-oriented artifacts and records quality, latency, and memory regressions.
- `preflight` remains available as evidence gathering for source checkpoints, but it is no longer the product-facing stage model.

The layered donor remains a separate research track. It is still useful, but it is not part of the mainline `merge -> abliterate -> quantize` release path.

## Artifact Contract
Each run lives under:

```text
reports/runs/<run_id>/
  manifest.json
  report-index.json
  merge/
    step-result.json
    eval-summary.json
    README.md
    samples/
  abliterate/
    step-result.json
    eval-summary.json
    README.md
    samples/
  quantize/
    step-result.json
    eval-summary.json
    README.md
    samples/
```

Large checkpoints and quantized binaries stay in remote or object storage. The repo stores metadata, summaries, thumbnails, and URIs only.

## CLI
Primary commands:

```bash
python3 -m pip install -e .
q19 preflight --dry-run
q19 merge --run-id run-001 --run-profile full
q19 abliterate --run-id run-001
q19 quantize --run-id run-001
q19 report
q19 report --serve --host 127.0.0.1 --port 8000
```

If `q19` still shows the old `stage1..stage5` help text, you are running an older installed console script. Reinstall the workspace package with `python3 -m pip install -e .` and check again.

The old `stage*` CLI surface has been removed. Use the 5 commands above.

## Results Server
The internal results server exposes stage-neutral endpoints:

- `GET /api/runs`
- `GET /api/runs/{run_id}`
- `GET /api/runs/{run_id}/steps/{step}`
- `GET /api/runs/{run_id}/steps/{step}/samples`

Its job is to visualize lineage, metrics, reports, and image samples across all stages from a single dashboard.

## Governance
`abliterate` is an internal-only, policy-sensitive operation in this repo. A run is not considered complete unless:

- per-step eval summaries exist
- refusal-behavior deltas are recorded after `abliterate`
- regression metrics are recorded after `quantize`
- the run manifest and report index are present for the shared dashboard

## Repo Map
```text
src/      CLI, workflow orchestration, contracts, stage modules, results server
configs/  model metadata and merge/quantization recipes
reports/  committed examples plus run-scoped JSON/Markdown outputs
docs/     architecture and execution notes
tests/    CLI, contract, and workflow coverage
```
