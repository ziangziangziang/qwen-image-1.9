# TODO

## Public Pipeline
- [ ] Replace remaining user-facing `stageN` references in docs and reports with `merge`, `abliterate`, `quantize`, or `preflight`.
- [ ] Add explicit resume/retry semantics to `abliterate` and `quantize`.
- [ ] Decide whether `report --serve` should also serve static thumbnails and galleries.

## Merge
- [ ] Replace placeholder merge-level eval metrics with real remote summaries when available.
- [ ] Move the current layered bridge work behind a dedicated research-track entrypoint instead of letting it leak into the main merge narrative.

## Abliterate
- [ ] Replace the placeholder abliteration command stub with the real refusal-direction removal implementation.
- [ ] Record the exact method, target layers, and review sign-off in the step result.
- [ ] Add stricter gating around internal-only outputs and release posture.

## Quantize
- [ ] Wire real quantized artifact URIs and per-format benchmark outputs into the step result.
- [ ] Append throughput/memory measurements from remote hardware instead of placeholder values.

## Shared Results Server
- [ ] Add an HTML dashboard on top of the JSON API.
- [ ] Add thumbnail discovery and before/after image presentation for all steps.
- [ ] Add filtering by tag, status, and recency across runs.
