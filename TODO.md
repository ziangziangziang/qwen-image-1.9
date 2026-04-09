# TODO

## Training
- [ ] Add image-conditioned training for layering (currently caption-only from PrismLayersReal).
- [ ] Per-capability loss tracking (separate metrics for generation / editing / layering prompts).
- [ ] Experiment with capability-weighted sampling ratios instead of pure round-robin.

## Pipeline
- [ ] Add explicit resume/retry semantics to `abliterate` and `quantize`.
- [ ] Replace placeholder eval metrics with real remote summaries when available.

## Abliterate
- [ ] Record the exact method, target layers, and review sign-off in the step result.
- [ ] Add stricter gating around internal-only outputs and release posture.

## Quantize
- [ ] Wire real quantized artifact URIs and per-format benchmark outputs into the step result.
- [ ] Append throughput/memory measurements from remote hardware instead of placeholder values.

## Dashboard
- [ ] Add an HTML dashboard on top of the JSON API.
- [ ] Add thumbnail discovery and before/after image presentation for all steps.
- [ ] Add filtering by tag, status, and recency across runs.
