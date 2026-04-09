# AGENTS.md — Qwen-Image 1.9

## Project Overview
Tri-capability checkpoint pipeline for Qwen-Image models: **merge → train → abliterate → quantize → publish**. Produces a single LoRA-tuned model serving generation, editing, and layering from one checkpoint. Each step emits run-scoped JSON contracts and Markdown reports under `reports/runs/<run_id>/`. A lightweight HTTP server exposes a JSON API for dashboard visualization.

## Execution Environment
- **This environment is dev-only.** Do not attempt to run the pipeline locally.
- All `--execute` workloads run on a remote machine with **2× 80G VRAM GPUs** (AMD MI300X, ROCm 6.2).
- This environment is for editing code, running `make test`, validating contracts, and dry-run verification only.
- Remote execution is configured via `configs/remote/*.yaml`, `.env` files, or `REMOTE_*` environment variables resolved by `default_remote_context()`.
- GPU commands require `sg render -c "..."` wrapper.

## Commands
- **Test:** `make test` (runs `python -m unittest discover -s tests -p 'test_*.py'`)
- **Install:** `make install-editable` (runs `pip install -e .`)
- **CLI entry:** `q19 <command>` or `python -m qwen_image_19.cli <command>`

### CLI Commands
| Command | Purpose |
| --- | --- |
| `q19 preflight` | Inspect source checkpoints, build compatibility matrix |
| `q19 merge` | Build merged checkpoint lineage (tri-capability SLERP) |
| `q19 post-merge-train` | LoRA fine-tune across generation + editing + layering |
| `q19 abliterate` | Apply refusal-direction removal |
| `q19 post-abliterate-train` | Stabilize abliterated model |
| `q19 quantize` | Produce GGUF, GPTQ, EXL2 artifacts |
| `q19 eval` | Post-quantize quality audit |
| `q19 report` | Generate shared results index, optionally serve API |
| `q19 publish` | Upload to HuggingFace Hub |

### Common Flags
- `--dry-run` — resolve configs and print outputs without writing
- `--smoke-run` — minimal quick pass to prove pipeline wiring
- `--execute` — execute full workload (resource-intensive)
- `--resume` — resume from prior outputs instead of overwriting
- `--run-id <id>` — stable run identifier under `reports/runs/<run_id>`

## Architecture

### Core Modules
| File | Responsibility |
| --- | --- |
| `src/qwen_image_19/cli.py` | CLI parser (argparse) and command dispatcher |
| `src/qwen_image_19/workflow_v2.py` | Orchestration for all pipeline steps |
| `src/qwen_image_19/contracts.py` | Run manifest, step result, artifact reference contracts |
| `src/qwen_image_19/abliterate.py` | Refusal-direction tensor removal (real implementation with safetensors) |
| `src/qwen_image_19/reporting.py` | Run index, dashboard generation, run collection |
| `src/qwen_image_19/webserver.py` | Lightweight JSON API server (`ThreadingHTTPServer`) |
| `src/qwen_image_19/config_io.py` | JSON/YAML I/O, repo root resolution |
| `src/qwen_image_19/logging_utils.py` | Rich console logging with fallback |
| `src/qwen_image_19/remote/__init__.py` | Remote execution context resolution (env/JSON/.env) |

### Pipeline Modules
| Module | Purpose |
| --- | --- |
| `pipeline/training.py` | LoRA fine-tuning (all 3 capabilities: generation + editing + layering) |
| `pipeline/abliteration.py` | Tensor-level refusal direction removal |
| `pipeline/eval_runner.py` | Quality evaluation runner |

### Stage Modules
| Module | Purpose |
| --- | --- |
| `stage_1_analysis/` | Checkpoint inspection, tensor comparison, compatibility matrix, weight analysis |
| `stage_2_fusion/` | Merge planning, dataset manifests, job execution, training reports |
| `stage_3_eval/` | Eval summary builder, per-step eval suite definitions |
| `stage_4_quant/` | Quantization profile loading (GGUF imatrix, EXL2/GPTQ) |
| `stage_6_publish/` | HuggingFace Hub upload |

## Run Contract
Every pipeline run lives under `reports/runs/<run_id>/`:

```
<run_id>/
  manifest.json          # Canonical run manifest (source models, steps, tags, notes)
  report-index.json      # Dashboard-facing index
  merge/
    step-result.json     # Merge execution record
    eval-summary.json    # Post-merge evaluation
    README.md            # Human-readable report
    samples/             # Image samples
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

### Manifest Schema
- `run_id`, `created_at`, `updated_at` — identity and timestamps
- `artifact_root` — path to run directory
- `source_models` — model metadata from `configs/models/*.yaml`
- `steps` — keyed by `merge`, `abliterate`, `quantize`, each with status, checkpoints, artifacts, metrics
- `report_index` — path to `report-index.json`
- `tags`, `notes` — operator-facing metadata

## Results Server API
| Endpoint | Response |
| --- | --- |
| `GET /` | Plain text welcome |
| `GET /api/runs` | List of all runs with status |
| `GET /api/runs/{run_id}` | Full run manifest |
| `GET /api/runs/{run_id}/steps/{step}` | Step result JSON |
| `GET /api/runs/{run_id}/steps/{step}/samples` | Sample file listing |

## Key Conventions
- **Path resolution:** `public_path()` produces repo-relative POSIX paths; remote URIs (s3://, gs://, hf://, etc.) pass through unchanged
- **Timestamps:** UTC via `utc_now()` (ISO 8601, no microseconds)
- **JSON:** Written with `sort_keys=True, indent=2`, trailing newline
- **Optional dependencies:** `torch`, `safetensors`, `rich`, `tqdm`, `yaml` all guarded with try/except
- **Remote execution:** `default_remote_context()` resolves from env vars, `.env` files, or JSON configs
- **Dry-run mode:** Returns results dict without writing any files
- **`--execute` flag:** Required for real workload execution; abliterate additionally requires `--recipe-config`
- **Step chaining:** `abliterate` reads from merge output; `quantize` reads from abliterate output (overridable via `--input-checkpoint`)

## Configuration
- **Model metadata:** `configs/models/*.yaml` (JSON format with alias, model_id, role)
- **Merge profiles:** `configs/merge/*.yaml` (run profiles, candidate weights)
- **Training datasets:** `configs/merge/stage-2-synthetic-dataset.yaml` (generation, editing, layering splits)
- **Training config:** `configs/merge/stage-2-training-full.yaml` (LoRA rank, LR, steps)
- **Quantization:** `configs/quant/*.yaml` (GGUF imatrix, EXL2/GPTQ profiles)
- **Abliteration:** `configs/abliterate/*.yaml` (recipe with measurements, ablation orders)
- **Remote:** `configs/remote/*.yaml` (launcher, paths, cache maps)
- **Deployment:** `configs/deploy/*.yaml` (vLLM stage config templates)

## Testing
- **Framework:** `unittest` (not pytest)
- **Command:** `make test` or `PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py'`
- **Fixtures:** `tests/conftest.py` provides shared test utilities
- **Coverage areas:**
  - CLI argument parsing and dispatch
  - Contract functions (manifests, step results, public paths)
  - Workflow orchestration
  - Stage configs (quant, eval registry, stage-5 schema)
  - Merge recipes and diffusion runtime
  - Abliteration modes (CLI, tensor ops)

## TODO
See `TODO.md` for open items. Key areas:
- Resume/retry semantics for `abliterate` and `quantize`
- Real eval metrics replacing placeholder values
- HTML dashboard on top of JSON API
- Thumbnail serving and before/after image presentation
- Layered bridge research track isolation
