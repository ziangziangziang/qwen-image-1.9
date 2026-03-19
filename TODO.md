# TODO

## Stage 1: Structural Fusion Analysis
Goal: produce the DNA report that decides what can merge directly, what needs delta logic, and what is structurally incompatible.

Deliverables:
- [ ] `reports/stage-1-dna-report.md`
- [ ] `reports/stage-1-compatibility-matrix.json`

Tasks:
- [ ] S1.1 Export model metadata from remote checkpoints
- [ ] S1.2 Compare `state_dict` key overlap between 2512 and 2511
- [ ] S1.3 Quantify tensor-family similarity and mismatch rates
- [ ] S1.4 Analyze RGB-VAE vs RGBA-VAE compatibility
- [ ] S1.5 Probe text encoder and RoPE assumptions
- [ ] S1.6 Render DNA visuals and compatibility tables

Dependencies:
- Remote paths for all source checkpoints
- Access to enough metadata to inspect tensor names and shapes

Remote execution notes:
- Run inspection scripts on the remote machine and sync only JSON/Markdown artifacts back into this repo.
- Keep all weight paths in `configs/remote/paths.example.yaml`, not in code.

Acceptance criteria:
- [ ] Every major subsystem is labeled `direct-merge`, `delta-merge`, `adapter-only`, or `incompatible`.

## Stage 2: Fusion
Goal: produce one reproducible BF16 merge recipe for a community-bridge checkpoint or a clearly documented multi-part bundle.

Deliverables:
- [ ] `reports/stage-2/README.md`
- [ ] `reports/stage-2/merge-manifest.json`
- [ ] `reports/stage-2/dataset-manifest.json`

Tasks:
- [ ] S2.1 Build transformer-only edit delta recipe from `Qwen/Qwen-Image` ancestry into `2512`
- [ ] S2.2 Sweep coefficients `0.20`, `0.30`, `0.35`, `0.40` and pick a provisional stable core
- [ ] S2.3 Generate a synthetic teacher dataset from `2512`, `Edit-2511`, and `Layered`
- [ ] S2.4 Define the Layered bridge scope and trainable adapter/gate modules
- [ ] S2.5 Compose a dual-track manifest for the stable core and experimental bridge branch
- [ ] S2.6 Run smoke prompts for generation, editing, and bridge behavior

Dependencies:
- Stage 1 compatibility matrix
- Stage 1 weight analysis
- Remote BF16 merge runtime

Remote execution notes:
- Store remote job IDs, artifact URIs, and exact command lines in the manifest.
- Keep manifest and report paths relative; avoid absolute path leakage in public artifacts.

Acceptance criteria:
- [ ] One stable core recipe exists that another engineer can run without reverse-engineering notebook fragments.
- [ ] One experimental Layered bridge branch is documented with explicit exclusions for direct `text_encoder`, `vae`, and `rope` merge paths.

## Stage 3: Safety Evaluation and Risk Report
Goal: measure what the merged system can do, where it fails, and what should block a public release candidate.

Deliverables:
- [ ] `reports/stage-3-eval-report.md`

Tasks:
- [ ] S3.1 Run benign generation and editing capability suite
- [ ] S3.2 Evaluate layered decomposition reliability
- [ ] S3.3 Record prompt-boundary and failure-case behavior
- [ ] S3.4 Draft release notes and risk posture

Dependencies:
- Stage 2 BF16 artifact

Remote execution notes:
- Keep datasets remote. Sync summary metrics and curated examples only.

Acceptance criteria:
- [ ] Risk posture is explicit enough that a public release decision is defensible.

## Stage 4: Compression
Goal: make the merged system smaller without casually deleting its eyesight.

Deliverables:
- [ ] `reports/stage-4-quantization-report.md`

Tasks:
- [ ] S4.1 Choose target GGUF and EXL2/GPTQ recipes
- [ ] S4.2 Generate imatrix from remote image set
- [ ] S4.3 Run quality regression comparisons against BF16
- [ ] S4.4 Benchmark throughput and memory on remote hardware
- [ ] S4.5 Recommend one GGUF path and one EXL2/GPTQ path

Dependencies:
- Stage 2 BF16 artifact
- Remote quantization toolchain

Remote execution notes:
- Quantized artifacts stay remote or in a release bucket; local repo stores manifests and reports.

Acceptance criteria:
- [ ] Each recommended recipe has reproducible commands and regression notes.

## Stage 5: Deployment and vLLM-Omni Integration
Goal: define a working stage graph for the merged pipeline and benchmark it remotely.

Deliverables:
- [ ] `reports/stage-5-deployment-report.md`
- [ ] generated stage config from `configs/deploy/stage-5-vllm-omni-stage-config.yaml`

Tasks:
- [ ] S5.1 Generate deployment config from template and remote launcher settings
- [ ] S5.2 Validate stage graph schema
- [ ] S5.3 Benchmark runtime on remote AMD hardware
- [ ] S5.4 Record known limitations and operator notes

Dependencies:
- Stage 2 BF16 artifact
- Remote vLLM-Omni environment

Remote execution notes:
- Record exact kernel flags, ROCm version, and runtime knobs in the deployment report.

Acceptance criteria:
- [ ] Deployment recipe is documented end-to-end and benchmarked.
