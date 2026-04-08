# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-04-08

### Added
- Tri-capability LoRA training across all 60 MMDiT blocks (generation + editing + layering).
- `artplus/PrismLayersReal` dataset integration for layer-aware training.
- Round-robin prompt interleaving across all three capability splits.
- `scripts/` directory for utility scripts (download, generate, upload).
- `py.typed` marker for PEP 561 type hint support.
- `[project.optional-dependencies]` for dev tooling.
- Pre-commit configuration with ruff and standard hooks.
- `MANIFEST.in` for proper sdist packaging.
- `CHANGELOG.md`, `CONTRIBUTING.md` community files.

### Changed
- LoRA target expanded from bridge window (blocks 40–59) to full backbone (all 60 blocks).
- Prompt producer rewritten for balanced 3-split round-robin instead of sequential generation-only.
- Project version sourced from `importlib.metadata` (single source of truth).
- Makefile GPU env variable made portable via `.env` override.
- README fully rewritten for release.

### Removed
- Legacy `workflow.py` (replaced by `workflow_v2.py`).
- Legacy `webserver.py` (replaced by `server.py`).
- Stale temporary files and runtime artifacts from repo root.

## [0.2.0] - 2026-03-29

### Added
- Delta-edit merge pipeline (foundation + edit donor at 0.35 coefficient).
- Post-merge LoRA fine-tuning with bridge-window targeting (blocks 40–59).
- Abliteration module with refusal-direction tensor removal.
- Quantization profiles for GGUF, GPTQ, EXL2.
- Results server with JSON API.
- CLI with 8 pipeline commands.

## [0.1.0] - 2026-03-15

### Added
- Initial project structure with src layout.
- Checkpoint inspection and compatibility matrix.
- Stage-based pipeline architecture.
