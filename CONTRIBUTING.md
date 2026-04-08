# Contributing to Qwen-Image 1.9

Thanks for your interest in contributing! This document covers the basics.

## Development Setup

```bash
git clone https://github.com/ThirdMiddle/Qwen-Image-1.9.git
cd Qwen-Image-1.9
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## Running Tests

```bash
make test
```

All tests use `unittest`. Run a specific test file:

```bash
PYTHONPATH=src python3 -m unittest tests.test_cli_stage2_modes -v
```

## Code Style

This project uses [ruff](https://docs.astral.sh/ruff/) for linting and formatting.
Pre-commit hooks run automatically on each commit. To run manually:

```bash
pre-commit run --all-files
```

## Making Changes

1. Create a branch from `main`.
2. Make your changes with clear, focused commits.
3. Ensure `make test` passes with no failures.
4. Open a pull request with a description of what changed and why.

## Pipeline Execution

GPU workloads (`--execute`) require an AMD MI300X with ROCm. The dev environment
is for editing code, running tests, and `--dry-run` verification only. Do not
attempt to run `--execute` commands locally without GPU access.

## Reporting Issues

Open an issue with:
- What you expected to happen
- What actually happened
- Steps to reproduce
- Python version and OS
