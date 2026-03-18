# Architecture Notes

## Core Principle
This repo is intentionally split into stage-shaped modules because the workflow is the product. The code is not trying to be a monolith that “just knows” how to fuse three large image systems. It is trying to make each decision inspectable, repeatable, and debuggable.

## Local vs Remote
- Local: CLI, config validation, report generation, dry-runs, test coverage.
- Remote: model materialization, tensor scans, merge execution, quantization, deployment benchmarks.

## Why The Layered Model Is Treated Differently
`Qwen-Image-Layered` brings genuinely useful decomposition behavior, but it also carries the most obvious structural mismatch: RGBA-VAE and Layer3D-style positional behavior. The scaffold treats it as a donor of capabilities, not as a naive merge peer.

