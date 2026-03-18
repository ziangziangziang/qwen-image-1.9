# Remote Execution

## Why remote-first
The scaffold assumes the local machine is for authoring and sanity checks. Real runs happen on a remote system with the actual weights, storage, and accelerator stack.

## Expected flow
1. Copy or clone this repo to the remote machine.
2. Fill in `configs/remote/cluster.example.env` and `configs/remote/paths.example.yaml`.
3. Run dry-runs locally until the manifests look sane.
4. Execute stage scripts remotely and sync the Markdown/JSON artifacts back.

## What should come back
- Stage reports in `reports/`
- JSON manifests
- Selected benchmark summaries

