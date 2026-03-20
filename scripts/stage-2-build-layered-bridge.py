#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import time

from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from qwen_image_19.stage_2_fusion import fuse

try:
    from safetensors.torch import save_file
except Exception as exc:  # pragma: no cover
    raise SystemExit("safetensors is required for bridge artifact writing.") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output-adapter", help="Relative adapter output path.")
    parser.add_argument("--output-checkpoint", help="Relative checkpoint output path.")
    parser.add_argument("--metrics-output", help="Relative metrics output JSON path.")
    parser.add_argument("--dataset-root", default="stage-2/datasets/teacher-db")
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def write_text_json(path: str, payload: dict[str, object]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def image_to_tensor(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    width, height = image.size
    data = torch.tensor(list(image.getdata()), dtype=torch.float32).view(height, width, 3)
    return data.permute(2, 0, 1).contiguous() / 255.0


class TinyBridge(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


if __name__ == "__main__":
    args = parse_args()
    if not args.execute:
        print(json.dumps(fuse(dry_run=True)["manifest"]["layered_bridge_recipe"], indent=2))
        raise SystemExit(0)
    if not args.output_adapter or not args.output_checkpoint:
        raise SystemExit("--output-adapter and --output-checkpoint are required with --execute")
    if not torch.cuda.is_available():
        raise SystemExit("GPU is required for true Stage 2 smoke execution.")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset_root = Path(args.dataset_root)
    image_paths = sorted(dataset_root.rglob("*.png"))
    if not image_paths:
        raise SystemExit(f"No dataset PNG files found under: {args.dataset_root}")

    tensors = [image_to_tensor(path) for path in image_paths]
    device = "cuda"
    tensors = [tensor.to(device) for tensor in tensors]
    model = TinyBridge().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_curve: list[float] = []
    started = time.perf_counter()
    for _ in range(max(1, args.max_steps)):
        batch = random.sample(tensors, k=min(len(tensors), max(1, args.batch_size)))
        batch_tensor = torch.stack(batch, dim=0)
        prediction = model(batch_tensor)
        loss = F.mse_loss(prediction, batch_tensor)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss_curve.append(float(loss.detach().cpu().item()))

    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    output_adapter = Path(args.output_adapter)
    output_checkpoint = Path(args.output_checkpoint)
    output_adapter.parent.mkdir(parents=True, exist_ok=True)
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, str(output_adapter), metadata={"artifact_type": "layered-bridge-adapter-smoke-poc"})
    save_file(state, str(output_checkpoint), metadata={"artifact_type": "layered-bridge-checkpoint-smoke-poc"})
    elapsed = time.perf_counter() - started
    del model
    torch.cuda.empty_cache()

    if args.metrics_output:
        write_text_json(
            args.metrics_output,
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "max_steps": args.max_steps,
                "batch_size": args.batch_size,
                "dataset_root": args.dataset_root,
                "dataset_image_count": len(image_paths),
                "loss_curve": loss_curve,
                "final_loss": loss_curve[-1] if loss_curve else None,
                "elapsed_seconds": round(elapsed, 3),
            },
        )
    print(
        json.dumps(
            {
                "status": "ok",
                "output_adapter": args.output_adapter,
                "output_checkpoint": args.output_checkpoint,
                "metrics_output": args.metrics_output,
            },
            indent=2,
        )
    )
