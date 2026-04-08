"""Device performance benchmark — preflight stage.

Runs a ~5-minute suite of GPU/CPU/IO micro-benchmarks and produces:

  - ``benchmark.json``           machine-readable results for this run
  - ``figures/gemm.svg``         GEMM throughput vs matrix size
  - ``figures/bandwidth.svg``    memory bandwidth vs tensor size
  - ``figures/attention.svg``    SDPA tokens/s vs sequence length
  - ``figures/summary.svg``      normalised performance dashboard

Shared (append-on-each-run) artefacts at ``reports/``:

  - ``benchmark-history.json``   all historical benchmark entries
  - ``benchmark-report.md``      visualization-rich Markdown, one section per device/run

Usage::

    from qwen_image_19.stage_1_analysis.benchmark import run_benchmark
    result = run_benchmark(
        run_id="prod-001",
        output_dir=Path("reports/runs/prod-001/preflight"),
        shard_paths=[...],   # optional: disk-IO sample shards
        target_seconds=300,
    )
"""
from __future__ import annotations

import gc
import json
import os
import platform
import socket
import time
from pathlib import Path
from typing import Any


# ── Theoretical reference peaks ──────────────────────────────────────
# Used only to compute "% of theoretical" labels in reports.
# Keyed by substring matches against torch.cuda.get_device_name().

_THEORY_PEAKS: list[tuple[str, dict[str, float]]] = [
    # device_name substring → {peak_tflops_bf16, peak_bw_gbs, label}
    ("MI300X", {"gemm_tflops": 653.7,  "bw_gbs": 5325.0, "label": "MI300X 192 GB"}),
    ("MI250X", {"gemm_tflops": 383.0,  "bw_gbs": 3276.8, "label": "MI250X"}),
    ("MI300A", {"gemm_tflops": 383.0,  "bw_gbs": 3072.0, "label": "MI300A"}),
    ("A100",   {"gemm_tflops": 312.0,  "bw_gbs": 2000.0, "label": "A100 80 GB"}),
    ("H100",   {"gemm_tflops": 989.5,  "bw_gbs": 3350.0, "label": "H100 SXM"}),
    ("H200",   {"gemm_tflops": 989.5,  "bw_gbs": 4800.0, "label": "H200 SXM"}),
    ("4090",   {"gemm_tflops": 165.2,  "bw_gbs":  1008.0, "label": "RTX 4090"}),
    ("3090",   {"gemm_tflops": 71.0,   "bw_gbs":   936.2, "label": "RTX 3090"}),
    ("V100",   {"gemm_tflops": 125.0,  "bw_gbs":   900.0, "label": "V100"}),
]

_DEFAULT_PEAK = {"gemm_tflops": 100.0, "bw_gbs": 1000.0, "label": "unknown"}


def _lookup_peaks(device_name: str) -> dict[str, float]:
    dn = device_name.upper()
    for substring, peaks in _THEORY_PEAKS:
        if substring.upper() in dn:
            return peaks
    return _DEFAULT_PEAK


# ── Text helpers ─────────────────────────────────────────────────────

def _ascii_bar(value: float, max_value: float, width: int = 32) -> str:
    """Return a unicode block bar string proportional to value/max_value."""
    if max_value <= 0:
        return "░" * width
    ratio = min(value / max_value, 1.0)
    filled = int(round(ratio * width))
    return "█" * filled + "░" * (width - filled)


def _pct(value: float, peak: float) -> str:
    if peak <= 0:
        return "N/A"
    return f"{100 * value / peak:.0f}%"


def _si(n: float, unit: str = "", decimals: int = 1) -> str:
    """Format a number with SI prefix."""
    for prefix, threshold in (("T", 1e12), ("G", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(n) >= threshold:
            return f"{n / threshold:.{decimals}f} {prefix}{unit}"
    return f"{n:.{decimals}f} {unit}"


# ── Hardware probe ────────────────────────────────────────────────────

def _probe_device() -> dict[str, Any]:
    """Detect GPU and CPU specs. Returns device info dict."""
    info: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "os": platform.platform(terse=True),
        "python": platform.python_version(),
        "cpu_model": "unknown",
        "cpu_cores": 1,
        "ram_bytes": 0,
        "gpu_available": False,
        "gpu_name": "CPU only",
        "gpu_vram_bytes": 0,
        "torch_version": "unavailable",
        "rocm_version": None,
        "cuda_version": None,
    }

    # CPU
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except Exception:
        info["cpu_model"] = platform.processor() or "unknown"

    import multiprocessing
    info["cpu_cores"] = multiprocessing.cpu_count()

    try:
        import psutil
        info["ram_bytes"] = psutil.virtual_memory().total
    except Exception:
        pass

    # GPU via torch
    try:
        import torch
        info["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu_available"] = True
            info["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["gpu_vram_bytes"] = props.total_memory
            # ROCm version embedded in torch version string
            if "rocm" in torch.__version__.lower():
                info["rocm_version"] = torch.__version__.split("+")[-1]
            else:
                info["cuda_version"] = torch.version.cuda
    except Exception:
        pass

    return info


# ── GEMM benchmark ────────────────────────────────────────────────────

def _bench_gemm(device: str, budget_seconds: float) -> dict[str, Any]:
    """Measure bf16 GEMM throughput at multiple matrix sizes.

    Returns TFLOPS values per matrix dimension N (square M=K=N).
    """
    try:
        import torch
    except ImportError:
        return {"error": "torch unavailable", "sizes": {}}

    if device == "cpu":
        dtype = torch.float32
    else:
        dtype = torch.bfloat16

    # Target sizes: adaptive to available VRAM (skip sizes that would OOM)
    candidate_dims = [1024, 2048, 4096, 8192, 16384]
    if device != "cpu":
        try:
            vram_bytes = torch.cuda.get_device_properties(0).total_memory
            # Each NxN bf16 matrix = N²×2 bytes; three matrices (A, B, C) = 6×N²
            candidate_dims = [
                n for n in candidate_dims
                if n * n * 6 * 2 < vram_bytes * 0.25  # keep under 25% VRAM
            ]
        except Exception:
            candidate_dims = candidate_dims[:3]

    if not candidate_dims:
        candidate_dims = [1024, 2048]

    results: dict[str, Any] = {}
    t_remaining = budget_seconds
    warmup_done = False

    for dim in candidate_dims:
        if t_remaining <= 0:
            break

        try:
            a = torch.randn(dim, dim, dtype=dtype, device=device)
            b = torch.randn(dim, dim, dtype=dtype, device=device)

            if not warmup_done and device != "cpu":
                # Warmup: one matmul to prime the GPU
                for _ in range(3):
                    c = torch.mm(a, b)
                if device != "cpu":
                    torch.cuda.synchronize()
                warmup_done = True

            # Benchmark: repeat until ~5s or budget exhausted
            iters = max(1, min(20, int(5.0 / max(dim / 8192, 0.1))))
            t_per_iter: list[float] = []

            if device != "cpu":
                start_ev = torch.cuda.Event(enable_timing=True)
                end_ev = torch.cuda.Event(enable_timing=True)
                # Warm this dim
                c = torch.mm(a, b)
                torch.cuda.synchronize()
                start_ev.record()
                for _ in range(iters):
                    c = torch.mm(a, b)
                end_ev.record()
                torch.cuda.synchronize()
                elapsed_ms = start_ev.elapsed_time(end_ev)
                avg_ms = elapsed_ms / iters
            else:
                t0 = time.perf_counter()
                for _ in range(iters):
                    c = torch.mm(a, b)
                avg_ms = (time.perf_counter() - t0) / iters * 1000

            # FLOPS for NxN matmul: 2*N^3
            flops = 2 * (dim ** 3)
            tflops = flops / (avg_ms * 1e-3) / 1e12

            results[str(dim)] = {
                "dim": dim,
                "iters": iters,
                "avg_ms": round(avg_ms, 3),
                "tflops": round(tflops, 2),
            }
            t_remaining -= (avg_ms * iters / 1000 + 0.5)

            del a, b, c
            if device != "cpu":
                torch.cuda.empty_cache()

        except Exception as exc:
            results[str(dim)] = {"dim": dim, "error": str(exc)}

    peak_tflops = max((v["tflops"] for v in results.values() if "tflops" in v), default=0.0)
    return {"sizes": results, "peak_tflops": round(peak_tflops, 2)}


# ── Memory bandwidth benchmark ────────────────────────────────────────

def _bench_bandwidth(device: str, budget_seconds: float) -> dict[str, Any]:
    """Measure GPU/CPU memory read+write bandwidth at several tensor sizes.

    Returns GB/s for read (copy-from-device) and write (fill) operations.
    """
    try:
        import torch
    except ImportError:
        return {"error": "torch unavailable", "sizes": {}}

    if device == "cpu":
        # Smaller sizes for CPU
        size_specs = [
            ("256MB", 256 * 1024 * 1024 // 4),   # float32 elements
            ("1GB",   1024 * 1024 * 1024 // 4),
            ("4GB",   4 * 1024 * 1024 * 1024 // 4),
        ]
        dtype = torch.float32
    else:
        vram_bytes = torch.cuda.get_device_properties(0).total_memory
        # bf16 = 2 bytes per element; conserve memory
        def _nelems(target_gb: float) -> int:
            return int(target_gb * 1024**3 / 2)

        size_specs_raw = [
            ("1GB",   _nelems(1)),
            ("4GB",   _nelems(4)),
            ("16GB",  _nelems(16)),
            ("32GB",  _nelems(32)),
        ]
        # Filter to fit in 40% of VRAM (need 2 copies)
        size_specs = [
            (label, n) for label, n in size_specs_raw
            if n * 2 * 2 < vram_bytes * 0.4
        ]
        if not size_specs:
            size_specs = [("1GB", _nelems(1))]
        dtype = torch.bfloat16

    results: dict[str, Any] = {}
    t_remaining = budget_seconds

    for label, nelems in size_specs:
        if t_remaining <= 0:
            break
        try:
            src = torch.zeros(nelems, dtype=dtype, device=device)
            dst = torch.empty_like(src)
            nbytes = src.nbytes
            iters = max(2, min(8, int(6.0 * 1e9 / nbytes) + 1))

            if device != "cpu":
                torch.cuda.synchronize()
                start_ev = torch.cuda.Event(enable_timing=True)
                end_ev = torch.cuda.Event(enable_timing=True)
                # Write benchmark (fill)
                start_ev.record()
                for _ in range(iters):
                    dst.fill_(0.0)
                end_ev.record()
                torch.cuda.synchronize()
                write_ms = start_ev.elapsed_time(end_ev) / iters
                # Read benchmark (copy)
                start_ev.record()
                for _ in range(iters):
                    dst.copy_(src)
                end_ev.record()
                torch.cuda.synchronize()
                read_ms = start_ev.elapsed_time(end_ev) / iters
            else:
                t0 = time.perf_counter()
                for _ in range(iters):
                    dst.fill_(0.0)
                write_ms = (time.perf_counter() - t0) / iters * 1000
                t0 = time.perf_counter()
                for _ in range(iters):
                    dst.copy_(src)
                read_ms = (time.perf_counter() - t0) / iters * 1000

            write_gbs = nbytes / (write_ms * 1e-3) / 1e9
            read_gbs = nbytes / (read_ms * 1e-3) / 1e9

            results[label] = {
                "nbytes": nbytes,
                "write_gbs": round(write_gbs, 1),
                "read_gbs": round(read_gbs, 1),
                "avg_gbs": round((write_gbs + read_gbs) / 2, 1),
            }
            t_remaining -= (write_ms + read_ms) * iters / 1000

            del src, dst
            if device != "cpu":
                torch.cuda.empty_cache()

        except Exception as exc:
            results[label] = {"error": str(exc)}

    peak_bw = max((v["avg_gbs"] for v in results.values() if "avg_gbs" in v), default=0.0)
    return {"sizes": results, "peak_gbs": round(peak_bw, 1)}


# ── SDPA attention benchmark ──────────────────────────────────────────

def _bench_attention(device: str, budget_seconds: float) -> dict[str, Any]:
    """Measure scaled-dot-product-attention throughput at several sequence lengths.

    Simulates a 20B model transformer block: batch=4, heads=32, head_dim=128.
    Returns tokens/second per sequence-length configuration.
    """
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        return {"error": "torch unavailable", "configs": {}}

    if device == "cpu":
        batch, heads, head_dim = 1, 8, 64
        seq_lens = [128, 256, 512]
        dtype = torch.float32
    else:
        batch, heads, head_dim = 4, 32, 128
        seq_lens = [512, 1024, 2048, 4096]
        dtype = torch.bfloat16

    results: dict[str, Any] = {}
    t_remaining = budget_seconds

    for seq_len in seq_lens:
        if t_remaining <= 0:
            break
        try:
            q = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device=device)
            k = torch.randn_like(q)
            v = torch.randn_like(q)

            # Warmup
            if device != "cpu":
                with torch.no_grad():
                    _ = F.scaled_dot_product_attention(q, k, v)
                torch.cuda.synchronize()

            iters = max(2, min(20, int(4.0 * 1e9 / (seq_len * seq_len * heads * batch * 2))))

            if device != "cpu":
                start_ev = torch.cuda.Event(enable_timing=True)
                end_ev = torch.cuda.Event(enable_timing=True)
                start_ev.record()
                with torch.no_grad():
                    for _ in range(iters):
                        out = F.scaled_dot_product_attention(q, k, v)
                end_ev.record()
                torch.cuda.synchronize()
                avg_ms = start_ev.elapsed_time(end_ev) / iters
            else:
                t0 = time.perf_counter()
                with torch.no_grad():
                    for _ in range(iters):
                        out = F.scaled_dot_product_attention(q, k, v)
                avg_ms = (time.perf_counter() - t0) / iters * 1000

            total_tokens = batch * seq_len
            tokens_per_sec = total_tokens / (avg_ms * 1e-3)
            # Attention FLOPs: 4 * B * H * S^2 * D  (QK^T + softmax + @V ~ 4×)
            attn_flops = 4 * batch * heads * seq_len * seq_len * head_dim
            tflops = attn_flops / (avg_ms * 1e-3) / 1e12

            results[str(seq_len)] = {
                "seq_len": seq_len,
                "batch": batch,
                "heads": heads,
                "head_dim": head_dim,
                "avg_ms": round(avg_ms, 3),
                "tokens_per_sec": round(tokens_per_sec),
                "attn_tflops": round(tflops, 3),
            }
            t_remaining -= avg_ms * iters / 1000 + 0.5

            del q, k, v, out
            if device != "cpu":
                torch.cuda.empty_cache()

        except Exception as exc:
            results[str(seq_len)] = {"seq_len": seq_len, "error": str(exc)}
            break  # usually OOM — stop here

    peak_tok = max(
        (v["tokens_per_sec"] for v in results.values() if "tokens_per_sec" in v), default=0
    )
    return {"configs": results, "peak_tokens_per_sec": peak_tok}


# ── CPU parallel benchmark ────────────────────────────────────────────

def _bench_cpu(budget_seconds: float) -> dict[str, Any]:
    """Quick bf16 (or fp32) matmul benchmark on CPU to measure compute throughput."""
    try:
        import torch
    except ImportError:
        return {"error": "torch unavailable"}

    dims = [256, 512, 1024]
    results: dict[str, Any] = {}

    for dim in dims:
        try:
            a = torch.randn(dim, dim, dtype=torch.float32)
            b = torch.randn(dim, dim, dtype=torch.float32)
            iters = max(3, int(3.0 / max((dim / 512) ** 3 * 0.01, 0.001)))
            iters = min(iters, 50)
            t0 = time.perf_counter()
            for _ in range(iters):
                c = torch.mm(a, b)
            elapsed = time.perf_counter() - t0
            avg_s = elapsed / iters
            tflops = 2 * dim ** 3 / avg_s / 1e12
            gflops = tflops * 1000
            results[str(dim)] = {
                "dim": dim,
                "avg_ms": round(avg_s * 1000, 2),
                "gflops": round(gflops, 2),
            }
        except Exception as exc:
            results[str(dim)] = {"error": str(exc)}

    peak_gflops = max((v["gflops"] for v in results.values() if "gflops" in v), default=0.0)
    return {"sizes": results, "peak_gflops": round(peak_gflops, 2)}


# ── Disk I/O benchmark ────────────────────────────────────────────────

def _bench_disk_io(shard_paths: list[Path], budget_seconds: float) -> dict[str, Any]:
    """Measure safetensors shard read throughput from local HF cache."""
    if not shard_paths:
        return {"error": "no shards provided", "read_gbs": 0.0}

    total_bytes = 0
    total_elapsed = 0.0
    file_results: list[dict[str, Any]] = []

    for fpath in shard_paths:
        if total_elapsed > budget_seconds:
            break
        if not fpath.exists():
            continue
        nbytes = fpath.stat().st_size
        t0 = time.perf_counter()
        try:
            _ = fpath.read_bytes()  # raw read into memory to measure throughput
        except Exception as exc:
            file_results.append({"file": fpath.name, "error": str(exc)})
            continue
        elapsed = time.perf_counter() - t0
        gbs = nbytes / elapsed / 1e9
        total_bytes += nbytes
        total_elapsed += elapsed
        file_results.append({
            "file": fpath.name,
            "size_bytes": nbytes,
            "elapsed_s": round(elapsed, 3),
            "read_gbs": round(gbs, 2),
        })
        gc.collect()  # release bytes object

    avg_gbs = total_bytes / total_elapsed / 1e9 if total_elapsed > 0 else 0.0
    return {
        "files_read": len(file_results),
        "total_bytes": total_bytes,
        "total_elapsed_s": round(total_elapsed, 3),
        "read_gbs": round(avg_gbs, 2),
        "file_results": file_results[:8],  # cap detail rows
    }


# ── GPU Stress Test ───────────────────────────────────────────────────

def run_gpu_stress_test(
    *,
    run_id: str,
    output_dir: Path,
    target_seconds: int = 300,
) -> dict[str, Any]:
    """Run a continuous GPU stress test at 100% utilization for target_seconds.

    This test fills GPU memory with large BF16 matrices and runs continuous
    matmul operations to push GPU utilization to maximum. Reports sustained
    TFLOPS and average GPU utilization.

    Parameters
    ----------
    run_id:
        Pipeline run identifier.
    output_dir:
        Where to write ``stress-test.json``.
    target_seconds:
        Duration to run the stress test in seconds (default 300 = 5 min).

    Returns
    -------
    dict with stress test metrics including sustained_tflops, avg_gpu_util,
    and wall_seconds.
    """
    import time

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_seconds": 0,
        "error": None,
    }

    try:
        import torch
    except ImportError:
        result["error"] = "torch unavailable"
        return result

    if not torch.cuda.is_available():
        result["error"] = "cuda not available"
        return result

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    vram_bytes = torch.cuda.get_device_properties(0).total_memory
    vram_gb = vram_bytes / 1e9

    print(f"[stress] GPU={gpu_name} VRAM={vram_gb:.0f}GB target={target_seconds}s", flush=True)

    target_mem = vram_bytes * 0.01
    dim = int((target_mem / 12) ** 0.5)
    dim = min(dim, 32768)
    dim = max(8192, (dim // 256) * 256)
    actual_gb = dim * dim * 6 * 2 / 1e9
    print(f"[stress] Allocating {dim}x{dim} bf16 matrices (~{actual_gb:.1f}GB of {vram_gb:.0f}GB)", flush=True)

    a = torch.randn(dim, dim, dtype=torch.bfloat16, device=device)
    b = torch.randn(dim, dim, dtype=torch.bfloat16, device=device)
    c = torch.zeros(dim, dim, dtype=torch.bfloat16, device=device)

    torch.cuda.synchronize()

    flops_per_iter = 2 * (dim ** 3)

    print("[stress] Warming up...", flush=True)
    for _ in range(10):
        c = torch.mm(a, b)
    torch.cuda.synchronize()

    print(f"[stress] Starting {target_seconds}s stress test...", flush=True)

    wall_start = time.perf_counter()
    iter_count = 0
    total_flops = 0.0

    gpu_utils: list[float] = []

    def get_gpu_util() -> float:
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return float(util.gpu)
        except Exception:
            return 0.0

    pynvml_available = False
    try:
        import pynvml
        pynvml.nvmlInit()
        pynvml_available = True
    except Exception:
        pass

    report_interval = 10.0
    next_report = report_interval

    while True:
        c = torch.mm(a, b)
        iter_count += 1
        total_flops += flops_per_iter

        if pynvml_available:
            gpu_utils.append(get_gpu_util())

        elapsed = time.perf_counter() - wall_start

        if elapsed >= next_report:
            sustained_tflops = total_flops / elapsed / 1e12
            avg_util = sum(gpu_utils) / len(gpu_utils) if gpu_utils else 0.0
            print(f"[stress] {elapsed:.0f}s | {iter_count} iters | "
                  f"{sustained_tflops:.1f} TFLOPS | GPU {avg_util:.0f}%", flush=True)
            next_report += report_interval

        if elapsed >= target_seconds:
            break

    torch.cuda.synchronize()
    wall_elapsed = time.perf_counter() - wall_start

    sustained_tflops = total_flops / wall_elapsed / 1e12
    avg_gpu_util = sum(gpu_utils) / len(gpu_utils) if gpu_utils else 0.0
    peak_tflops = flops_per_iter / (wall_elapsed / iter_count) / 1e12

    print(f"\n[stress] === RESULTS ===", flush=True)
    print(f"[stress] Duration:     {wall_elapsed:.1f}s", flush=True)
    print(f"[stress] Iterations:   {iter_count}", flush=True)
    print(f"[stress] Matrix size:  {dim}x{dim} bf16", flush=True)
    print(f"[stress] Peak TFLOPS:  {peak_tflops:.1f} TFLOPS (single iter)", flush=True)
    print(f"[stress] Sustained:    {sustained_tflops:.1f} TFLOPS (avg over run)", flush=True)
    if pynvml_available:
        print(f"[stress] GPU Util:     {avg_gpu_util:.1f}% (avg)", flush=True)
    print(f"[stress] ================\n", flush=True)

    result.update({
        "wall_seconds": round(wall_elapsed, 1),
        "gpu_name": gpu_name,
        "vram_gb": round(vram_gb, 1),
        "matrix_dim": dim,
        "iterations": iter_count,
        "peak_tflops": round(peak_tflops, 2),
        "sustained_tflops": round(sustained_tflops, 2),
        "avg_gpu_util_pct": round(avg_gpu_util, 1),
        "target_seconds": target_seconds,
        "gpu_util_samples": len(gpu_utils),
    })

    out_json = output_dir / "stress-test.json"
    out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[stress] Wrote {out_json}", flush=True)

    return result


# ── Main orchestrator ─────────────────────────────────────────────────

def run_benchmark(
    *,
    run_id: str,
    output_dir: Path,
    target_seconds: int = 300,
    shard_paths: list[Path] | None = None,
    skip_io: bool = False,
) -> dict[str, Any]:
    """Run the full ~5-minute benchmark suite.

    Parameters
    ----------
    run_id:
        Pipeline run identifier (used in history entries).
    output_dir:
        Where to write ``benchmark.json`` and ``figures/``.
    target_seconds:
        Approximate wall-clock target for all GPU sub-benchmarks (default 300 s).
    shard_paths:
        Safetensors shard files from the HF cache to use for the disk I/O test.
        If *None* the test is skipped.
    skip_io:
        Skip disk I/O sub-benchmark entirely.
    """
    wall_start = time.perf_counter()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device_info = _probe_device()
    device = "cuda" if device_info["gpu_available"] else "cpu"
    gpu_name = device_info["gpu_name"]
    peaks = _lookup_peaks(gpu_name)

    def _log(msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        print(f"[bench {ts}] {msg}", flush=True)

    _log(f"device={gpu_name!r}  target={target_seconds}s")

    # Time budget allocation
    if device == "cpu":
        budgets = {"gemm": 40, "bandwidth": 20, "attention": 20, "cpu": 20, "disk": 20}
    else:
        remaining = target_seconds - 30  # reserve for overhead/reporting
        budgets = {
            "gemm":      int(remaining * 0.30),  # ~90s
            "bandwidth": int(remaining * 0.22),  # ~66s
            "attention": int(remaining * 0.22),  # ~66s
            "cpu":       int(remaining * 0.10),  # ~30s
            "disk":      int(remaining * 0.16),  # ~48s
        }

    _log(f"budgets: {budgets}")

    # ── Run sub-benchmarks ──────────────────────────────────────────
    _log("gemm …")
    t0 = time.perf_counter()
    gemm = _bench_gemm(device, budgets["gemm"])
    _log(f"gemm done  ({time.perf_counter() - t0:.1f}s)  peak={gemm.get('peak_tflops',0):.1f} TFLOPS")

    _log("bandwidth …")
    t0 = time.perf_counter()
    bandwidth = _bench_bandwidth(device, budgets["bandwidth"])
    _log(f"bw done  ({time.perf_counter() - t0:.1f}s)  peak={bandwidth.get('peak_gbs',0):.0f} GB/s")

    _log("attention …")
    t0 = time.perf_counter()
    attention = _bench_attention(device, budgets["attention"])
    _log(f"attn done  ({time.perf_counter() - t0:.1f}s)  peak={attention.get('peak_tokens_per_sec',0):,} tok/s")

    _log("cpu …")
    t0 = time.perf_counter()
    cpu_bench = _bench_cpu(budgets["cpu"])
    _log(f"cpu done  ({time.perf_counter() - t0:.1f}s)  peak={cpu_bench.get('peak_gflops',0):.1f} GFLOPS")

    disk_bench: dict[str, Any] = {"skipped": True}
    if not skip_io and shard_paths:
        _log(f"disk io  ({len(shard_paths)} shards) …")
        t0 = time.perf_counter()
        disk_bench = _bench_disk_io(shard_paths, budgets["disk"])
        _log(f"disk done  ({time.perf_counter() - t0:.1f}s)  {disk_bench.get('read_gbs',0):.2f} GB/s")

    wall_elapsed = time.perf_counter() - wall_start

    result: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_seconds": round(wall_elapsed, 1),
        "device": device_info,
        "theory_peaks": peaks,
        "gemm": gemm,
        "bandwidth": bandwidth,
        "attention": attention,
        "cpu": cpu_bench,
        "disk_io": disk_bench,
    }

    # Write benchmark.json
    out_json = output_dir / "benchmark.json"
    out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _log(f"wrote {out_json}")

    # Generate per-run charts
    figs_dir = output_dir / "figures"
    figs_dir.mkdir(exist_ok=True)
    chart_paths = generate_benchmark_charts(result, figs_dir)
    result["chart_paths"] = {k: str(v) for k, v in chart_paths.items()}

    # Append to shared history and regenerate shared report
    history_path = _repo_root() / "reports" / "benchmark-history.json"
    shared_report_path = _repo_root() / "reports" / "benchmark-report.md"
    history = update_benchmark_history(result, history_path)
    report_md = render_benchmark_report(history, chart_base="../runs")
    shared_report_path.write_text(report_md, encoding="utf-8")
    _log(f"updated shared report → {shared_report_path}")

    # Also write per-run report section
    per_run_md = render_benchmark_section(result, chart_dir=Path("figures"))
    (output_dir / "benchmark-report.md").write_text(per_run_md, encoding="utf-8")

    # Print ASCII summary to terminal
    _print_ascii_summary(result, peaks)

    return result


def _repo_root() -> Path:
    """Return the repo root (parent of src/)."""
    try:
        from qwen_image_19.config_io import repo_root
        return repo_root()
    except Exception:
        # Fallback: walk up from this file
        current = Path(__file__).resolve()
        for parent in current.parents:
            if (parent / "src").exists():
                return parent
        return Path.cwd()


# ── ASCII summary (terminal) ──────────────────────────────────────────

def _print_ascii_summary(result: dict[str, Any], peaks: dict[str, float]) -> None:
    """Print a Unicode block summary table to stdout."""
    sep = "─" * 72
    gpu_name = result["device"].get("gpu_name", "unknown")
    vram_gb = result["device"].get("gpu_vram_bytes", 0) / 1e9
    cpu = result["device"].get("cpu_model", "unknown")[:50]
    ram_gb = result["device"].get("ram_bytes", 0) / 1e9
    wall = result.get("wall_seconds", 0)

    print(f"\n{'━'*72}")
    print(f"  🔬 DEVICE BENCHMARK RESULTS")
    print(f"  GPU  : {gpu_name}  ({vram_gb:.0f} GB VRAM)")
    print(f"  CPU  : {cpu}")
    print(f"  RAM  : {ram_gb:.0f} GB   ·  Run: {result.get('run_id', '?')}  ·  Wall: {wall:.0f}s")
    print(f"{'━'*72}")

    bar_w = 28

    def _row(label: str, val_str: str, ratio: float) -> None:
        bar = _ascii_bar(ratio, 1.0, bar_w)
        pct_s = f"{ratio * 100:.0f}%"
        print(f"  {label:<22}  {val_str:>12}   {bar}  {pct_s:>4}")

    print(f"\n  {'Metric':<22}  {'Value':>12}   {'Progress':>{bar_w}}  {'%':>4}")
    print(f"  {sep}")

    # GEMM
    peak_tflops = result.get("gemm", {}).get("peak_tflops", 0)
    theory_gemm = peaks.get("gemm_tflops", 100)
    _row("GEMM peak (bf16)", f"{peak_tflops:.1f} TFLOPS", peak_tflops / theory_gemm)

    # Bandwidth
    peak_bw = result.get("bandwidth", {}).get("peak_gbs", 0)
    theory_bw = peaks.get("bw_gbs", 1000)
    _row("Mem bandwidth", f"{peak_bw:.0f} GB/s", peak_bw / theory_bw)

    # Attention
    peak_tok = result.get("attention", {}).get("peak_tokens_per_sec", 0)
    _row("Attention peak", _si(peak_tok, "tok/s"), min(peak_tok / 2e6, 1.0))

    # CPU
    peak_cpu = result.get("cpu", {}).get("peak_gflops", 0)
    _row("CPU matmul", f"{peak_cpu:.0f} GFLOPS", min(peak_cpu / 1000, 1.0))

    # Disk IO
    disk_gbs = result.get("disk_io", {}).get("read_gbs", 0)
    if disk_gbs > 0:
        _row("Disk I/O", f"{disk_gbs:.2f} GB/s", min(disk_gbs / 10, 1.0))

    print(f"{'━'*72}\n")

    # Per-size GEMM table
    gemm_sizes = result.get("gemm", {}).get("sizes", {})
    if gemm_sizes:
        print("  GEMM throughput (bf16 square matmul):")
        for dim_str, v in sorted(gemm_sizes.items(), key=lambda x: int(x[0])):
            if "tflops" not in v:
                continue
            tfl = v["tflops"]
            bar = _ascii_bar(tfl, theory_gemm, 24)
            print(f"    {dim_str:>6}×{dim_str:<6}  {tfl:>7.1f} TFLOPS  {bar}")

    # Per-seqlen attention table
    attn_cfgs = result.get("attention", {}).get("configs", {})
    if attn_cfgs:
        print("\n  SDPA attention throughput:")
        max_tok = max(
            (v["tokens_per_sec"] for v in attn_cfgs.values() if "tokens_per_sec" in v),
            default=1,
        )
        for sl_str, v in sorted(attn_cfgs.items(), key=lambda x: int(x[0])):
            if "tokens_per_sec" not in v:
                continue
            tok = v["tokens_per_sec"]
            bar = _ascii_bar(tok, max_tok, 24)
            print(f"    seq={sl_str:>5}  {_si(tok, 'tok/s'):>12}  {bar}")

    print()


# ── Matplotlib chart generation ───────────────────────────────────────

def generate_benchmark_charts(
    result: dict[str, Any],
    output_dir: Path,
) -> dict[str, Path]:
    """Generate SVG charts for the benchmark result. Returns path dict."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        HAS_MPL = True
    except ImportError:
        HAS_MPL = False

    if not HAS_MPL:
        return paths

    gpu_name = result["device"].get("gpu_name", "device")
    peaks = result.get("theory_peaks", _DEFAULT_PEAK)
    _AMD_RED = "#E84B22"
    _BLUE = "#3B82F6"
    _GREEN = "#22C55E"
    _GRAY_BG = "#1E1E2E"
    _GRAY_GRID = "#3A3A5C"
    _TEXT = "#E2E8F0"

    def _style_ax(ax, title: str) -> None:
        ax.set_facecolor(_GRAY_BG)
        ax.set_title(title, color=_TEXT, fontsize=11, pad=10, fontweight="bold")
        ax.tick_params(colors=_TEXT, labelsize=8)
        ax.xaxis.label.set_color(_TEXT)
        ax.yaxis.label.set_color(_TEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor(_GRAY_GRID)
        ax.grid(color=_GRAY_GRID, linestyle="--", linewidth=0.5, alpha=0.7)

    # ── Chart 1: GEMM throughput ──────────────────────────────────
    gemm_data = result.get("gemm", {}).get("sizes", {})
    if gemm_data:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        fig.patch.set_facecolor(_GRAY_BG)
        dims = sorted([int(k) for k, v in gemm_data.items() if "tflops" in v])
        tflops = [gemm_data[str(d)]["tflops"] for d in dims]
        ax.plot(dims, tflops, color=_AMD_RED, linewidth=2.5, marker="o",
                markersize=7, markerfacecolor=_AMD_RED, zorder=3, label="Measured")
        ax.fill_between(dims, tflops, alpha=0.15, color=_AMD_RED)
        theory = peaks.get("gemm_tflops", 0)
        if theory > 0:
            ax.axhline(theory, color=_TEXT, linestyle=":", linewidth=1.2,
                       alpha=0.6, label=f"Theory peak ({theory:.0f} TFLOPS)")
        ax.set_xscale("log", base=2)
        ax.set_xticks(dims)
        ax.set_xticklabels([f"{d}×{d}" for d in dims], rotation=20, ha="right")
        ax.set_xlabel("Matrix dimension (N×N)", fontsize=9)
        ax.set_ylabel("bf16 TFLOPS", fontsize=9)
        _style_ax(ax, f"GEMM Throughput — {gpu_name}")
        ax.legend(facecolor=_GRAY_BG, edgecolor=_GRAY_GRID, labelcolor=_TEXT, fontsize=8)
        fig.tight_layout()
        p = output_dir / "gemm.svg"
        fig.savefig(p, format="svg", bbox_inches="tight", facecolor=_GRAY_BG)
        plt.close(fig)
        paths["gemm"] = p

    # ── Chart 2: Memory bandwidth ─────────────────────────────────
    bw_data = result.get("bandwidth", {}).get("sizes", {})
    if bw_data:
        valid_keys = [k for k, v in bw_data.items() if "read_gbs" in v]
        if valid_keys:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            fig.patch.set_facecolor(_GRAY_BG)
            x = range(len(valid_keys))
            reads  = [bw_data[k]["read_gbs"] for k in valid_keys]
            writes = [bw_data[k]["write_gbs"] for k in valid_keys]
            bw = 0.35
            bars_r = ax.bar([xi - bw/2 for xi in x], reads,  width=bw,
                            color=_BLUE,    alpha=0.9, label="Read",  zorder=3)
            bars_w = ax.bar([xi + bw/2 for xi in x], writes, width=bw,
                            color=_GREEN,   alpha=0.9, label="Write", zorder=3)
            for bar, val in zip(list(bars_r) + list(bars_w), reads + writes):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                        f"{val:.0f}", ha="center", va="bottom", color=_TEXT, fontsize=7)
            theory_bw = peaks.get("bw_gbs", 0)
            if theory_bw > 0:
                ax.axhline(theory_bw, color=_TEXT, linestyle=":", linewidth=1.2,
                           alpha=0.6, label=f"Theory ({theory_bw:.0f} GB/s)")
            ax.set_xticks(list(x))
            ax.set_xticklabels(valid_keys, fontsize=9)
            ax.set_xlabel("Tensor size", fontsize=9)
            ax.set_ylabel("Bandwidth (GB/s)", fontsize=9)
            _style_ax(ax, f"Memory Bandwidth — {gpu_name}")
            ax.legend(facecolor=_GRAY_BG, edgecolor=_GRAY_GRID, labelcolor=_TEXT, fontsize=8)
            fig.tight_layout()
            p = output_dir / "bandwidth.svg"
            fig.savefig(p, format="svg", bbox_inches="tight", facecolor=_GRAY_BG)
            plt.close(fig)
            paths["bandwidth"] = p

    # ── Chart 3: Attention performance ────────────────────────────
    attn_data = result.get("attention", {}).get("configs", {})
    if attn_data:
        valid_sl = sorted(
            [int(k) for k, v in attn_data.items() if "tokens_per_sec" in v]
        )
        if valid_sl:
            fig, ax1 = plt.subplots(figsize=(8, 4.5))
            fig.patch.set_facecolor(_GRAY_BG)
            toks = [attn_data[str(sl)]["tokens_per_sec"] for sl in valid_sl]
            ms   = [attn_data[str(sl)]["avg_ms"] for sl in valid_sl]
            ax1.plot(valid_sl, [t / 1e6 for t in toks], color=_AMD_RED, linewidth=2.5,
                     marker="o", markersize=7, label="Tokens/s (M)", zorder=3)
            ax1.fill_between(valid_sl, [t / 1e6 for t in toks], alpha=0.12, color=_AMD_RED)
            ax1.set_xlabel("Sequence length", fontsize=9)
            ax1.set_ylabel("Throughput (M tokens/s)", fontsize=9, color=_AMD_RED)
            ax1.tick_params(axis="y", labelcolor=_AMD_RED)
            ax2 = ax1.twinx()
            ax2.plot(valid_sl, ms, color=_BLUE, linewidth=1.8, linestyle="--",
                     marker="s", markersize=5, label="Latency (ms)", zorder=2)
            ax2.set_ylabel("Avg latency (ms)", fontsize=9, color=_BLUE)
            ax2.tick_params(axis="y", labelcolor=_BLUE, labelsize=8)
            ax2.set_facecolor(_GRAY_BG)
            _style_ax(ax1, f"SDPA Attention Performance — {gpu_name}")
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2,
                       facecolor=_GRAY_BG, edgecolor=_GRAY_GRID,
                       labelcolor=_TEXT, fontsize=8)
            fig.tight_layout()
            p = output_dir / "attention.svg"
            fig.savefig(p, format="svg", bbox_inches="tight", facecolor=_GRAY_BG)
            plt.close(fig)
            paths["attention"] = p

    # ── Chart 4: Summary dashboard (horizontal bars) ──────────────
    metrics: list[tuple[str, float, float, str]] = []
    if gemm_data:
        pk = result.get("gemm", {}).get("peak_tflops", 0)
        th = peaks.get("gemm_tflops", 100)
        metrics.append(("GEMM peak (bf16)", pk, th, f"{pk:.1f} TFLOPS"))
    if bw_data:
        pk = result.get("bandwidth", {}).get("peak_gbs", 0)
        th = peaks.get("bw_gbs", 1000)
        metrics.append(("Mem bandwidth", pk, th, f"{pk:.0f} GB/s"))
    if attn_data:
        pk = result.get("attention", {}).get("peak_tokens_per_sec", 0) / 1e6
        metrics.append(("Attention (M tok/s)", pk, max(pk, 0.1), f"{pk:.2f} M tok/s"))
    cpu_pk = result.get("cpu", {}).get("peak_gflops", 0)
    if cpu_pk:
        metrics.append(("CPU matmul", cpu_pk, max(cpu_pk, 1), f"{cpu_pk:.0f} GFLOPS"))
    disk_gbs = result.get("disk_io", {}).get("read_gbs", 0)
    if disk_gbs:
        metrics.append(("Disk I/O", disk_gbs, max(disk_gbs, 0.1), f"{disk_gbs:.2f} GB/s"))

    if metrics:
        fig, ax = plt.subplots(figsize=(9, max(3, len(metrics) * 0.85 + 1.2)))
        fig.patch.set_facecolor(_GRAY_BG)
        labels  = [m[0] for m in metrics]
        ratios  = [min(m[1] / m[2], 1.0) for m in metrics]
        val_strs = [m[3] for m in metrics]
        y = range(len(metrics))
        bar_colors = [
            _GREEN if r > 0.80 else _BLUE if r > 0.55 else _AMD_RED if r > 0.30 else "#6B7280"
            for r in ratios
        ]
        hbars = ax.barh(list(y), ratios, color=bar_colors, height=0.55, zorder=3)
        ax.barh(list(y), [1.0] * len(y), color="#2A2A3E", height=0.55, zorder=2)
        for i, (bar, val_str, ratio) in enumerate(zip(hbars, val_strs, ratios)):
            ax.text(min(ratio, 0.98) + 0.01, i, f"{val_str}  ({ratio*100:.0f}% of theory)",
                    va="center", color=_TEXT, fontsize=8)
        ax.set_yticks(list(y))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlim(0, 1.5)
        ax.set_xlabel("% of theoretical peak", fontsize=9)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
        _style_ax(ax, f"Performance Dashboard — {gpu_name}")
        # legend patches
        legend_items = [
            mpatches.Patch(color=_GREEN,   label=">80% — excellent"),
            mpatches.Patch(color=_BLUE,    label="55–80% — good"),
            mpatches.Patch(color=_AMD_RED, label="30–55% — fair"),
            mpatches.Patch(color="#6B7280",label="< 30%  — limited"),
        ]
        ax.legend(handles=legend_items, facecolor=_GRAY_BG, edgecolor=_GRAY_GRID,
                  labelcolor=_TEXT, fontsize=7, loc="lower right")
        fig.tight_layout()
        p = output_dir / "summary.svg"
        fig.savefig(p, format="svg", bbox_inches="tight", facecolor=_GRAY_BG)
        plt.close(fig)
        paths["summary"] = p

    return paths


# ── Per-run Markdown section ──────────────────────────────────────────

def render_benchmark_section(
    result: dict[str, Any],
    chart_dir: Path | None = None,
) -> str:
    """Render a Markdown section for one benchmark run (device+timestamp)."""
    dev = result["device"]
    gpu_name = dev.get("gpu_name", "CPU only")
    cpu_model = dev.get("cpu_model", "unknown")
    vram_gb = dev.get("gpu_vram_bytes", 0) / 1e9
    ram_gb = dev.get("ram_bytes", 0) / 1e9
    run_id = result.get("run_id", "?")
    ts = result.get("timestamp", "?")
    wall = result.get("wall_seconds", 0)
    peaks = result.get("theory_peaks", _DEFAULT_PEAK)
    theory_label = peaks.get("label", "unknown")

    def _img(name: str) -> str:
        if chart_dir is None:
            return ""
        p = chart_dir / f"{name}.svg"
        return f"\n![{name}]({p})\n"

    bar_w = 30

    # Summary table rows
    def _row(label: str, value: float, theory: float, val_str: str) -> str:
        if theory <= 0:
            return ""
        ratio = min(value / theory, 1.0)
        bar = _ascii_bar(ratio, 1.0, 24)
        pct = f"{ratio * 100:.0f}%"
        return f"| {label} | {val_str} | `{bar}` {pct} |\n"

    summary_rows = ""
    pk_gemm = result.get("gemm", {}).get("peak_tflops", 0)
    if pk_gemm:
        summary_rows += _row("GEMM peak (bf16)", pk_gemm,
                             peaks.get("gemm_tflops", 100), f"{pk_gemm:.1f} TFLOPS")

    pk_bw = result.get("bandwidth", {}).get("peak_gbs", 0)
    if pk_bw:
        summary_rows += _row("Memory bandwidth", pk_bw,
                             peaks.get("bw_gbs", 1000), f"{pk_bw:.0f} GB/s")

    pk_tok = result.get("attention", {}).get("peak_tokens_per_sec", 0)
    if pk_tok:
        summary_rows += _row("Attention throughput", float(pk_tok), 2e6,
                             _si(pk_tok, "tok/s"))

    pk_cpu = result.get("cpu", {}).get("peak_gflops", 0)
    if pk_cpu:
        summary_rows += _row("CPU matmul (fp32)", pk_cpu, max(pk_cpu, 1),
                             f"{pk_cpu:.0f} GFLOPS")

    disk_gbs = result.get("disk_io", {}).get("read_gbs", 0)
    if disk_gbs:
        summary_rows += _row("Disk I/O (shard read)", disk_gbs, 10.0,
                             f"{disk_gbs:.2f} GB/s")

    # GEMM per-size table
    gemm_sizes = result.get("gemm", {}).get("sizes", {})
    gemm_rows = ""
    theory_gemm = peaks.get("gemm_tflops", 100)
    for dim_str in sorted(gemm_sizes, key=int):
        v = gemm_sizes[dim_str]
        if "tflops" not in v:
            continue
        tfl = v["tflops"]
        ms = v.get("avg_ms", 0)
        bar = _ascii_bar(tfl, theory_gemm, 20)
        gemm_rows += f"| {dim_str}×{dim_str} | {ms:.1f} | {tfl:.1f} | `{bar}` |\n"

    # BW per-size table
    bw_sizes = result.get("bandwidth", {}).get("sizes", {})
    bw_rows = ""
    for label, v in bw_sizes.items():
        if "read_gbs" not in v:
            continue
        r = v["read_gbs"]
        w = v.get("write_gbs", 0)
        theory_bw = peaks.get("bw_gbs", 1000)
        bar = _ascii_bar((r + w) / 2, theory_bw, 20)
        bw_rows += f"| {label} | {r:.0f} | {w:.0f} | `{bar}` |\n"

    # Attention per-seqlen table
    attn_cfgs = result.get("attention", {}).get("configs", {})
    attn_rows = ""
    max_tok = max(
        (v.get("tokens_per_sec", 0) for v in attn_cfgs.values()), default=1
    )
    for sl_str in sorted(attn_cfgs, key=int):
        v = attn_cfgs[sl_str]
        if "tokens_per_sec" not in v:
            continue
        tok = v["tokens_per_sec"]
        ms = v.get("avg_ms", 0)
        tfl = v.get("attn_tflops", 0)
        bar = _ascii_bar(tok, max_tok, 20)
        attn_rows += f"| {sl_str} | {ms:.1f} ms | {_si(tok, 'tok/s')} | {tfl:.2f} | `{bar}` |\n"

    # Disk IO lines
    files_read = result.get("disk_io", {}).get("files_read", 0)
    total_gb = result.get("disk_io", {}).get("total_bytes", 0) / 1e9
    disk_summary = ""
    if disk_gbs > 0:
        disk_summary = (
            f"\n**Disk I/O:** {files_read} shards · "
            f"{total_gb:.1f} GB total · **{disk_gbs:.2f} GB/s** avg read\n"
        )

    section = f"""
---

## 🖥️ {gpu_name} &nbsp;·&nbsp; run `{run_id}` &nbsp;·&nbsp; {ts[:10]}

> **Host:** `{dev.get('hostname', '?')}`  ·  **OS:** `{dev.get('os', '?')}`
> **GPU:** {gpu_name}  ({vram_gb:.0f} GB VRAM)  ·  **CPU:** {cpu_model}  ·  **RAM:** {ram_gb:.0f} GB
> **PyTorch:** `{dev.get('torch_version', '?')}`  ·  **Duration:** {wall:.0f}s
> **Reference device:** {theory_label}

### Performance Summary

| Metric | Value | vs Theory |
|--------|-------|-----------|
{summary_rows}

{_img("summary")}

### GEMM Throughput (bf16 square matmul)

{_img("gemm")}

| Matrix (M=K=N) | Latency (ms) | TFLOPS | Utilisation |
|----------------|-------------|--------|-------------|
{gemm_rows}

### Memory Bandwidth

{_img("bandwidth")}

| Tensor size | Read (GB/s) | Write (GB/s) | vs Theory |
|-------------|------------|-------------|-----------|
{bw_rows}

### SDPA Attention Performance

> Config: batch={result.get('attention', {}).get('configs', {}).get(next(iter(attn_cfgs), '512'), {}).get('batch', 4)}, \
heads={result.get('attention', {}).get('configs', {}).get(next(iter(attn_cfgs), '512'), {}).get('heads', 32)}, \
head_dim={result.get('attention', {}).get('configs', {}).get(next(iter(attn_cfgs), '512'), {}).get('head_dim', 128)}

{_img("attention")}

| Seq len | Latency | Tokens/s | TFLOPS | Utilisation |
|---------|---------|----------|--------|-------------|
{attn_rows}

### CPU Baseline
{disk_summary}
**CPU matmul peak:** {result.get('cpu', {}).get('peak_gflops', 0):.0f} GFLOPS \
({dev.get('cpu_model', '?')}, {dev.get('cpu_cores', 1)} cores)
"""
    return section.strip()


# ── History append + shared report ───────────────────────────────────

def update_benchmark_history(
    result: dict[str, Any],
    history_path: Path,
) -> list[dict[str, Any]]:
    """Load existing history, append/update this run entry, and write back."""
    history_path = Path(history_path)
    history_path.parent.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, Any]] = []
    if history_path.exists():
        try:
            history = json.loads(history_path.read_text(encoding="utf-8"))
        except Exception:
            history = []

    # Build a compact snapshot (no per-file detail) for history
    entry = {
        "run_id": result.get("run_id"),
        "timestamp": result.get("timestamp"),
        "wall_seconds": result.get("wall_seconds"),
        "device": result.get("device"),
        "theory_peaks": result.get("theory_peaks"),
        "gemm_peak_tflops": result.get("gemm", {}).get("peak_tflops", 0),
        "bw_peak_gbs": result.get("bandwidth", {}).get("peak_gbs", 0),
        "attn_peak_tokens_per_sec": result.get("attention", {}).get("peak_tokens_per_sec", 0),
        "cpu_peak_gflops": result.get("cpu", {}).get("peak_gflops", 0),
        "disk_read_gbs": result.get("disk_io", {}).get("read_gbs", 0),
        "gemm_sizes": result.get("gemm", {}).get("sizes", {}),
        "bw_sizes": result.get("bandwidth", {}).get("sizes", {}),
        "attn_configs": result.get("attention", {}).get("configs", {}),
    }

    # Replace existing entry for same run_id if present
    history = [e for e in history if e.get("run_id") != result.get("run_id")]
    history.append(entry)
    # Keep only latest 50 entries
    history = history[-50:]

    history_path.write_text(json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return history


def render_benchmark_report(
    history: list[dict[str, Any]],
    chart_base: str = "runs",
) -> str:
    """Render the shared benchmark-report.md from all history entries."""
    if not history:
        return "# Device Benchmark Report\n\n*No benchmark results yet.*\n"

    # Unique devices by gpu_name for comparison table
    device_labels = []
    seen: set[str] = set()
    for e in reversed(history):
        gpu = e.get("device", {}).get("gpu_name", "CPU only")
        label = f"{gpu} (run `{e.get('run_id', '?')}`)"
        if label not in seen:
            seen.add(label)
            device_labels.append((e, label))

    # Cross-device comparison table (all metrics, all runs)
    cmp_header = "| Metric |"
    cmp_sep    = "| --- |"
    cmp_gemm   = "| GEMM peak (TFLOPS) |"
    cmp_bw     = "| Mem BW peak (GB/s) |"
    cmp_attn   = "| Attn peak (tok/s) |"
    cmp_cpu    = "| CPU peak (GFLOPS) |"
    cmp_disk   = "| Disk I/O (GB/s) |"
    cmp_wall   = "| Benchmark time (s) |"

    for e, label in device_labels:
        cmp_header += f" {label} |"
        cmp_sep    += " --- |"
        cmp_gemm   += f" {e.get('gemm_peak_tflops', 0):.1f} |"
        cmp_bw     += f" {e.get('bw_peak_gbs', 0):.0f} |"
        tok = e.get("attn_peak_tokens_per_sec", 0)
        cmp_attn   += f" {_si(tok, '')} |"
        cmp_cpu    += f" {e.get('cpu_peak_gflops', 0):.0f} |"
        cmp_disk   += f" {e.get('disk_read_gbs', 0):.2f} |"
        cmp_wall   += f" {e.get('wall_seconds', 0):.0f} |"

    comparison_table = "\n".join([cmp_header, cmp_sep, cmp_gemm, cmp_bw,
                                  cmp_attn, cmp_cpu, cmp_disk, cmp_wall])

    # Per-run sections
    sections: list[str] = []
    for e in reversed(history):
        rid = e.get("run_id", "?")
        chart_dir_str = f"{chart_base}/{rid}/preflight/figures"
        # Reconstruct a minimal result-like dict for rendering
        mini_result = {
            "run_id": rid,
            "timestamp": e.get("timestamp", "?"),
            "wall_seconds": e.get("wall_seconds", 0),
            "device": e.get("device", {}),
            "theory_peaks": e.get("theory_peaks", _DEFAULT_PEAK),
            "gemm": {"peak_tflops": e.get("gemm_peak_tflops", 0), "sizes": e.get("gemm_sizes", {})},
            "bandwidth": {"peak_gbs": e.get("bw_peak_gbs", 0), "sizes": e.get("bw_sizes", {})},
            "attention": {"peak_tokens_per_sec": e.get("attn_peak_tokens_per_sec", 0),
                          "configs": e.get("attn_configs", {})},
            "cpu": {"peak_gflops": e.get("cpu_peak_gflops", 0)},
            "disk_io": {"read_gbs": e.get("disk_read_gbs", 0)},
        }
        sections.append(render_benchmark_section(mini_result, chart_dir=Path(chart_dir_str)))

    return f"""# Device Benchmark Report

> Auto-generated by `qwen_image_19.stage_1_analysis.benchmark`.
> Each run appends a new section for the device that executed it.
> Benchmark target: **~5 minutes** per device.

## Cross-Device Comparison

{comparison_table}

{''.join(sections)}
"""


# ── CLI entry point ───────────────────────────────────────────────────

def _main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Qwen-Image 1.9 device benchmark")
    ap.add_argument("--run-id", default=f"bench-{time.strftime('%Y%m%d-%H%M%S')}")
    ap.add_argument("--output-dir", type=Path,
                    default=Path("reports/runs/bench/preflight"))
    ap.add_argument("--target-seconds", type=int, default=300)
    ap.add_argument("--skip-io", action="store_true")
    ap.add_argument("--shard", action="append", type=Path, dest="shards",
                    help="Safetensors shard path(s) for disk I/O benchmark")
    args = ap.parse_args()
    run_benchmark(
        run_id=args.run_id,
        output_dir=args.output_dir,
        target_seconds=args.target_seconds,
        shard_paths=args.shards,
        skip_io=args.skip_io,
    )


if __name__ == "__main__":
    _main()
