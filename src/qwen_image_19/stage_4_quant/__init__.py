"""Stage 4 — Quantization driver.

Produces compressed deployment artifacts from the abliterated checkpoint:

  - GGUF Q4_K_M, IQ4_XS (via llama.cpp — for ollama/llama.cpp)
  - GPTQ 4-bit marlin (via auto-gptq — PRIMARY vllm-omni format)
  - EXL2 4.0bpw (via exllamav2 — for local exllamav2/TabbyAPI serving)

vllm-omni serving requires the GPTQ format with marlin kernel.
Load with: ``--quantization gptq_marlin`` in vllm.

All heavy compute runs on the MI300X (ROCm). For dry-run the plan is
returned without writing files or invoking tools.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from qwen_image_19.config_io import repo_root, write_json
from qwen_image_19.contracts import public_path, utc_now


class QuantError(RuntimeError):
    """Raised when quantization planning or execution fails."""


# ── Config ───────────────────────────────────────────────────────────

def _load_gguf_config() -> dict[str, Any]:
    p = repo_root() / "configs" / "quant" / "stage-4-gguf-imatrix.yaml"
    if p.exists():
        try:
            import yaml
            return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except Exception:
            pass
    return {"targets": ["Q4_K_M", "IQ4_XS"]}


def _load_exl2_gptq_config() -> dict[str, Any]:
    p = repo_root() / "configs" / "quant" / "stage-4-exl2-gptq.yaml"
    if p.exists():
        try:
            import yaml
            return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except Exception:
            pass
    return {"gptq": {"targets": ["4bit"]}, "exl2": {"targets": ["4.0bpw"]}}


# ── llama.cpp helpers ─────────────────────────────────────────────────

def _find_llama_cpp(search_dirs: list[Path] | None = None) -> Path | None:
    dirs = search_dirs or [
        repo_root() / "tools" / "llama.cpp",
        Path("/opt/llama.cpp"),
        Path.home() / "llama.cpp",
    ]
    for d in dirs:
        convert = d / "convert_hf_to_gguf.py"
        if convert.exists():
            return d
    # Try PATH
    if shutil.which("llama-quantize") or shutil.which("llama_quantize"):
        return Path(".")
    return None


def _find_exllamav2() -> bool:
    try:
        import exllamav2  # noqa: F401
        return True
    except ImportError:
        return bool(shutil.which("convert_hf_to_exl2.py") or shutil.which("exllamav2-convert"))


# ── Planning ──────────────────────────────────────────────────────────

def plan_quantize(
    *,
    input_checkpoint: str,
    run_dir: Path,
    quant_method: str = "all",
    remote_config: str | None = None,
) -> dict[str, Any]:
    """Build a quantization plan without executing anything.

    *quant_method* selects which formats to produce:
      ``"all"``  — GGUF + GPTQ (vllm-omni) + EXL2
      ``"gptq"`` — GPTQ only (primary vllm-omni format)
      ``"gguf"`` — GGUF only (llama.cpp / ollama)
      ``"exl2"`` — EXL2 only (exllamav2 / TabbyAPI)
    """
    from qwen_image_19.remote import default_remote_context
    ctx = default_remote_context(remote_config)
    gguf_config = _load_gguf_config()
    exl2_gptq_config = _load_exl2_gptq_config()
    gptq_config = exl2_gptq_config.get("gptq", {})
    exl2_config = exl2_gptq_config.get("exl2", {})

    quant_dir = run_dir / "quantize"
    gguf_dir = quant_dir / "gguf"
    gptq_dir = quant_dir / "gptq"
    exl2_dir = quant_dir / "exl2"

    targets: list[dict[str, Any]] = []

    if quant_method in ("all", "gguf"):
        for fmt in gguf_config.get("targets", ["Q4_K_M"]):
            targets.append({
                "format": "GGUF",
                "quantization": fmt,
                "output_path": str(gguf_dir / f"model-{fmt.lower()}.gguf"),
                "tool": "llama.cpp",
                "runtime": "llama.cpp/ollama",
            })

    if quant_method in ("all", "gptq"):
        for bits in gptq_config.get("targets", ["4bit"]):
            targets.append({
                "format": "GPTQ",
                "quantization": bits,
                "group_size": gptq_config.get("group_size", 128),
                "desc_act": gptq_config.get("desc_act", False),
                "sym": gptq_config.get("sym", True),
                "damp_percent": gptq_config.get("damp_percent", 0.01),
                "output_path": str(gptq_dir / bits),
                "tool": "auto-gptq",
                "runtime": "vllm-omni (--quantization gptq_marlin)",
                "vllm_quantization": "gptq_marlin",
            })

    if quant_method in ("all", "exl2"):
        for bpw in exl2_config.get("targets", ["4.0bpw"]):
            targets.append({
                "format": "EXL2",
                "quantization": bpw,
                "output_path": str(exl2_dir / bpw),
                "tool": "exllamav2",
                "runtime": "exllamav2/TabbyAPI (not vllm-omni)",
            })

    declared_outputs = [
        f"{ctx['artifact_dir']}/runs/{run_dir.name}/quantize/{t['format'].lower()}/"
        + Path(t["output_path"]).name
        for t in targets
    ]

    return {
        "input_checkpoint": input_checkpoint,
        "quant_dir": str(quant_dir),
        "targets": targets,
        "declared_outputs": declared_outputs,
        "log_path": str(quant_dir / "quantize.log"),
        "gguf_config": gguf_config,
        "gptq_config": gptq_config,
        "exl2_config": exl2_config,
        "remote_job": {
            "name": "quantize",
            "workdir": ctx["workdir"],
            "artifact_dir": ctx["artifact_dir"],
            "status": "planned",
        },
        "metrics": {
            "quant_method": quant_method,
            "target_count": len(targets),
            "vllm_omni_format": "GPTQ (gptq_marlin)",
        },
    }


# ── GGUF execution ────────────────────────────────────────────────────

def _run_gguf(
    *,
    input_path: Path,
    gguf_dir: Path,
    targets: list[dict[str, Any]],
    imatrix_dataset: Path | None,
    log,
) -> list[dict[str, Any]]:
    """Convert to FP16 GGUF then quantize to each target format."""
    import sys

    llama_dir = _find_llama_cpp()
    if llama_dir is None:
        log.write("[quant/gguf] llama.cpp not found — cloning from GitHub …\n")
        clone_dir = repo_root() / "tools" / "llama.cpp"
        subprocess.run(
            ["git", "clone", "--depth=1",
             "https://github.com/ggerganov/llama.cpp.git",
             str(clone_dir)],
            check=True,
        )
        llama_dir = clone_dir

    gguf_dir.mkdir(parents=True, exist_ok=True)
    fp16_path = gguf_dir / "model-f16.gguf"

    # Step 1: convert HF checkpoint → FP16 GGUF
    convert_script = llama_dir / "convert_hf_to_gguf.py"
    log.write(f"[quant/gguf] converting {input_path} → {fp16_path}\n")
    log.flush()
    result = subprocess.run(
        [sys.executable, str(convert_script),
         str(input_path),
         "--outfile", str(fp16_path),
         "--outtype", "f16"],
        capture_output=True, text=True,
    )
    log.write(result.stdout + result.stderr)
    log.flush()
    if result.returncode != 0:
        raise QuantError(f"GGUF conversion failed (exit {result.returncode}). See log.")

    # Step 2: optional imatrix
    imatrix_path: Path | None = None
    if imatrix_dataset and imatrix_dataset.exists():
        imatrix_path = gguf_dir / "imatrix.dat"
        imatrix_bin = llama_dir / "llama-imatrix"
        if imatrix_bin.exists():
            log.write(f"[quant/gguf] generating imatrix from {imatrix_dataset}\n")
            subprocess.run(
                [str(imatrix_bin), "-m", str(fp16_path),
                 "-f", str(imatrix_dataset),
                 "-o", str(imatrix_path)],
                check=False, capture_output=True, text=True,
            )

    # Step 3: quantize each target
    quant_bin = llama_dir / "llama-quantize"
    if not quant_bin.exists():
        quant_bin = llama_dir / "build" / "bin" / "llama-quantize"
    if not quant_bin.exists():
        raise QuantError("llama-quantize binary not found. Build llama.cpp first.")

    artifacts: list[dict[str, Any]] = []
    for target in targets:
        if target.get("format") != "GGUF":
            continue
        fmt = target["quantization"]
        out_path = Path(target["output_path"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [str(quant_bin), str(fp16_path), str(out_path), fmt]
        if imatrix_path and imatrix_path.exists():
            cmd = [str(quant_bin), "--imatrix", str(imatrix_path),
                   str(fp16_path), str(out_path), fmt]
        log.write(f"[quant/gguf] {fmt}: {out_path.name}\n")
        log.flush()
        r = subprocess.run(cmd, capture_output=True, text=True)
        log.write(r.stdout + r.stderr)
        log.flush()
        size = out_path.stat().st_size if out_path.exists() else 0
        artifacts.append({
            "format": "GGUF",
            "quantization": fmt,
            "path": str(out_path),
            "size_bytes": size,
            "status": "succeeded" if r.returncode == 0 else "failed",
        })
    return artifacts


def _run_gptq(
    *,
    input_path: Path,
    gptq_dir: Path,
    targets: list[dict[str, Any]],
    log,
) -> list[dict[str, Any]]:
    """Quantize to GPTQ format using auto-gptq.

    The output directory layout is compatible with vllm-omni:
      ``--model <gptq_dir>/<bits>  --quantization gptq_marlin``
    """
    artifacts: list[dict[str, Any]] = []
    for target in targets:
        if target.get("format") != "GPTQ":
            continue
        bits_str = target["quantization"]
        bits = int(bits_str.replace("bit", "").strip())
        group_size = int(target.get("group_size", 128))
        desc_act = bool(target.get("desc_act", False))
        sym = bool(target.get("sym", True))
        damp_percent = float(target.get("damp_percent", 0.01))
        out_path = Path(target["output_path"])
        out_path.mkdir(parents=True, exist_ok=True)
        log.write(f"[quant/gptq] {bits}bit group_size={group_size} → {out_path}\n")
        log.flush()
        try:
            from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig  # type: ignore
            from transformers import AutoTokenizer  # type: ignore

            quantize_config = BaseQuantizeConfig(
                bits=bits,
                group_size=group_size,
                desc_act=desc_act,
                sym=sym,
                damp_percent=damp_percent,
            )
            log.write(f"[quant/gptq] loading tokenizer from {input_path}\n")
            tokenizer = AutoTokenizer.from_pretrained(str(input_path), trust_remote_code=True)
            log.write(f"[quant/gptq] loading model for GPTQ calibration …\n")
            model = AutoGPTQForCausalLM.from_pretrained(
                str(input_path),
                quantize_config=quantize_config,
            )
            # Minimal calibration dataset — GPTQ calibrates on text
            calibration_data = [
                tokenizer(text, return_tensors="pt")
                for text in [
                    "A high quality photograph of a scenic landscape.",
                    "Edit this image: change the sky to a dramatic sunset.",
                    "Generate an image with transparent background layers.",
                    "A professional product photograph with soft lighting.",
                    "A detailed illustration of a fantasy castle at night.",
                ]
            ]
            log.write("[quant/gptq] running GPTQ calibration …\n")
            log.flush()
            model.quantize(calibration_data)
            model.save_quantized(str(out_path), use_safetensors=True)
            tokenizer.save_pretrained(str(out_path))
            log.write(f"[quant/gptq] saved to {out_path}\n")
            artifacts.append({
                "format": "GPTQ",
                "quantization": bits_str,
                "path": str(out_path),
                "size_bytes": sum(f.stat().st_size for f in out_path.rglob("*.safetensors")),
                "status": "succeeded",
                "vllm_load_args": f"--quantization gptq_marlin",
            })
        except ImportError as e:
            log.write(f"[quant/gptq] auto-gptq not available: {e}\n"
                      "[quant/gptq] Install with: pip install auto-gptq --extra-index-url "
                      "https://huggingface.github.io/autogptq-index/whl/rocm573/\n")
            log.flush()
            artifacts.append({
                "format": "GPTQ",
                "quantization": bits_str,
                "path": str(out_path),
                "size_bytes": 0,
                "status": "failed",
                "error": f"auto-gptq not installed: {e}",
            })
        except Exception as e:
            log.write(f"[quant/gptq] failed: {e}\n")
            log.flush()
            artifacts.append({
                "format": "GPTQ",
                "quantization": bits_str,
                "path": str(out_path),
                "size_bytes": 0,
                "status": "failed",
                "error": str(e),
            })
    return artifacts



    *,
    input_path: Path,
    exl2_dir: Path,
    targets: list[dict[str, Any]],
    log,
) -> list[dict[str, Any]]:
    """Convert to EXL2 format using exllamav2."""
    import sys

    artifacts: list[dict[str, Any]] = []
    for target in targets:
        if target.get("format") != "EXL2":
            continue
        bpw = target["quantization"].replace("bpw", "")
        out_path = Path(target["output_path"])
        out_path.mkdir(parents=True, exist_ok=True)
        log.write(f"[quant/exl2] bpw={bpw} → {out_path}\n")
        log.flush()
        cmd = [
            sys.executable, "-m", "exllamav2.convert",
            "--input_dir", str(input_path),
            "--output_dir", str(out_path),
            "--bits", bpw,
            "--fasttensors",
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        log.write(r.stdout + r.stderr)
        log.flush()
        artifacts.append({
            "format": "EXL2",
            "quantization": f"{bpw}bpw",
            "path": str(out_path),
            "size_bytes": 0,
            "status": "succeeded" if r.returncode == 0 else "failed",
        })
    return artifacts


# ── Main executor ─────────────────────────────────────────────────────

def execute_quantize(plan: dict[str, Any]) -> dict[str, Any]:
    """Execute all quantization targets defined in the plan."""
    input_path = Path(plan["input_checkpoint"])
    if not input_path.exists():
        raise QuantError(
            f"Input checkpoint `{input_path}` not found. "
            "Run merge and abliterate first."
        )

    quant_dir = Path(plan["quant_dir"])
    quant_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(plan["log_path"])
    started_at = utc_now()
    t0 = time.perf_counter()

    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"[quant] started={started_at}\n")
        log.write(f"[quant] input={input_path}\n")
        log.flush()

        # For diffusers-style componentized checkpoints (model_index.json),
        # GGUF conversion targets the text_encoder component (LLM).
        # The MMDiT transformer and VAE are not LLM-compatible with llama.cpp.
        gguf_input_path = input_path
        if (input_path / "model_index.json").exists():
            te_dir = input_path / "text_encoder"
            if te_dir.is_dir() and (te_dir / "config.json").exists():
                gguf_input_path = te_dir
                log.write(f"[quant] diffusers layout detected — GGUF will target text_encoder at {te_dir}\n")
            else:
                log.write("[quant] diffusers layout but no text_encoder/config.json — GGUF skipped\n")
                gguf_input_path = None  # type: ignore[assignment]
        log.flush()

        gguf_targets = [t for t in plan["targets"] if t.get("format") == "GGUF"]
        gptq_targets = [t for t in plan["targets"] if t.get("format") == "GPTQ"]
        exl2_targets = [t for t in plan["targets"] if t.get("format") == "EXL2"]

        all_artifacts: list[dict[str, Any]] = []

        if gguf_targets and gguf_input_path is not None:
            gguf_dir = quant_dir / "gguf"
            all_artifacts.extend(
                _run_gguf(
                    input_path=gguf_input_path,
                    gguf_dir=gguf_dir,
                    targets=gguf_targets,
                    imatrix_dataset=None,
                    log=log,
                )
            )
        elif gguf_targets:
            log.write("[quant] GGUF targets skipped (no LLM component detected)\n")

        if gptq_targets:
            # GPTQ targets the text_encoder for diffusers-style checkpoints,
            # same as GGUF. Falls back to full checkpoint for non-diffusers layouts.
            gptq_input = gguf_input_path if gguf_input_path is not None else input_path
            gptq_dir = quant_dir / "gptq"
            all_artifacts.extend(
                _run_gptq(
                    input_path=gptq_input,
                    gptq_dir=gptq_dir,
                    targets=gptq_targets,
                    log=log,
                )
            )

        if exl2_targets:
            exl2_dir = quant_dir / "exl2"
            all_artifacts.extend(
                _run_exl2(
                    input_path=input_path,
                    exl2_dir=exl2_dir,
                    targets=exl2_targets,
                    log=log,
                )
            )

        log.write(f"[quant] done: {len(all_artifacts)} artifacts\n")

    duration = time.perf_counter() - t0
    succeeded = [a for a in all_artifacts if a.get("status") == "succeeded"]
    return {
        "status": "succeeded" if len(succeeded) == len(all_artifacts) else "partial",
        "started_at": started_at,
        "ended_at": utc_now(),
        "duration_seconds": round(duration, 2),
        "artifacts": all_artifacts,
        "succeeded_count": len(succeeded),
        "total_count": len(all_artifacts),
        "log_path": str(log_path),
    }


def quantize(
    *,
    input_checkpoint: str,
    run_dir: Path,
    quant_method: str = "all",
    remote_config: str | None = None,
) -> dict[str, Any]:
    """High-level wrapper: plan and execute quantization."""
    plan = plan_quantize(
        input_checkpoint=input_checkpoint,
        run_dir=run_dir,
        quant_method=quant_method,
        remote_config=remote_config,
    )
    return execute_quantize(plan)
