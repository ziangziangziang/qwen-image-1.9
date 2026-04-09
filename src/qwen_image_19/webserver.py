from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from qwen_image_19.config_io import repo_root
from qwen_image_19.reporting import collect_runs

MIME_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript",
    ".css": "text/css",
    ".json": "application/json",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
    ".webp": "image/webp",
    ".woff": "font/woff",
    ".woff2": "font/woff2",
    ".ttf": "font/ttf",
    ".txt": "text/plain; charset=utf-8",
    ".md": "text/markdown",
}

STATIC_DIR = Path(__file__).resolve().parent.parent.parent.parent / "web" / "dist"


def _json_bytes(payload: dict[str, Any] | list[dict[str, Any]]) -> bytes:
    return json.dumps(payload, indent=2).encode("utf-8")


def _read_json_file(path: Path) -> dict[str, Any] | list[dict[str, Any]] | None:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def resolve_api_request(path: str, runs_root: Path) -> tuple[int, dict[str, Any] | list[dict[str, Any]] | str | bytes, str] | None:
    normalized = unquote(path.rstrip("/"))
    if normalized in ("", "/"):
        return 200, "Qwen Image 1.9 results server\n", "text/plain; charset=utf-8"
    if normalized == "/api/runs":
        runs = collect_runs(runs_root)
        payload = [
            {
                "run_id": run["run_id"],
                "updated_at": run["updated_at"],
                "report_index": run["report_index"],
                "steps": run["steps"],
            }
            for run in runs
        ]
        return 200, payload, "application/json"
    parts = normalized.split("/")
    if len(parts) == 4 and parts[:3] == ["", "api", "runs"]:
        run_id = parts[3]
        manifest_path = runs_root / run_id / "manifest.json"
        if manifest_path.exists():
            return 200, json.loads(manifest_path.read_text(encoding="utf-8")), "application/json"
    if len(parts) == 6 and parts[:3] == ["", "api", "runs"] and parts[4] == "steps":
        run_id = parts[3]
        step = parts[5]
        result_path = runs_root / run_id / step / "step-result.json"
        if result_path.exists():
            return 200, json.loads(result_path.read_text(encoding="utf-8")), "application/json"
    if len(parts) == 7 and parts[:3] == ["", "api", "runs"] and parts[4] == "steps" and parts[6] == "samples":
        run_id = parts[3]
        step = parts[5]
        sample_dir = runs_root / run_id / step / "samples"
        payload = {
            "run_id": run_id,
            "step": step,
            "samples": sorted(item.name for item in sample_dir.glob("*")) if sample_dir.exists() else [],
        }
        return 200, payload, "application/json"
    if len(parts) == 7 and parts[:3] == ["", "api", "runs"] and parts[4] == "steps" and parts[6] == "eval-summary":
        run_id = parts[3]
        step = parts[5]
        eval_path = runs_root / run_id / step / "eval-summary.json"
        data = _read_json_file(eval_path)
        if data is not None:
            return 200, data, "application/json"
        return 404, "Eval summary not found\n", "text/plain; charset=utf-8"
    if len(parts) == 7 and parts[:3] == ["", "api", "runs"] and parts[4] == "steps" and parts[6] == "metrics":
        run_id = parts[3]
        step = parts[5]
        step_result_path = runs_root / run_id / step / "step-result.json"
        step_result = _read_json_file(step_result_path)
        if step_result is None:
            return 404, "Step result not found\n", "text/plain; charset=utf-8"
        metrics_files: list[dict[str, Any]] = []
        artifact_dir = None
        extra = step_result.get("extra", {})
        if "legacy_stage_result" in extra:
            artifact_dir = extra["legacy_stage_result"].get("artifact_dir")
        if artifact_dir:
            artifacts_base = Path(artifact_dir)
            if not artifacts_base.is_absolute():
                artifacts_base = repo_root() / artifacts_base
            metrics_dir = artifacts_base / "metrics"
            if metrics_dir.exists():
                for mf in sorted(metrics_dir.glob("*.json")):
                    try:
                        metrics_files.append({"name": mf.stem, **json.loads(mf.read_text(encoding="utf-8"))})
                    except (json.JSONDecodeError, OSError):
                        continue
        payload = {
            "run_id": run_id,
            "step": step,
            "metrics": metrics_files,
        }
        return 200, payload, "application/json"
    if len(parts) == 7 and parts[:3] == ["", "api", "runs"] and parts[4] == "steps" and parts[6] == "training-config":
        run_id = parts[3]
        step = parts[5]
        result_path = runs_root / run_id / step / "step-result.json"
        step_result = _read_json_file(result_path)
        if step_result is None:
            return 404, "Step result not found\n", "text/plain; charset=utf-8"
        training_config = step_result.get("extra", {}).get("training_config")
        if training_config is None:
            return 404, "No training config for this step\n", "text/plain; charset=utf-8"
        return 200, training_config, "application/json"
    if len(parts) == 8 and parts[:3] == ["", "api", "runs"] and parts[4] == "steps" and parts[6] == "samples":
        run_id = parts[3]
        step = parts[5]
        filename = parts[7]
        sample_path = runs_root / run_id / step / "samples" / filename
        if sample_path.exists() and sample_path.is_file():
            ext = sample_path.suffix.lower()
            content_type = MIME_TYPES.get(ext, "application/octet-stream")
            return 200, sample_path.read_bytes(), content_type
        return 404, "Sample not found\n", "text/plain; charset=utf-8"
    return None


def resolve_static_request(path: str) -> tuple[int, bytes, str] | None:
    if not STATIC_DIR.exists():
        return None
    normalized = unquote(path)
    if normalized.startswith("/assets/"):
        asset_path = STATIC_DIR / normalized.lstrip("/")
        if asset_path.exists() and asset_path.is_file():
            ext = asset_path.suffix.lower()
            content_type = MIME_TYPES.get(ext, "application/octet-stream")
            return 200, asset_path.read_bytes(), content_type
    return None


def resolve_spa_fallback() -> tuple[int, bytes, str] | None:
    if not STATIC_DIR.exists():
        return None
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return 200, index_path.read_bytes(), "text/html; charset=utf-8"
    return None


class ResultsRequestHandler(BaseHTTPRequestHandler):
    server_version = "Q19Results/0.2"

    def do_GET(self) -> None:  # noqa: N802
        root = Path(self.server.runs_root)  # type: ignore[attr-defined]
        api_result = resolve_api_request(self.path, root)
        if api_result is not None:
            status, payload, content_type = api_result
            if content_type == "application/json":
                self._respond_json(payload, status)  # type: ignore[arg-type]
                return
            if isinstance(payload, bytes):
                self._respond_binary(payload, status, content_type)
                return
            self._respond_text(str(payload), status)
            return
        static_result = resolve_static_request(self.path)
        if static_result is not None:
            status, body, content_type = static_result
            if content_type.startswith("image/") or content_type.startswith("font/"):
                self._respond_binary(body, status, content_type)
            else:
                self._respond_bytes(body, status, content_type)
            return
        spa_result = resolve_spa_fallback()
        if spa_result is not None:
            status, body, content_type = spa_result
            self._respond_html(body, status)
            return
        self._respond_text("Not found\n", 404)

    def log_message(self, format: str, *args: object) -> None:
        return

    def _respond_json(self, payload: dict[str, Any] | list[dict[str, Any]], status: int) -> None:
        body = _json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _respond_text(self, payload: str, status: int) -> None:
        body = payload.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _respond_html(self, body: bytes, status: int) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _respond_binary(self, body: bytes, status: int, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "public, max-age=31536000, immutable")
        self.end_headers()
        self.wfile.write(body)

    def _respond_bytes(self, body: bytes, status: int, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def serve_results(*, host: str = "127.0.0.1", port: int = 8000, runs_root: Path | None = None) -> None:
    root = runs_root or repo_root() / "reports" / "runs"
    server = ThreadingHTTPServer((host, port), ResultsRequestHandler)
    server.runs_root = str(root)  # type: ignore[attr-defined]
    try:
        server.serve_forever()
    finally:
        server.server_close()
