from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from qwen_image_19.config_io import repo_root
from qwen_image_19.reporting import collect_runs


def _json_bytes(payload: dict[str, Any] | list[dict[str, Any]]) -> bytes:
    return json.dumps(payload, indent=2).encode("utf-8")


def resolve_api_request(path: str, runs_root: Path) -> tuple[int, dict[str, Any] | list[dict[str, Any]] | str, str]:
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
    return 404, "Not found\n", "text/plain; charset=utf-8"


class ResultsRequestHandler(BaseHTTPRequestHandler):
    server_version = "Q19Results/0.1"

    def do_GET(self) -> None:  # noqa: N802
        root = Path(self.server.runs_root)  # type: ignore[attr-defined]
        status, payload, content_type = resolve_api_request(self.path, root)
        if content_type == "application/json":
            self._respond_json(payload, status)  # type: ignore[arg-type]
            return
        self._respond_text(str(payload), status)

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


def serve_results(*, host: str = "127.0.0.1", port: int = 8000, runs_root: Path | None = None) -> None:
    root = runs_root or repo_root() / "reports" / "runs"
    server = ThreadingHTTPServer((host, port), ResultsRequestHandler)
    server.runs_root = str(root)  # type: ignore[attr-defined]
    try:
        server.serve_forever()
    finally:
        server.server_close()
