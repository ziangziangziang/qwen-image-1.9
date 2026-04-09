"""Qwen-Image 1.9 results server.

Pure-Python HTTP server (``ThreadingHTTPServer``) that serves:
- JSON API at ``/api/…`` for run data and sample images
- Embedded HTML dashboard at ``/`` (no React build required)
"""
from __future__ import annotations

import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from qwen_image_19.config_io import repo_root
from qwen_image_19.contracts import PIPELINE_STEPS
from qwen_image_19.reporting import collect_runs

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}


# ── JSON helpers ────────────────────────────────────────────────────

def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, indent=2, default=str).encode("utf-8")


def _read_json(path: Path) -> Any | None:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
    return None


# ── API router ──────────────────────────────────────────────────────

def resolve_api(path: str, runs_root: Path) -> tuple[int, Any, str] | None:
    """Match an API path and return (status, body, content_type) or None."""
    p = unquote(path.rstrip("/"))
    parts = p.split("/")

    # GET /api/runs
    if p == "/api/runs":
        runs = collect_runs(runs_root)
        items = [
            {
                "run_id": r["run_id"],
                "updated_at": r["updated_at"],
                "steps": {
                    s: r["steps"][s]["status"]
                    for s in PIPELINE_STEPS if s in r.get("steps", {})
                },
                "tags": r.get("tags", []),
            }
            for r in runs
        ]
        return 200, items, "application/json"

    # GET /api/runs/{run_id}
    if len(parts) == 4 and parts[:3] == ["", "api", "runs"]:
        data = _read_json(runs_root / parts[3] / "manifest.json")
        if data:
            return 200, data, "application/json"
        return 404, {"error": "run not found"}, "application/json"

    # GET /api/runs/{run_id}/steps/{step}
    if len(parts) == 6 and parts[4] == "steps":
        run_id, step = parts[3], parts[5]
        data = _read_json(runs_root / run_id / step / "step-result.json")
        if data:
            return 200, data, "application/json"
        return 404, {"error": "step not found"}, "application/json"

    # GET /api/runs/{run_id}/steps/{step}/eval-summary
    if len(parts) == 7 and parts[4] == "steps" and parts[6] == "eval-summary":
        run_id, step = parts[3], parts[5]
        data = _read_json(runs_root / run_id / step / "eval-summary.json")
        if data:
            return 200, data, "application/json"
        return 404, {"error": "eval summary not found"}, "application/json"

    # GET /api/runs/{run_id}/steps/{step}/samples
    if len(parts) == 7 and parts[4] == "steps" and parts[6] == "samples":
        run_id, step = parts[3], parts[5]
        sdir = runs_root / run_id / step / "samples"
        samples = sorted(f.name for f in sdir.glob("*") if f.is_file()) if sdir.exists() else []
        return 200, {"run_id": run_id, "step": step, "samples": samples}, "application/json"

    # GET /api/runs/{run_id}/steps/{step}/samples/{filename}
    if len(parts) == 8 and parts[4] == "steps" and parts[6] == "samples":
        run_id, step, fname = parts[3], parts[5], parts[7]
        fpath = runs_root / run_id / step / "samples" / fname
        if fpath.exists() and fpath.is_file():
            ct = mimetypes.guess_type(fpath.name)[0] or "application/octet-stream"
            return 200, fpath.read_bytes(), ct
        return 404, b"not found", "text/plain"

    # GET /api/runs/{run_id}/steps/{step}/metrics
    if len(parts) == 7 and parts[4] == "steps" and parts[6] == "metrics":
        run_id, step = parts[3], parts[5]
        step_result = _read_json(runs_root / run_id / step / "step-result.json")
        metrics_files: list[dict[str, Any]] = []
        if step_result:
            extra = step_result.get("extra", {})
            adir = (extra.get("legacy_stage_result") or {}).get("artifact_dir")
            if adir:
                mdir = Path(adir) if Path(adir).is_absolute() else repo_root() / adir
                mdir = mdir / "metrics"
                if mdir.exists():
                    for mf in sorted(mdir.glob("*.json")):
                        try:
                            metrics_files.append({"name": mf.stem, **json.loads(mf.read_text("utf-8"))})
                        except (json.JSONDecodeError, OSError):
                            pass
        return 200, {"run_id": run_id, "step": step, "metrics": metrics_files}, "application/json"

    # GET /api/runs/{run_id}/steps/{step}/training-config
    if len(parts) == 7 and parts[4] == "steps" and parts[6] == "training-config":
        run_id, step = parts[3], parts[5]
        step_result = _read_json(runs_root / run_id / step / "step-result.json")
        if step_result:
            tc = (step_result.get("extra") or {}).get("training_config")
            if tc:
                return 200, tc, "application/json"
        return 404, {"error": "no training config"}, "application/json"

    return None


# ── Embedded HTML dashboard ─────────────────────────────────────────

def _dashboard_html() -> str:
    """Return the full single-page dashboard HTML."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Qwen-Image 1.9 Dashboard</title>
<style>
:root {
  --bg: #0f1117; --bg2: #1a1d27; --card: #222533; --hover: #2a2d3e;
  --border: #2e3148; --text: #e4e6f0; --text2: #9498b3; --muted: #6b7094;
  --accent: #6c5ce7; --ok: #00b894; --warn: #fdcb6e; --err: #e17055; --info: #74b9ff;
  --r: 8px;
}
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Arial,sans-serif;
       background:var(--bg); color:var(--text); line-height:1.6; }
a { color:var(--accent); text-decoration:none; }
a:hover { text-decoration:underline; }
.wrap { max-width:1200px; margin:0 auto; padding:0 24px; }
.hdr { padding:32px 0 24px; border-bottom:1px solid var(--border); margin-bottom:32px; }
.hdr h1 { font-size:28px; font-weight:700; letter-spacing:-0.5px; }
.hdr p { color:var(--text2); margin-top:4px; }
.card { background:var(--card); border:1px solid var(--border); border-radius:12px; padding:20px; margin-bottom:16px; }
.card h3 { font-size:16px; font-weight:600; margin-bottom:12px; }
table { width:100%; border-collapse:collapse; }
th,td { text-align:left; padding:12px 16px; border-bottom:1px solid var(--border); }
th { color:var(--text2); font-size:12px; font-weight:600; text-transform:uppercase; letter-spacing:.5px; }
td { font-size:14px; }
tr:hover td { background:var(--hover); }
.badge { display:inline-flex; padding:2px 10px; border-radius:999px; font-size:12px;
         font-weight:600; text-transform:uppercase; letter-spacing:.5px; }
.badge-completed,.badge-succeeded { background:rgba(0,184,148,.15); color:var(--ok); }
.badge-pending { background:rgba(107,112,148,.2); color:var(--muted); }
.badge-planned { background:rgba(253,203,110,.15); color:var(--warn); }
.badge-failed,.badge-error { background:rgba(225,112,85,.15); color:var(--err); }
.badge-running,.badge-ready { background:rgba(116,185,255,.15); color:var(--info); }
.metrics-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(180px,1fr)); gap:12px; }
.metric { background:var(--bg2); border-radius:var(--r); padding:16px; border-left:3px solid var(--accent); }
.metric .label { font-size:11px; color:var(--muted); text-transform:uppercase; letter-spacing:.5px; }
.metric .value { font-size:22px; font-weight:700; margin-top:4px; }
.gallery { display:grid; grid-template-columns:repeat(auto-fill,minmax(220px,1fr)); gap:16px; }
.gallery img { width:100%; border-radius:var(--r); cursor:pointer; transition:transform .2s; }
.gallery img:hover { transform:scale(1.03); }
.gallery .caption { font-size:11px; color:var(--muted); margin-top:4px; text-align:center; }
.stepper { display:flex; align-items:center; gap:0; padding:12px 0; }
.step-item { display:flex; flex-direction:column; align-items:center; flex:1; gap:6px; }
.step-dot { width:36px; height:36px; border-radius:50%; display:flex; align-items:center;
            justify-content:center; font-size:16px; border:2px solid var(--border); cursor:pointer; }
.step-dot.active { border-color:var(--ok); background:rgba(0,184,148,.15); }
.step-dot.pending { opacity:.4; }
.step-label { font-size:11px; color:var(--text2); text-transform:uppercase; }
.step-line { flex:1; height:2px; background:var(--border); margin:0 -8px; }
.step-line.done { background:var(--ok); }
.back { font-size:14px; margin-bottom:16px; display:inline-block; }
.loading { text-align:center; padding:64px; color:var(--text2); }
.error { background:rgba(225,112,85,.1); border:1px solid rgba(225,112,85,.3);
         border-radius:var(--r); padding:16px; color:var(--err); }
.empty { text-align:center; padding:64px; color:var(--muted); }
.tabs { display:flex; gap:4px; margin-bottom:16px; }
.tab { padding:8px 16px; border-radius:var(--r) var(--r) 0 0; cursor:pointer;
       font-size:13px; font-weight:600; border:1px solid var(--border); border-bottom:none;
       background:var(--bg2); color:var(--text2); }
.tab.active { background:var(--card); color:var(--text); }
#lightbox { display:none; position:fixed; inset:0; background:rgba(0,0,0,.85);
            z-index:1000; align-items:center; justify-content:center; cursor:pointer; }
#lightbox.show { display:flex; }
#lightbox img { max-width:90vw; max-height:90vh; border-radius:12px; }
</style>
</head>
<body>
<div class="wrap" id="app"></div>
<div id="lightbox" onclick="this.classList.remove('show')"><img id="lb-img"></div>
<script>
const STEPS = ['merge','post_merge_train','abliterate','post_abliterate_train','quantize','post_quantize_eval'];
const STEP_LABELS = {merge:'Merge',post_merge_train:'Train',abliterate:'Abliterate',
  post_abliterate_train:'Train',quantize:'Quantize',post_quantize_eval:'Eval'};
const STEP_ICONS = {merge:'🔀',post_merge_train:'🏋️',abliterate:'✂️',
  post_abliterate_train:'🏋️',quantize:'📦',post_quantize_eval:'📊'};

const $ = s => document.querySelector(s);
const api = async p => { const r=await fetch('/api'+p); if(!r.ok) throw new Error(r.statusText); return r.json(); };

function badge(status) {
  return `<span class="badge badge-${status||'pending'}">${status||'pending'}</span>`;
}

function showLightbox(src) {
  $('#lb-img').src = src;
  $('#lightbox').classList.add('show');
}

// ── Run List ─────────────────────────────────────────────────────
async function showRunList() {
  $('#app').innerHTML = '<div class="loading">Loading runs...</div>';
  try {
    const runs = await api('/runs');
    if (!runs.length) { $('#app').innerHTML = '<div class="empty"><h3>No runs found</h3><p>Run <code>q19 merge</code> to start.</p></div>'; return; }
    const stepHeaders = STEPS.map(s=>`<th>${STEP_LABELS[s]}</th>`).join('');
    const rows = runs.map(r => {
      const cells = STEPS.map(s => `<td>${badge(r.steps[s]||'pending')}</td>`).join('');
      return `<tr onclick="showRunDetail('${r.run_id}')" style="cursor:pointer">
        <td><strong>${r.run_id}</strong></td>
        <td style="color:var(--text2);font-size:13px">${new Date(r.updated_at).toLocaleString()}</td>
        ${cells}
        <td style="color:var(--muted);font-size:12px">${(r.tags||[]).join(', ')||'—'}</td>
      </tr>`;
    }).join('');
    $('#app').innerHTML = `
      <div class="hdr"><h1>Qwen-Image 1.9 Dashboard</h1><p>${runs.length} run${runs.length!==1?'s':''}</p></div>
      <table><thead><tr><th>Run ID</th><th>Updated</th>${stepHeaders}<th>Tags</th></tr></thead>
      <tbody>${rows}</tbody></table>`;
  } catch(e) { $('#app').innerHTML = `<div class="error">Error: ${e.message}</div>`; }
}

// ── Run Detail ───────────────────────────────────────────────────
async function showRunDetail(runId) {
  $('#app').innerHTML = '<div class="loading">Loading run...</div>';
  try {
    const m = await api('/runs/'+runId);
    const stepper = renderStepper(m.steps, runId);
    const models = Object.values(m.source_models||{}).map(mo =>
      `<tr><td><code>${mo.alias}</code></td><td>${mo.model_id}</td><td>${badge(mo.role)}</td>
       <td style="font-size:12px;color:var(--text2)">${mo.architecture?.backbone||'—'}</td></tr>`
    ).join('');
    const stepRows = STEPS.map(s => {
      const rec = m.steps[s]; if(!rec) return '';
      const ok = rec.status==='completed'||rec.status==='succeeded';
      return `<tr ${ok?`onclick="showStepDetail('${runId}','${s}')" style="cursor:pointer"`:''}>
        <td><strong>${STEP_LABELS[s]}</strong></td><td>${badge(rec.status)}</td>
        <td style="font-size:12px;max-width:250px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${rec.input_checkpoint||'—'}</td>
        <td style="font-size:12px;max-width:250px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${rec.output_checkpoint||'—'}</td></tr>`;
    }).join('');
    $('#app').innerHTML = `
      <a class="back" href="#" onclick="showRunList();return false">← Back to runs</a>
      <div class="hdr"><h1>${m.run_id}</h1>
        <p>Created ${new Date(m.created_at).toLocaleString()} · Updated ${new Date(m.updated_at).toLocaleString()}</p></div>
      ${stepper}
      <div class="card"><h3>Source Models (HuggingFace)</h3>
        <table><thead><tr><th>Alias</th><th>Model ID</th><th>Role</th><th>Architecture</th></tr></thead>
        <tbody>${models}</tbody></table></div>
      <div class="card"><h3>Pipeline Steps</h3>
        <table><thead><tr><th>Step</th><th>Status</th><th>Input</th><th>Output</th></tr></thead>
        <tbody>${stepRows}</tbody></table></div>
      ${m.notes?`<div class="card"><h3>Notes</h3><p style="color:var(--text2)">${m.notes}</p></div>`:''}`;
  } catch(e) { $('#app').innerHTML = `<div class="error">Error: ${e.message}</div>`; }
}

function renderStepper(steps, runId) {
  let html = '<div class="card"><h3>Pipeline Progress</h3><div class="stepper">';
  STEPS.forEach((s,i) => {
    const rec = steps[s]||{};
    const ok = rec.status==='completed'||rec.status==='succeeded';
    const cls = ok ? 'active' : (rec.status==='pending'||!rec.status) ? 'pending' : '';
    html += `<div class="step-item" onclick="showStepDetail('${runId}','${s}')">
      <div class="step-dot ${cls}">${STEP_ICONS[s]}</div>
      <div class="step-label">${STEP_LABELS[s]}</div></div>`;
    if (i < STEPS.length-1) html += `<div class="step-line ${ok?'done':''}"></div>`;
  });
  return html + '</div></div>';
}

// ── Step Detail ──────────────────────────────────────────────────
async function showStepDetail(runId, step) {
  $('#app').innerHTML = '<div class="loading">Loading step...</div>';
  try {
    const m = await api('/runs/'+runId);
    const rec = m.steps[step]; if(!rec) { $('#app').innerHTML='<div class="empty">Step not found</div>'; return; }
    const ok = rec.status==='completed'||rec.status==='succeeded';
    const stepper = renderStepper(m.steps, runId);

    // Metrics
    let metricsHtml = '';
    if (rec.metrics && Object.keys(rec.metrics).length) {
      const cards = Object.entries(rec.metrics).filter(([,v])=>v!=null).map(([k,v]) =>
        `<div class="metric"><div class="label">${k.replace(/_/g,' ')}</div>
         <div class="value">${typeof v==='number'?(Number.isInteger(v)?v:v.toFixed(4)):v}</div></div>`
      ).join('');
      metricsHtml = `<div class="card"><h3>Metrics</h3><div class="metrics-grid">${cards}</div></div>`;
    }

    // Samples gallery
    let galleryHtml = '';
    try {
      const sData = await api(`/runs/${runId}/steps/${step}/samples`);
      const imgs = (sData.samples||[]).filter(f=>/\\.(png|jpg|jpeg|gif|webp)$/i.test(f));
      if (imgs.length) {
        const tiles = imgs.map(f => {
          const url = `/api/runs/${runId}/steps/${step}/samples/${f}`;
          return `<div><img src="${url}" alt="${f}" onclick="showLightbox('${url}')" loading="lazy">
                  <div class="caption">${f}</div></div>`;
        }).join('');
        galleryHtml = `<div class="card"><h3>Generated / Modified Images (${imgs.length})</h3>
          <div class="gallery">${tiles}</div></div>`;
      }
    } catch(_) {}

    // Training loss
    let lossHtml = '';
    try {
      const mData = await api(`/runs/${runId}/steps/${step}/metrics`);
      const lossSeries = (mData.metrics||[]).filter(m=>m.loss_curve&&m.loss_curve.length);
      if (lossSeries.length) {
        lossHtml = '<div class="card"><h3>Training Loss</h3>';
        lossSeries.forEach(s => {
          const maxVal = Math.max(...s.loss_curve);
          const h = 200;
          const w = 600;
          const pts = s.loss_curve.map((v,i)=>`${(i/s.loss_curve.length)*w},${h-(v/maxVal)*h}`).join(' ');
          lossHtml += `<div style="margin-bottom:12px">
            <div style="font-size:13px;color:var(--text2);margin-bottom:8px">${s.name}
              — final: ${s.final_loss?.toFixed(6)||'n/a'}, min: ${s.min_loss?.toFixed(6)||'n/a'}</div>
            <svg viewBox="0 0 ${w} ${h}" style="width:100%;max-width:${w}px;height:auto;background:var(--bg2);border-radius:var(--r)">
              <polyline points="${pts}" fill="none" stroke="var(--accent)" stroke-width="2"/>
            </svg></div>`;
        });
        lossHtml += '</div>';
      }
    } catch(_) {}

    // Eval summary
    let evalHtml = '';
    try {
      const ev = await api(`/runs/${runId}/steps/${step}/eval-summary`);
      if (ev.aggregate_metrics && Object.keys(ev.aggregate_metrics).length) {
        const rows = Object.entries(ev.aggregate_metrics).map(([k,v]) =>
          `<tr><td><code>${k}</code></td><td>${typeof v==='number'?v.toFixed(4):v}</td></tr>`
        ).join('');
        evalHtml = `<div class="card"><h3>Evaluation Summary</h3>
          <table><thead><tr><th>Metric</th><th>Value</th></tr></thead>
          <tbody>${rows}</tbody></table></div>`;
      }
    } catch(_) {}

    // Artifacts
    let artHtml = '';
    if (rec.artifacts?.length) {
      const rows = rec.artifacts.map(a =>
        `<tr><td><code>${a.kind}</code></td><td style="font-size:12px">${a.path_or_uri}</td><td>${a.content_type}</td></tr>`
      ).join('');
      artHtml = `<div class="card"><h3>Artifacts</h3>
        <table><thead><tr><th>Kind</th><th>Path</th><th>Type</th></tr></thead>
        <tbody>${rows}</tbody></table></div>`;
    }

    // Execution details
    const execHtml = `<div class="card"><h3>Execution Details</h3>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;font-size:13px">
        <div><div style="color:var(--muted);font-size:11px;text-transform:uppercase;margin-bottom:4px">Command</div>
          <code style="font-size:12px">${(rec.command||[]).join(' ')||'—'}</code></div>
        <div><div style="color:var(--muted);font-size:11px;text-transform:uppercase;margin-bottom:4px">Remote Job</div>
          <div>${rec.remote_job?.name||'—'}</div></div>
        <div><div style="color:var(--muted);font-size:11px;text-transform:uppercase;margin-bottom:4px">Input</div>
          <code style="font-size:11px;word-break:break-all">${rec.input_checkpoint||'—'}</code></div>
        <div><div style="color:var(--muted);font-size:11px;text-transform:uppercase;margin-bottom:4px">Output</div>
          <code style="font-size:11px;word-break:break-all">${rec.output_checkpoint||'—'}</code></div>
      </div></div>`;

    $('#app').innerHTML = `
      <a class="back" href="#" onclick="showRunDetail('${runId}');return false">← Back to ${runId}</a>
      <div class="hdr">
        <h1>${runId} / ${STEP_LABELS[step]||step}</h1>
        <p>${badge(rec.status)}</p></div>
      ${stepper}
      ${!ok?`<div class="card"><h3>Step Not Complete</h3>
        <p style="color:var(--text2)">This step is <code>${rec.status}</code>. Run the pipeline with --execute to produce results.</p></div>`:''}
      ${metricsHtml}
      ${lossHtml}
      ${galleryHtml}
      ${evalHtml}
      ${artHtml}
      ${execHtml}`;
  } catch(e) { $('#app').innerHTML = `<div class="error">Error: ${e.message}</div>`; }
}

// ── Init ─────────────────────────────────────────────────────────
showRunList();
</script>
</body>
</html>"""


# ── Request handler ─────────────────────────────────────────────────

class _Handler(BaseHTTPRequestHandler):
    server_version = "Q19/0.3"

    def do_GET(self) -> None:  # noqa: N802
        root = Path(self.server.runs_root)  # type: ignore[attr-defined]

        # API routes
        api_result = resolve_api(self.path, root)
        if api_result is not None:
            status, body, ct = api_result
            if ct == "application/json" and not isinstance(body, bytes):
                self._send(status, _json_bytes(body), ct)
            elif isinstance(body, bytes):
                self._send(status, body, ct)
            else:
                self._send(status, str(body).encode("utf-8"), ct)
            return

        # Dashboard (any non-API path)
        html = _dashboard_html().encode("utf-8")
        self._send(200, html, "text/html; charset=utf-8")

    def _send(self, status: int, body: bytes, ct: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(body)))
        if ct.startswith("image/"):
            self.send_header("Cache-Control", "public, max-age=86400")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: object) -> None:
        return  # suppress default stderr logging


# ── Entry point ─────────────────────────────────────────────────────

def serve_results(*, host: str = "127.0.0.1", port: int = 8000, runs_root: Path | None = None) -> None:
    root = runs_root or repo_root() / "reports" / "runs"
    server = ThreadingHTTPServer((host, port), _Handler)
    server.runs_root = str(root)  # type: ignore[attr-defined]
    print(f"Qwen-Image 1.9 dashboard: http://{host}:{port}/")
    try:
        server.serve_forever()
    finally:
        server.server_close()
