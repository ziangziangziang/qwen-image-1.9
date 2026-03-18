from __future__ import annotations

from pathlib import Path
from typing import Any

from qwen_image_19.config_io import env_or_default, load_env_file, load_json_yaml, repo_root


def default_remote_context(remote_config: str | None = None) -> dict[str, Any]:
    base = {
        "name": env_or_default("REMOTE_NAME", "local-dry-run"),
        "workdir": env_or_default("REMOTE_WORKDIR", "/mnt/experiments/qwen-image-1.9"),
        "cache_dir": env_or_default("REMOTE_CACHE_DIR", "/mnt/cache/qwen-image"),
        "artifact_dir": env_or_default("REMOTE_ARTIFACT_DIR", "/mnt/artifacts/qwen-image-1.9"),
        "python": env_or_default("REMOTE_PYTHON", "python3"),
    }
    if not remote_config:
        return base

    config_path = Path(remote_config)
    if config_path.suffix == ".env":
        env_values = load_env_file(config_path)
        return {
            "name": env_values.get("REMOTE_NAME", base["name"]),
            "workdir": env_values.get("REMOTE_WORKDIR", base["workdir"]),
            "cache_dir": env_values.get("REMOTE_CACHE_DIR", base["cache_dir"]),
            "artifact_dir": env_values.get("REMOTE_ARTIFACT_DIR", base["artifact_dir"]),
            "python": env_values.get("REMOTE_PYTHON", base["python"]),
        }

    payload = load_json_yaml(config_path)
    launcher = payload.get("launcher", {})
    return {
        "name": launcher.get("type", base["name"]),
        "workdir": payload.get("remote", {}).get("workdir", base["workdir"]),
        "cache_dir": payload.get("remote", {}).get("cache_dir", base["cache_dir"]),
        "artifact_dir": payload.get("remote", {}).get("artifact_dir", base["artifact_dir"]),
        "python": launcher.get("python", base["python"]),
    }


def default_remote_paths_config() -> Path:
    return repo_root() / "configs" / "remote" / "paths.example.yaml"

