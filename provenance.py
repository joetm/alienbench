"""Reproducibility metadata.

Writes a ``run_manifest.json`` into ``results_dir`` that captures the config
that produced the outputs, the package versions in use, and a timestamp.
Downstream users can diff manifests to verify that two result directories were
produced by the same configuration.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from importlib import metadata as _metadata
from pathlib import Path

from alienbench.config import Config

_TRACKED_PACKAGES = (
    "openai", "pydantic", "pyyaml", "pandas", "scipy",
    "matplotlib", "seaborn", "krippendorff", "tqdm",
)


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for pkg in _TRACKED_PACKAGES:
        try:
            versions[pkg] = _metadata.version(pkg)
        except _metadata.PackageNotFoundError:
            versions[pkg] = "not-installed"
    return versions


def config_hash(cfg: Config) -> str:
    """SHA-256 over the config's JSON representation (stable key order)."""
    payload = cfg.model_dump_json(exclude={"allowed_providers"})
    # re-dump with sorted keys for stability across Pydantic versions
    obj = json.loads(payload)
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def write_run_manifest(cfg: Config, results_dir: Path, stage: str) -> Path:
    """Append a run entry to ``run_manifest.json`` in ``results_dir``.

    Each entry records the stage name, a config hash, the full resolved config,
    package versions, Python version, and a UTC timestamp. The file is a JSON
    array so re-running stages accumulates history.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / "run_manifest.json"

    entry = {
        "stage": stage,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config_hash": config_hash(cfg),
        "config": json.loads(cfg.model_dump_json()),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": _package_versions(),
    }

    if path.exists():
        try:
            history = json.loads(path.read_text())
            if not isinstance(history, list):
                history = [history]
        except json.JSONDecodeError:
            history = []
    else:
        history = []

    history.append(entry)
    path.write_text(json.dumps(history, indent=2) + "\n")
    return path
