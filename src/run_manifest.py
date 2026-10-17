"""Immutable run manifests.

A result file without a manifest is not evidence: "spurious trend rate 31%" means
nothing without the dataset it was measured on, the prompt template, the sampling
settings and the resolved model id.  Every run writes one manifest beside its
results, and a manifest that already exists may not be silently overwritten with
different content -- appending to a result file under changed conditions would
mix two experiments into one denominator.

The manifest's ``fingerprint`` excludes its invocation id and timestamp, so
resuming the same configuration produces the same fingerprint while retaining
the original run id and recording when the run began.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

#: Fields deliberately excluded from the reproducibility fingerprint. A fresh
#: invocation has a fresh run id, but a resume is the same experiment.
_NON_IDENTIFYING = ("created_at", "run_id")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def git_commit_hash(cwd: str | Path | None = None) -> Optional[str]:
    """The commit the run was made from, or None.

    None is a legitimate answer -- this project is not always in a git
    repository -- and recording None honestly is better than recording a
    placeholder that later reads as a real commit.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def git_is_dirty(cwd: str | Path | None = None) -> Optional[bool]:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return bool(out.stdout.strip())


def package_versions() -> Dict[str, str]:
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("pydantic", "yaml", "pytest"):
        try:
            module = __import__(name)
        except ImportError:  # pragma: no cover - environment dependent
            continue
        versions[name] = str(
            getattr(module, "VERSION", None)
            or getattr(module, "__version__", None)
            or "unknown"
        )
    return versions


class RunManifest(BaseModel):
    """Everything needed to interpret or reproduce one result file."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    arm: str
    prompt_variant: Optional[str]

    provider: str
    model: str
    requested_model_id: str

    dataset_path: str
    dataset_sha256: str
    competence_dataset_sha256: Optional[str] = None
    config_path: str
    config_sha256: str

    temperature: Optional[float]
    max_tokens: int
    seed: Optional[int]
    structured_output: bool
    max_transport_retries: int

    #: Provider-specific pins such as a local model digest or gateway route.
    provider_constraints: Dict[str, Any] = Field(default_factory=dict)
    #: Runtime hardware and explicit subset selection, when applicable.
    hardware: Optional[Dict[str, Any]] = None
    selected_case_ids: Optional[List[str]] = None

    git_commit: Optional[str] = None
    git_dirty: Optional[bool] = None
    package_versions: Dict[str, str]

    created_at: str

    @property
    def fingerprint(self) -> str:
        """Hash of experiment conditions, excluding invocation metadata.

        Two runs of the same configuration share a fingerprint; a different
        invocation id or timestamp does not change the meaning of the results.
        """
        payload = {
            k: v for k, v in self.model_dump(mode="json").items() if k not in _NON_IDENTIFYING
        }
        return sha256_text(json.dumps(payload, sort_keys=True, ensure_ascii=False))

    def identifying_fields(self) -> Dict[str, Any]:
        return {
            k: v for k, v in self.model_dump(mode="json").items() if k not in _NON_IDENTIFYING
        }


class ManifestConflict(RuntimeError):
    """An existing manifest describes a different experiment than this run."""


def build_manifest(
    *,
    run_id: str,
    arm: str,
    prompt_variant: Optional[str],
    provider: str,
    model: str,
    dataset_path: str | Path,
    config_path: str | Path,
    temperature: Optional[float],
    max_tokens: int,
    seed: Optional[int],
    structured_output: bool,
    max_transport_retries: int,
    competence_dataset_path: str | Path | None = None,
    requested_model_id: Optional[str] = None,
    provider_constraints: Optional[Dict[str, Any]] = None,
    hardware: Optional[Dict[str, Any]] = None,
    selected_case_ids: Optional[List[str]] = None,
    repo_root: str | Path | None = None,
    now: Optional[datetime] = None,
) -> RunManifest:
    return RunManifest(
        run_id=run_id,
        arm=arm,
        prompt_variant=prompt_variant,
        provider=provider,
        model=model,
        requested_model_id=requested_model_id or model,
        dataset_path=str(Path(dataset_path).as_posix()),
        dataset_sha256=sha256_file(dataset_path),
        competence_dataset_sha256=(
            sha256_file(competence_dataset_path) if competence_dataset_path else None
        ),
        config_path=str(Path(config_path).as_posix()),
        config_sha256=sha256_file(config_path),
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
        structured_output=structured_output,
        max_transport_retries=max_transport_retries,
        provider_constraints=dict(provider_constraints or {}),
        hardware=dict(hardware) if hardware is not None else None,
        selected_case_ids=list(selected_case_ids) if selected_case_ids is not None else None,
        git_commit=git_commit_hash(repo_root),
        git_dirty=git_is_dirty(repo_root),
        package_versions=package_versions(),
        created_at=(now or datetime.now(timezone.utc)).isoformat(),
    )


def manifest_path_for(result_path: str | Path) -> Path:
    return Path(result_path).with_suffix(".manifest.json")


def write_manifest(manifest: RunManifest, result_path: str | Path) -> Path:
    """Write, or verify that an existing manifest describes the same experiment.

    Refusing to overwrite is the point: appending new rows to a result file whose
    manifest has changed would silently merge two experiments.
    """
    path = manifest_path_for(result_path)
    if path.exists():
        existing = read_manifest(path)
        if existing.fingerprint != manifest.fingerprint:
            differing = sorted(
                k
                for k, v in manifest.identifying_fields().items()
                if existing.identifying_fields().get(k) != v
            )
            raise ManifestConflict(
                f"{path.name} already describes a different run; differing fields: "
                f"{differing}. Results for a changed configuration belong in a new file."
            )
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    return path


def read_manifest(path: str | Path) -> RunManifest:
    return RunManifest(**json.loads(Path(path).read_text(encoding="utf-8")))
