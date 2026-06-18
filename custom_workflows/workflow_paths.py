from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


DEFAULT_LOGS_ROOT = Path("logs") / "rsl_rl"
DEFAULT_RESULTS_ROOT = Path("source") / "results" / "rsl_rl"
DEFAULT_ARTIFACTS_ROOT = Path("artifacts") / "rsl_rl"
DEFAULT_EXPORT_DIRNAME = "exported"


def _coerce_root(root: str | Path | None, repo_root: Path, default_relative: Path) -> Path:
    if root is None:
        return (repo_root / default_relative).resolve()

    candidate = Path(root).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (repo_root / candidate).resolve()


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _run_directory_name(run_name: str | None, timestamp: str | None = None) -> str:
    run_stamp = timestamp or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if run_name:
        return f"{run_stamp}_{run_name}"
    return run_stamp


@dataclass(frozen=True)
class WorkflowPaths:
    repo_root: Path
    logs_root: Path
    results_root: Path
    artifacts_root: Path

    @classmethod
    def from_overrides(
        cls,
        repo_root: str | Path | None = None,
        logs_root: str | Path | None = None,
        results_root: str | Path | None = None,
        artifacts_root: str | Path | None = None,
    ) -> "WorkflowPaths":
        resolved_repo_root = Path(repo_root).expanduser().resolve() if repo_root else Path(__file__).resolve().parent.parent
        return cls(
            repo_root=resolved_repo_root,
            logs_root=_coerce_root(logs_root, resolved_repo_root, DEFAULT_LOGS_ROOT),
            results_root=_coerce_root(results_root, resolved_repo_root, DEFAULT_RESULTS_ROOT),
            artifacts_root=_coerce_root(artifacts_root, resolved_repo_root, DEFAULT_ARTIFACTS_ROOT),
        )

    def training_log_root(self, experiment_name: str) -> Path:
        return self.logs_root / experiment_name

    def training_run_dir(self, experiment_name: str, run_name: str | None, timestamp: str | None = None) -> Path:
        return self.training_log_root(experiment_name) / _run_directory_name(run_name, timestamp)

    def result_experiment_root(self, experiment_name: str) -> Path:
        return self.results_root / experiment_name

    def result_run_dir(self, experiment_name: str, load_run: str, checkpoint_name: str) -> Path:
        return self.result_experiment_root(experiment_name) / load_run / f"{Path(checkpoint_name).stem}_play"

    def result_file(self, experiment_name: str, load_run: str, checkpoint_name: str, filename: str) -> Path:
        return self.result_run_dir(experiment_name, load_run, checkpoint_name) / filename

    def export_dir(self, experiment_name: str, load_run: str, checkpoint_name: str) -> Path:
        return self.artifacts_root / experiment_name / load_run / Path(checkpoint_name).stem / DEFAULT_EXPORT_DIRNAME


def resolve_checkpoint_identifiers(
    resume_path: str | Path,
    configured_load_run: Optional[str] = None,
) -> tuple[str, str]:
    """Return stable ``(load_run, checkpoint_name)`` identifiers for result paths.

    ``get_checkpoint_path`` and custom absolute weight paths both eventually
    resolve to a concrete checkpoint file.  This helper prevents downstream
    scripts from losing the real run/checkpoint identity when a custom path is
    used.
    """

    checkpoint_path = Path(resume_path).expanduser()
    checkpoint_name = checkpoint_path.name
    load_run = configured_load_run or checkpoint_path.parent.name
    return load_run, checkpoint_name


def resolve_checkpoint_path(
    log_root_path: str | Path,
    load_run: Optional[str],
    checkpoint_name: Optional[str],
    fallback_resolver,
) -> str:
    """Resolve a checkpoint while preferring exact run/checkpoint paths.

    Isaac Lab's ``get_checkpoint_path`` treats ``load_run`` and checkpoint names
    like patterns.  That is useful for latest-run lookup, but unsafe for named
    baselines such as ``SS4`` because it can match ``SS4_DRB``.  This helper
    keeps explicit CLI contracts exact and only falls back to pattern lookup
    when the user intentionally leaves either part unspecified.
    """

    log_root = Path(log_root_path).expanduser()
    if load_run and checkpoint_name and "*" not in load_run and "*" not in checkpoint_name:
        checkpoint_file = checkpoint_name if str(checkpoint_name).endswith(".pt") else f"{checkpoint_name}.pt"
        candidate = log_root / load_run / checkpoint_file
        if candidate.exists():
            return str(candidate)
        raise FileNotFoundError(f"Exact checkpoint not found: {candidate}")
    return fallback_resolver(str(log_root), load_run, checkpoint_name)
