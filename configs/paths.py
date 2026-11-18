"""Project-root aware paths: never depend on the current working directory."""
from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ProjectPaths:
    """Resolve application paths consistently in Docker, WSL and local runs."""

    root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", self.root.expanduser().resolve())

    @classmethod
    def from_environment(cls) -> "ProjectPaths":
        root = os.environ.get("PROJECT_ROOT")
        return cls(Path(root) if root else Path(__file__).resolve().parents[1])

    def resolve(self, path: str | Path) -> Path:
        """Resolve relative files against the project root, not os.getcwd()."""
        candidate = Path(os.path.expandvars(str(path))).expanduser()
        return (candidate if candidate.is_absolute() else self.root / candidate).resolve()

    @property
    def config_file(self) -> Path:
        return self.root / "configs" / "config.yaml"

    @property
    def env_file(self) -> Path:
        return self.root / ".env"

    @property
    def benchmarks(self) -> Path:
        return self.root / "artifacts" / "benchmarks"

    def ensure_directory(self, path: str | Path) -> Path:
        directory = self.resolve(path)
        directory.mkdir(parents=True, exist_ok=True)
        return directory
