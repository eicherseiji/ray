from typing import Any, Optional, Type

_AnyscaleRunsArtifactRepositoryClass: Optional[Type[Any]] = None


def _create_anyscale_runs_artifact_repository_class() -> Type[Any]:
    """Create the AnyscaleRunsArtifactRepository class."""
    from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository

    from ray.anyscale.lineage.mlflow_lineage.store.artifact.artifact_repo_mixin import (
        AnyscaleArtifactRepositoryMixin,
    )

    class AnyscaleRunsArtifactRepository(
        AnyscaleArtifactRepositoryMixin, RunsArtifactRepository
    ):
        """Anyscale implementation of MLflow Runs artifact repository."""

        def __init__(self, artifact_uri: str, *args: Any, **kwargs: Any) -> None:
            super().__init__(artifact_uri, *args, **kwargs)

    return AnyscaleRunsArtifactRepository


def AnyscaleRunsArtifactRepository(*args: Any, **kwargs: Any) -> Any:  # noqa: N802
    """Factory function that creates AnyscaleRunsArtifactRepository instances."""
    global _AnyscaleRunsArtifactRepositoryClass
    if _AnyscaleRunsArtifactRepositoryClass is None:
        _AnyscaleRunsArtifactRepositoryClass = (
            _create_anyscale_runs_artifact_repository_class()
        )
    return _AnyscaleRunsArtifactRepositoryClass(*args, **kwargs)
