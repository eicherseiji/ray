from typing import Any, Optional, Type

_AnyscaleMLflowArtifactsRepositoryClass: Optional[Type[Any]] = None


def _create_anyscale_mlflow_artifacts_repository_class() -> Type[Any]:
    """Create the AnyscaleMLflowArtifactsRepository class."""
    from mlflow.store.artifact.mlflow_artifacts_repo import MlflowArtifactsRepository

    from ray.anyscale.lineage.mlflow_lineage.store.artifact.artifact_repo_mixin import (
        AnyscaleArtifactRepositoryMixin,
    )

    class AnyscaleMLflowArtifactsRepository(
        AnyscaleArtifactRepositoryMixin, MlflowArtifactsRepository
    ):
        """Anyscale implementation of MLflow MLflow artifacts repository."""

        def __init__(self, artifact_uri: str) -> None:
            super().__init__(artifact_uri)

    return AnyscaleMLflowArtifactsRepository


def AnyscaleMLflowArtifactsRepository(*args: Any, **kwargs: Any) -> Any:  # noqa: N802
    """Factory function that creates AnyscaleMLflowArtifactsRepository instances."""
    global _AnyscaleMLflowArtifactsRepositoryClass
    if _AnyscaleMLflowArtifactsRepositoryClass is None:
        _AnyscaleMLflowArtifactsRepositoryClass = (
            _create_anyscale_mlflow_artifacts_repository_class()
        )
    return _AnyscaleMLflowArtifactsRepositoryClass(*args, **kwargs)
