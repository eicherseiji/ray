from typing import Any, Optional, Type

_AnyscaleArtifactRepositoryClass: Optional[Type[Any]] = None


def _create_anyscale_artifact_repository_class() -> Type[Any]:
    """Create the AnyscaleArtifactRepository class."""
    # Import mlflow and mlflow_openlineage dependencies here to avoid
    # circular imports during plugin registration.
    from mlflow.store.artifact.artifact_repo import ArtifactRepository

    from ray.anyscale.lineage.mlflow_lineage.store.artifact.artifact_repo_mixin import (
        AnyscaleArtifactRepositoryMixin,
    )

    class AnyscaleArtifactRepository(
        AnyscaleArtifactRepositoryMixin, ArtifactRepository
    ):
        """Anyscale implementation of MLflow artifact repository."""

        def __init__(self, artifact_uri: str) -> None:
            super().__init__(artifact_uri)

    return AnyscaleArtifactRepository


def AnyscaleArtifactRepository(*args: Any, **kwargs: Any) -> Any:  # noqa: N802
    """Factory function that creates AnyscaleArtifactRepository instances."""
    global _AnyscaleArtifactRepositoryClass
    if _AnyscaleArtifactRepositoryClass is None:
        _AnyscaleArtifactRepositoryClass = _create_anyscale_artifact_repository_class()
    return _AnyscaleArtifactRepositoryClass(*args, **kwargs)
