from typing import Any, Optional, Type

_AnyscaleHTTPArtifactRepositoryClass: Optional[Type[Any]] = None


def _create_anyscale_http_artifact_repository_class() -> Type[Any]:
    """Create the AnyscaleHTTPArtifactRepository class."""
    from mlflow.store.artifact.http_artifact_repo import HttpArtifactRepository

    from ray.anyscale.lineage.mlflow_lineage.store.artifact.artifact_repo_mixin import (
        AnyscaleArtifactRepositoryMixin,
    )

    class AnyscaleHTTPArtifactRepository(
        AnyscaleArtifactRepositoryMixin, HttpArtifactRepository
    ):
        """Anyscale implementation of MLflow HTTP artifact repository."""

        def __init__(self, artifact_uri: str, *args: Any, **kwargs: Any) -> None:
            super().__init__(artifact_uri, *args, **kwargs)

    return AnyscaleHTTPArtifactRepository


def AnyscaleHTTPArtifactRepository(*args: Any, **kwargs: Any) -> Any:  # noqa: N802
    """Factory function that creates AnyscaleHTTPArtifactRepository instances."""
    global _AnyscaleHTTPArtifactRepositoryClass
    if _AnyscaleHTTPArtifactRepositoryClass is None:
        _AnyscaleHTTPArtifactRepositoryClass = (
            _create_anyscale_http_artifact_repository_class()
        )
    return _AnyscaleHTTPArtifactRepositoryClass(*args, **kwargs)
