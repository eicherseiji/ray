from typing import Any, Optional, Type

_AnyscaleGCSArtifactRepositoryClass: Optional[Type[Any]] = None


def _create_anyscale_gcs_artifact_repository_class() -> Type[Any]:
    """Create the AnyscaleGCSArtifactRepository class."""
    from mlflow.store.artifact.gcs_artifact_repo import GCSArtifactRepository

    from ray.anyscale.lineage.mlflow_lineage.store.artifact.artifact_repo_mixin import (
        AnyscaleArtifactRepositoryMixin,
    )

    class AnyscaleGCSArtifactRepository(
        AnyscaleArtifactRepositoryMixin, GCSArtifactRepository
    ):
        """Anyscale implementation of MLflow GCS artifact repository."""

        def __init__(self, artifact_uri: str, *args: Any, **kwargs: Any) -> None:
            super().__init__(artifact_uri, *args, **kwargs)

    return AnyscaleGCSArtifactRepository


def AnyscaleGCSArtifactRepository(*args: Any, **kwargs: Any) -> Any:  # noqa: N802
    """Factory function that creates AnyscaleGCSArtifactRepository instances."""
    global _AnyscaleGCSArtifactRepositoryClass
    if _AnyscaleGCSArtifactRepositoryClass is None:
        _AnyscaleGCSArtifactRepositoryClass = (
            _create_anyscale_gcs_artifact_repository_class()
        )
    return _AnyscaleGCSArtifactRepositoryClass(*args, **kwargs)
