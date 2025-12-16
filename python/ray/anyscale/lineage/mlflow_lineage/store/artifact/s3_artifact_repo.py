from typing import Any, Optional, Type

_AnyscaleS3ArtifactRepositoryClass: Optional[Type[Any]] = None


def _create_anyscale_s3_artifact_repository_class() -> Type[Any]:
    """Create the AnyscaleS3ArtifactRepository class."""
    from mlflow.store.artifact.s3_artifact_repo import S3ArtifactRepository

    from ray.anyscale.lineage.mlflow_lineage.store.artifact.artifact_repo_mixin import (
        AnyscaleArtifactRepositoryMixin,
    )

    class AnyscaleS3ArtifactRepository(
        AnyscaleArtifactRepositoryMixin, S3ArtifactRepository
    ):
        """Anyscale implementation of MLflow S3 artifact repository."""

        def __init__(self, artifact_uri: str, *args: Any, **kwargs: Any) -> None:
            super().__init__(artifact_uri, *args, **kwargs)

    return AnyscaleS3ArtifactRepository


def AnyscaleS3ArtifactRepository(*args: Any, **kwargs: Any) -> Any:  # noqa: N802
    """Factory function that creates AnyscaleS3ArtifactRepository instances."""
    global _AnyscaleS3ArtifactRepositoryClass
    if _AnyscaleS3ArtifactRepositoryClass is None:
        _AnyscaleS3ArtifactRepositoryClass = (
            _create_anyscale_s3_artifact_repository_class()
        )
    return _AnyscaleS3ArtifactRepositoryClass(*args, **kwargs)
