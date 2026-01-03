from typing import Any, Optional, Type

_AnyscaleOptimizedS3ArtifactRepositoryClass: Optional[Type[Any]] = None


def _create_anyscale_optimized_s3_artifact_repository_class() -> Type[Any]:
    """Create the AnyscaleOptimizedS3ArtifactRepository class."""
    from mlflow.store.artifact.optimized_s3_artifact_repo import (
        OptimizedS3ArtifactRepository,
    )

    from ray.anyscale.lineage.mlflow_lineage.store.artifact.artifact_repo_mixin import (
        AnyscaleArtifactRepositoryMixin,
    )

    class AnyscaleOptimizedS3ArtifactRepository(
        AnyscaleArtifactRepositoryMixin, OptimizedS3ArtifactRepository
    ):
        """Anyscale implementation of MLflow Optimized S3 artifact repository."""

        def __init__(self, artifact_uri: str, *args: Any, **kwargs: Any) -> None:
            super().__init__(artifact_uri, *args, **kwargs)

    return AnyscaleOptimizedS3ArtifactRepository


def AnyscaleOptimizedS3ArtifactRepository(
    *args: Any, **kwargs: Any
) -> Any:  # noqa: N802
    """Factory function that creates AnyscaleOptimizedS3ArtifactRepository instances."""
    global _AnyscaleOptimizedS3ArtifactRepositoryClass
    if _AnyscaleOptimizedS3ArtifactRepositoryClass is None:
        _AnyscaleOptimizedS3ArtifactRepositoryClass = (
            _create_anyscale_optimized_s3_artifact_repository_class()
        )
    return _AnyscaleOptimizedS3ArtifactRepositoryClass(*args, **kwargs)
