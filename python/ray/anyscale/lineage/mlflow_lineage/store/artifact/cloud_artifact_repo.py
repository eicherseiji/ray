from typing import Any, Optional, Type

# Cache for the dynamically created class
_AnyscaleCloudArtifactRepositoryClass: Optional[Type[Any]] = None


def _create_anyscale_cloud_artifact_repository_class() -> Type[Any]:
    """Create the AnyscaleCloudArtifactRepository class."""
    from mlflow.store.artifact.cloud_artifact_repo import CloudArtifactRepository

    from ray.anyscale.lineage.mlflow_lineage.store.artifact.artifact_repo_mixin import (
        AnyscaleArtifactRepositoryMixin,
    )

    class AnyscaleCloudArtifactRepository(
        AnyscaleArtifactRepositoryMixin, CloudArtifactRepository
    ):
        """Anyscale implementation of MLflow Cloud artifact repository."""

        def __init__(self, artifact_uri: str, *args: Any, **kwargs: Any) -> None:
            super().__init__(artifact_uri, *args, **kwargs)

    return AnyscaleCloudArtifactRepository


def AnyscaleCloudArtifactRepository(*args: Any, **kwargs: Any) -> Any:  # noqa: N802
    """Factory function that creates AnyscaleCloudArtifactRepository instances."""
    global _AnyscaleCloudArtifactRepositoryClass
    if _AnyscaleCloudArtifactRepositoryClass is None:
        _AnyscaleCloudArtifactRepositoryClass = (
            _create_anyscale_cloud_artifact_repository_class()
        )
    return _AnyscaleCloudArtifactRepositoryClass(*args, **kwargs)
