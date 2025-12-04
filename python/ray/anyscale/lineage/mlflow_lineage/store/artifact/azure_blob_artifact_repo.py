from typing import Any, Optional, Type

_AnyscaleAzureBlobArtifactRepositoryClass: Optional[Type[Any]] = None


def _create_anyscale_azure_blob_artifact_repository_class() -> Type[Any]:
    """Create the AnyscaleAzureBlobArtifactRepository class."""
    from mlflow.store.artifact.azure_blob_artifact_repo import (
        AzureBlobArtifactRepository,
    )

    from ray.anyscale.lineage.mlflow_lineage.store.artifact.artifact_repo_mixin import (
        AnyscaleArtifactRepositoryMixin,
    )

    class AnyscaleAzureBlobArtifactRepository(
        AnyscaleArtifactRepositoryMixin, AzureBlobArtifactRepository
    ):
        """Anyscale implementation of MLflow Azure Blob artifact repository."""

        def __init__(self, artifact_uri: str, client: Optional[Any] = None) -> None:
            super().__init__(artifact_uri, client)

    return AnyscaleAzureBlobArtifactRepository


def AnyscaleAzureBlobArtifactRepository(*args: Any, **kwargs: Any) -> Any:  # noqa: N802
    """Factory function that creates AnyscaleAzureBlobArtifactRepository instances."""
    global _AnyscaleAzureBlobArtifactRepositoryClass
    if _AnyscaleAzureBlobArtifactRepositoryClass is None:
        _AnyscaleAzureBlobArtifactRepositoryClass = (
            _create_anyscale_azure_blob_artifact_repository_class()
        )
    return _AnyscaleAzureBlobArtifactRepositoryClass(*args, **kwargs)
