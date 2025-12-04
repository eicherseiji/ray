from typing import Any, Optional, Type

_AnyscaleLocalArtifactRepositoryClass: Optional[Type[Any]] = None


def _create_anyscale_local_artifact_repository_class() -> Type[Any]:
    """Create the AnyscaleLocalArtifactRepository class."""
    from mlflow.store.artifact.local_artifact_repo import LocalArtifactRepository

    from ray.anyscale.lineage.mlflow_lineage.store.artifact.artifact_repo_mixin import (
        AnyscaleArtifactRepositoryMixin,
    )
    from ray.anyscale.lineage.mlflow_lineage.utils import (
        StoreType,
        extract_upstream_store_uri,
    )

    class AnyscaleLocalArtifactRepository(
        AnyscaleArtifactRepositoryMixin, LocalArtifactRepository
    ):
        """Anyscale implementation of MLflow Local artifact repository."""

        def __init__(self, artifact_uri: str) -> None:
            artifact_uri = extract_upstream_store_uri(
                artifact_uri, StoreType.ARTIFACT_REPO
            )
            super().__init__(artifact_uri)

    return AnyscaleLocalArtifactRepository


def AnyscaleLocalArtifactRepository(*args: Any, **kwargs: Any) -> Any:  # noqa: N802
    """Factory function that creates AnyscaleLocalArtifactRepository instances."""
    global _AnyscaleLocalArtifactRepositoryClass
    if _AnyscaleLocalArtifactRepositoryClass is None:
        _AnyscaleLocalArtifactRepositoryClass = (
            _create_anyscale_local_artifact_repository_class()
        )
    return _AnyscaleLocalArtifactRepositoryClass(*args, **kwargs)
