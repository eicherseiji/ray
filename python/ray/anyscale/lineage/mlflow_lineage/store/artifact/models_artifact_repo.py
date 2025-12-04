from typing import Any, Optional, Type

_AnyscaleModelsArtifactRepositoryClass: Optional[Type[Any]] = None


def _create_anyscale_models_artifact_repository_class() -> Type[Any]:
    """Create the AnyscaleModelsArtifactRepository class."""
    from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository

    from ray.anyscale.lineage.mlflow_lineage.store.artifact.artifact_repo_mixin import (
        AnyscaleArtifactRepositoryMixin,
    )
    from ray.anyscale.lineage.mlflow_lineage.utils import (
        StoreType,
        extract_upstream_store_uri,
    )

    class AnyscaleModelsArtifactRepository(
        AnyscaleArtifactRepositoryMixin, ModelsArtifactRepository
    ):
        """Anyscale implementation of MLflow Models artifact repository."""

        def __init__(self, artifact_uri: str) -> None:
            artifact_uri = extract_upstream_store_uri(
                artifact_uri, StoreType.ARTIFACT_REPO
            )
            super().__init__(artifact_uri)

    return AnyscaleModelsArtifactRepository


def AnyscaleModelsArtifactRepository(*args: Any, **kwargs: Any) -> Any:  # noqa: N802
    """Factory function that creates AnyscaleModelsArtifactRepository instances."""
    global _AnyscaleModelsArtifactRepositoryClass
    if _AnyscaleModelsArtifactRepositoryClass is None:
        _AnyscaleModelsArtifactRepositoryClass = (
            _create_anyscale_models_artifact_repository_class()
        )
    return _AnyscaleModelsArtifactRepositoryClass(*args, **kwargs)
