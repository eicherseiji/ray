from typing import Any, Optional, Type

_AnyscaleModelsArtifactRepositoryClass: Optional[Type[Any]] = None


def _create_anyscale_models_artifact_repository_class() -> Type[Any]:
    """Create the AnyscaleModelsArtifactRepository class."""
    from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository

    from ray.anyscale.lineage.mlflow_lineage.store.artifact.artifact_repo_mixin import (
        AnyscaleArtifactRepositoryMixin,
    )

    class AnyscaleModelsArtifactRepository(
        AnyscaleArtifactRepositoryMixin, ModelsArtifactRepository
    ):
        """Anyscale implementation of MLflow Models artifact repository."""

        def __init__(self, artifact_uri: str, *args: Any, **kwargs: Any) -> None:
            super().__init__(artifact_uri, *args, **kwargs)

    return AnyscaleModelsArtifactRepository


def AnyscaleModelsArtifactRepository(*args: Any, **kwargs: Any) -> Any:  # noqa: N802
    """Factory function that creates AnyscaleModelsArtifactRepository instances."""
    global _AnyscaleModelsArtifactRepositoryClass
    if _AnyscaleModelsArtifactRepositoryClass is None:
        _AnyscaleModelsArtifactRepositoryClass = (
            _create_anyscale_models_artifact_repository_class()
        )
    return _AnyscaleModelsArtifactRepositoryClass(*args, **kwargs)
