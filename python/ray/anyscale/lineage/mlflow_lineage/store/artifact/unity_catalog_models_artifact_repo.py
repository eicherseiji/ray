from typing import Any, Optional, Type

_AnyscaleUnityCatalogModelsArtifactRepositoryClass: Optional[Type[Any]] = None


def _create_anyscale_unity_catalog_models_artifact_repository_class() -> Type[Any]:
    """Create the AnyscaleUnityCatalogModelsArtifactRepository class."""
    from mlflow.store.artifact.unity_catalog_models_artifact_repo import (
        UnityCatalogModelsArtifactRepository,
    )

    from ray.anyscale.lineage.mlflow_lineage.store.artifact.artifact_repo_mixin import (
        AnyscaleArtifactRepositoryMixin,
    )

    class AnyscaleUnityCatalogModelsArtifactRepository(
        AnyscaleArtifactRepositoryMixin, UnityCatalogModelsArtifactRepository
    ):
        """Anyscale implementation of MLflow Unity Catalog models artifact repository."""

        def __init__(self, artifact_uri: str, *args: Any, **kwargs: Any) -> None:
            super().__init__(artifact_uri, *args, **kwargs)

    return AnyscaleUnityCatalogModelsArtifactRepository


def AnyscaleUnityCatalogModelsArtifactRepository(
    *args: Any, **kwargs: Any
) -> Any:  # noqa: N802
    """Factory function that creates AnyscaleUnityCatalogModelsArtifactRepository instances."""
    global _AnyscaleUnityCatalogModelsArtifactRepositoryClass
    if _AnyscaleUnityCatalogModelsArtifactRepositoryClass is None:
        _AnyscaleUnityCatalogModelsArtifactRepositoryClass = (
            _create_anyscale_unity_catalog_models_artifact_repository_class()
        )
    return _AnyscaleUnityCatalogModelsArtifactRepositoryClass(*args, **kwargs)
