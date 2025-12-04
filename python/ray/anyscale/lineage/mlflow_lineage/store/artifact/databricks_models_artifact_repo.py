from typing import Any, Optional, Type

_AnyscaleDatabricksModelsArtifactRepositoryClass: Optional[Type[Any]] = None


def _create_anyscale_databricks_models_artifact_repository_class() -> Type[Any]:
    """Create the AnyscaleDatabricksModelsArtifactRepository class."""
    from mlflow.store.artifact.databricks_models_artifact_repo import (
        DatabricksModelsArtifactRepository,
    )

    from ray.anyscale.lineage.mlflow_lineage.store.artifact.artifact_repo_mixin import (
        AnyscaleArtifactRepositoryMixin,
    )

    class AnyscaleDatabricksModelsArtifactRepository(
        AnyscaleArtifactRepositoryMixin, DatabricksModelsArtifactRepository
    ):
        """Anyscale implementation of MLflow Databricks models artifact repository."""

        def __init__(self, artifact_uri: str) -> None:
            super().__init__(artifact_uri)

    return AnyscaleDatabricksModelsArtifactRepository


def AnyscaleDatabricksModelsArtifactRepository(
    *args: Any, **kwargs: Any
) -> Any:  # noqa: N802
    """Factory function that creates AnyscaleDatabricksModelsArtifactRepository instances."""
    global _AnyscaleDatabricksModelsArtifactRepositoryClass
    if _AnyscaleDatabricksModelsArtifactRepositoryClass is None:
        _AnyscaleDatabricksModelsArtifactRepositoryClass = (
            _create_anyscale_databricks_models_artifact_repository_class()
        )
    return _AnyscaleDatabricksModelsArtifactRepositoryClass(*args, **kwargs)
