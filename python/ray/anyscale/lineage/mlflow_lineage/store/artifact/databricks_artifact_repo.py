from typing import Any, Optional, Type

_AnyscaleDatabricksArtifactRepositoryClass: Optional[Type[Any]] = None


def _create_anyscale_databricks_artifact_repository_class() -> Type[Any]:
    """Create the AnyscaleDatabricksArtifactRepository class."""
    from mlflow.store.artifact.databricks_artifact_repo import (
        DatabricksArtifactRepository,
    )

    from ray.anyscale.lineage.mlflow_lineage.store.artifact.artifact_repo_mixin import (
        AnyscaleArtifactRepositoryMixin,
    )

    class AnyscaleDatabricksArtifactRepository(
        AnyscaleArtifactRepositoryMixin, DatabricksArtifactRepository
    ):
        """Anyscale implementation of MLflow Databricks artifact repository."""

        def __init__(self, artifact_uri: str, *args: Any, **kwargs: Any) -> None:
            super().__init__(artifact_uri, *args, **kwargs)

    return AnyscaleDatabricksArtifactRepository


def AnyscaleDatabricksArtifactRepository(
    *args: Any, **kwargs: Any
) -> Any:  # noqa: N802
    """Factory function that creates AnyscaleDatabricksArtifactRepository instances."""
    global _AnyscaleDatabricksArtifactRepositoryClass
    if _AnyscaleDatabricksArtifactRepositoryClass is None:
        _AnyscaleDatabricksArtifactRepositoryClass = (
            _create_anyscale_databricks_artifact_repository_class()
        )
    return _AnyscaleDatabricksArtifactRepositoryClass(*args, **kwargs)
