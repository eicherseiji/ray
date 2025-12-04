from typing import Any, Optional, Type

_AnyscaleFTPArtifactRepositoryClass: Optional[Type[Any]] = None


def _create_anyscale_ftp_artifact_repository_class() -> Type[Any]:
    """Create the AnyscaleFTPArtifactRepository class."""
    from mlflow.store.artifact.ftp_artifact_repo import FTPArtifactRepository

    from ray.anyscale.lineage.mlflow_lineage.store.artifact.artifact_repo_mixin import (
        AnyscaleArtifactRepositoryMixin,
    )

    class AnyscaleFTPArtifactRepository(
        AnyscaleArtifactRepositoryMixin, FTPArtifactRepository
    ):
        """Anyscale implementation of MLflow FTP artifact repository."""

        def __init__(self, artifact_uri: str) -> None:
            super().__init__(artifact_uri)

    return AnyscaleFTPArtifactRepository


def AnyscaleFTPArtifactRepository(*args: Any, **kwargs: Any) -> Any:  # noqa: N802
    """Factory function that creates AnyscaleFTPArtifactRepository instances."""
    global _AnyscaleFTPArtifactRepositoryClass
    if _AnyscaleFTPArtifactRepositoryClass is None:
        _AnyscaleFTPArtifactRepositoryClass = (
            _create_anyscale_ftp_artifact_repository_class()
        )
    return _AnyscaleFTPArtifactRepositoryClass(*args, **kwargs)
