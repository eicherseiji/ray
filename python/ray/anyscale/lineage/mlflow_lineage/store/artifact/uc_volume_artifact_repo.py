from typing import Any, Optional, Type

_AnyscaleUCVolumeArtifactRepositoryClass: Optional[Type[Any]] = None


def _create_anyscale_uc_volume_artifact_repository_class() -> Type[Any]:
    """Create the AnyscaleUCVolumeArtifactRepository class."""
    from mlflow.store.artifact.uc_volume_artifact_repo import (
        UCVolumesArtifactRepository,
    )

    from ray.anyscale.lineage.mlflow_lineage.store.artifact.artifact_repo_mixin import (
        AnyscaleArtifactRepositoryMixin,
    )

    class AnyscaleUCVolumeArtifactRepository(
        AnyscaleArtifactRepositoryMixin, UCVolumesArtifactRepository
    ):
        """Anyscale implementation of MLflow UC Volume artifact repository."""

        def __init__(self, artifact_uri: str) -> None:
            super().__init__(artifact_uri)

    return AnyscaleUCVolumeArtifactRepository


def AnyscaleUCVolumeArtifactRepository(*args: Any, **kwargs: Any) -> Any:  # noqa: N802
    """Factory function that creates AnyscaleUCVolumeArtifactRepository instances."""
    global _AnyscaleUCVolumeArtifactRepositoryClass
    if _AnyscaleUCVolumeArtifactRepositoryClass is None:
        _AnyscaleUCVolumeArtifactRepositoryClass = (
            _create_anyscale_uc_volume_artifact_repository_class()
        )
    return _AnyscaleUCVolumeArtifactRepositoryClass(*args, **kwargs)
