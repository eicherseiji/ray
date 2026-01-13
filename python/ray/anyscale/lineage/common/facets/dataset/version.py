from openlineage.client.facet_v2 import dataset_version_dataset

VERSION_DATASET_FACET_KEY: str = "version"


def create_version_dataset_facet(
    version: str,
) -> dataset_version_dataset.DatasetVersionDatasetFacet:
    return dataset_version_dataset.DatasetVersionDatasetFacet(datasetVersion=version)
