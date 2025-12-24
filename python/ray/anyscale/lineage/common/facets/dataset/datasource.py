from typing import Optional

from openlineage.client.facet_v2 import datasource_dataset

DATA_SOURCE_DATASET_FACET_KEY: str = "dataSource"


def create_datasource_dataset_facet(
    name: Optional[str] = None,
    uri: Optional[str] = None,
) -> datasource_dataset.DatasourceDatasetFacet:
    return datasource_dataset.DatasourceDatasetFacet(name=name, uri=uri)
