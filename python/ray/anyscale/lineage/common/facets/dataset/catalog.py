from typing import Optional

from openlineage.client.facet_v2 import catalog_dataset

CATALOG_DATASET_FACET_KEY: str = "catalog"


def create_catalog_dataset_facet(
    framework: str,
    type: str,
    name: str,
    *,
    metadata_uri: Optional[str] = None,
    warehouse_uri: Optional[str] = None,
    source: Optional[str] = None,
) -> catalog_dataset.CatalogDatasetFacet:
    return catalog_dataset.CatalogDatasetFacet(
        framework=framework,
        type=type,
        name=name,
        metadataUri=metadata_uri,
        warehouseUri=warehouse_uri,
        source=source,
    )
