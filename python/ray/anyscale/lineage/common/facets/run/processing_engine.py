from typing import Optional

from openlineage.client.facet_v2 import processing_engine_run

PROCESSING_ENGINE_RUN_FACET_KEY: str = "processing_engine"


def create_processing_engine_run_facet(
    engine_name: str,
    engine_version: Optional[str] = None,
    openlineage_adapter_version: Optional[str] = None,
) -> processing_engine_run.ProcessingEngineRunFacet:
    return processing_engine_run.ProcessingEngineRunFacet(
        name=engine_name,
        version=engine_version or "",
        openlineageAdapterVersion=openlineage_adapter_version,
    )
