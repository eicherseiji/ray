from typing import Optional

import openlineage.client.facet_v2 as ol_facets

from ray.anyscale.lineage.common.facets import run as run_facets


class RunFacetConstructor:
    @staticmethod
    def construct_environment_variables_run_facet() -> dict[
        str, ol_facets.environment_variables_run.EnvironmentVariablesRunFacet
    ]:
        return {
            run_facets.ENVIRONMENT_VARIABLES_RUN_FACET_KEY: run_facets.create_environment_variables_run_facet(
                env_vars={},
            )
        }

    @staticmethod
    def construct_error_message_run_facet(
        message: str,
        stack_trace: Optional[str] = None,
    ) -> dict[str, ol_facets.error_message_run.ErrorMessageRunFacet]:
        return {
            run_facets.ERROR_MESSAGE_RUN_FACET_KEY: run_facets.create_error_message_run_facet(
                message=message,
                stack_trace=stack_trace,
            )
        }

    @staticmethod
    def construct_parent_run_facet(
        parent_job_namespace: str,
        parent_job_name: str,
        parent_run_id: str,
    ) -> dict[str, ol_facets.parent_run.ParentRunFacet]:
        return {
            run_facets.PARENT_RUN_FACET_KEY: run_facets.create_parent_run_facet(
                parent_job_namespace=parent_job_namespace,
                parent_job_name=parent_job_name,
                parent_run_id=parent_run_id,
            )
        }

    @staticmethod
    def construct_processing_engine_run_facet(
        engine_name: str,
        engine_version: str,
        openlineage_adapter_version: str,
    ) -> dict[str, ol_facets.processing_engine_run.ProcessingEngineRunFacet]:
        return {
            run_facets.PROCESSING_ENGINE_RUN_FACET_KEY: run_facets.create_processing_engine_run_facet(
                engine_name=engine_name,
                engine_version=engine_version,
                openlineage_adapter_version=openlineage_adapter_version,
            )
        }

    @staticmethod
    def construct_tags_run_facet() -> dict[str, ol_facets.tags_run.TagsRunFacet]:
        return {
            run_facets.TAGS_RUN_FACET_KEY: run_facets.create_tags_run_facet(
                tags={},
            )
        }
