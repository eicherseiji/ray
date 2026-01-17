from openlineage.client.facet_v2 import parent_run

PARENT_RUN_FACET_KEY: str = "parent"


def create_parent_run_facet(
    parent_job_namespace: str,
    parent_job_name: str,
    parent_run_id: str,
) -> parent_run.ParentRunFacet:
    return parent_run.ParentRunFacet(
        job=parent_run.Job(
            namespace=parent_job_namespace,
            name=parent_job_name,
        ),
        run=parent_run.Run(runId=parent_run_id),
    )
