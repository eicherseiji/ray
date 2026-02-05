import openlineage.client.facet_v2 as ol_facets

from ray.anyscale.lineage.common.facets import job as job_facets
from ray.anyscale.lineage.common.utils import (
    AnyscaleEnvironmentVariables,
    AnyscaleWorkloadTypes,
    get_anyscale_workload_id,
    get_os_env,
    get_os_version,
    get_python_version,
    get_ray_version,
)


class JobFacetConstructor:
    @staticmethod
    def construct_anyscale_workload_details_job_facet() -> dict[
        str, job_facets.AnyscaleWorkloadDetailsJobFacet
    ]:
        anyscale_workload_type = get_os_env(
            AnyscaleEnvironmentVariables.ANYSCALE_WORKLOAD_TYPE.value
        )
        if not anyscale_workload_type:
            raise ValueError("ANYSCALE_WORKLOAD_TYPE environment variable is not set")

        return {
            job_facets.ANYSCALE_WORKLOAD_DETAILS_JOB_FACET_KEY: job_facets.create_anyscale_workload_details_job_facet(
                type=AnyscaleWorkloadTypes(anyscale_workload_type.upper()),
                name=get_os_env(
                    AnyscaleEnvironmentVariables.ANYSCALE_WORKLOAD_NAME.value
                ),
                id=get_anyscale_workload_id(),
                organization_id=get_os_env(
                    AnyscaleEnvironmentVariables.ANYSCALE_ORGANIZATION_ID.value
                ),
                cloud_id=get_os_env(
                    AnyscaleEnvironmentVariables.ANYSCALE_CLOUD_ID.value
                ),
                project_id=get_os_env(
                    AnyscaleEnvironmentVariables.ANYSCALE_PROJECT_ID.value
                ),
                owner_email=get_os_env(
                    AnyscaleEnvironmentVariables.ANYSCALE_USER_EMAIL.value
                ),
                ray_version=get_ray_version(),
                python_version=get_python_version(),
                os_version=get_os_version(),
            )
        }

    @staticmethod
    def construct_ownership_job_facet() -> dict[
        str, ol_facets.ownership_job.OwnershipJobFacet
    ]:
        owner_emails = []
        anyscale_user_email = get_os_env(
            AnyscaleEnvironmentVariables.ANYSCALE_USER_EMAIL.value
        )

        if anyscale_user_email:
            owner_emails.append(anyscale_user_email)

        return {
            job_facets.OWNERSHIP_JOB_FACET_KEY: job_facets.create_ownership_job_facet(
                owner_emails=owner_emails,
            )
        }

    @staticmethod
    def construct_tags_job_facet() -> dict[str, ol_facets.tags_job.TagsJobFacet]:
        return {
            job_facets.TAGS_JOB_FACET_KEY: job_facets.create_tags_job_facet(
                tags={},
            )
        }
