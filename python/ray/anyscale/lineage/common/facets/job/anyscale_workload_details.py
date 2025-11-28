from typing import ClassVar, Optional

import attr
from openlineage.client.facet_v2 import JobFacet

from ray.anyscale.lineage.common.constants import REPO_URL
from ray.anyscale.lineage.common.utils import AnyscaleWorkloadTypes

ANYSCALE_WORKLOAD_DETAILS_JOB_FACET_KEY: str = "anyscaleWorkloadDetails"


@attr.define
class AnyscaleWorkloadDetailsJobFacet(JobFacet):
    """Job facet containing Anyscale workload details."""

    type: str
    """workload type - one of workspace, job, service"""

    name: str
    """workload name"""

    id: str
    """workload id"""

    organization_id: str
    """organization id"""

    cloud_id: str
    """cloud id"""

    project_id: str
    """project id"""

    owner_email: Optional[str] = None
    """owner email"""

    ray_version: Optional[str] = None
    """Ray version"""

    python_version: Optional[str] = None
    """Python version"""

    os_version: Optional[str] = None
    """OS version"""

    _additional_skip_redact: ClassVar[list[str]] = [
        "type",
        "name",
        "id",
        "organization_id",
        "cloud_id",
        "project_id",
        "owner_email",
        "ray_version",
        "python_version",
        "os_version",
    ]

    @staticmethod
    def _get_schema() -> str:
        return (
            f"{REPO_URL}/blob/main/lineage/common/facets/job/anyscale_workload_details"
        )


def create_anyscale_workload_details_job_facet(
    type: AnyscaleWorkloadTypes,
    name: str,
    id: str,
    organization_id: str,
    cloud_id: str,
    project_id: str,
    owner_email: Optional[str] = None,
    ray_version: Optional[str] = None,
    python_version: Optional[str] = None,
    os_version: Optional[str] = None,
) -> AnyscaleWorkloadDetailsJobFacet:
    return AnyscaleWorkloadDetailsJobFacet(
        type=type.value.lower(),
        name=name,
        id=id,
        organization_id=organization_id,
        cloud_id=cloud_id,
        project_id=project_id,
        owner_email=owner_email,
        ray_version=ray_version,
        python_version=python_version,
        os_version=os_version,
    )
