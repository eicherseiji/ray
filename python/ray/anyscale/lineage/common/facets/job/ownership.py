from typing import List

from openlineage.client.facet_v2 import ownership_job

OWNERSHIP_JOB_FACET_KEY: str = "ownership"


def create_ownership_job_facet(
    owner_emails: List[str],
) -> ownership_job.OwnershipJobFacet:
    return ownership_job.OwnershipJobFacet(
        owners=[
            ownership_job.Owner(name=owner_email, type="owner")
            for owner_email in owner_emails
        ]
    )
