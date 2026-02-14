from typing import List

from openlineage.client.facet_v2 import ownership_dataset

OWNERSHIP_DATASET_FACET_KEY: str = "ownership"


def create_ownership_dataset_facet(
    owner_emails: List[str],
) -> ownership_dataset.OwnershipDatasetFacet:
    return ownership_dataset.OwnershipDatasetFacet(
        owners=[
            ownership_dataset.Owner(name=owner_email, type="owner")
            for owner_email in owner_emails
        ]
    )
