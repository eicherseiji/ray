from openlineage.client.facet_v2 import tags_job

TAGS_JOB_FACET_KEY: str = "tags"


def create_tags_job_facet(tags: dict[str, str]) -> tags_job.TagsJobFacet:
    return tags_job.TagsJobFacet(
        tags=[
            tags_job.TagsJobFacetFields(key=key, value=value)
            for key, value in tags.items()
        ]
    )
