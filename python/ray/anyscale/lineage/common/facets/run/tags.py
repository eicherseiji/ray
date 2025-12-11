from openlineage.client.facet_v2 import tags_run

TAGS_RUN_FACET_KEY: str = "tags"


def create_tags_run_facet(tags: dict[str, str]) -> tags_run.TagsRunFacet:
    return tags_run.TagsRunFacet(
        tags=[
            tags_run.TagsRunFacetFields(key=key, value=value)
            for key, value in tags.items()
        ]
    )
