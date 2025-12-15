from openlineage.client.facet_v2 import tags_dataset

TAGS_DATASET_FACET_KEY: str = "tags"


def create_tags_dataset_facet(tags: dict[str, str]) -> tags_dataset.TagsDatasetFacet:
    return tags_dataset.TagsDatasetFacet(
        tags=[
            tags_dataset.TagsDatasetFacetFields(key=key, value=value)
            for key, value in tags.items()
        ]
    )
