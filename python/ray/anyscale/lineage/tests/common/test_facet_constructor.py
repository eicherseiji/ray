from unittest import mock

import pytest

from ray.anyscale.lineage.common import facet_constructor
from ray.anyscale.lineage.tests.test_constants import (
    TEST_ORG_ID,
    TEST_OWNER_EMAIL,
    TEST_USER_EMAIL,
    TEST_WORKLOAD_NAME,
)


@pytest.fixture
def job_facet_constructor():
    return facet_constructor.JobFacetConstructor()


@pytest.fixture
def dataset_facet_constructor():
    return facet_constructor.DatasetFacetConstructor()


class TestJobFacets:
    def test_construct_anyscale_workload_details_job_facet(
        self, job_facet_constructor, sample_anyscale_env
    ):
        with mock.patch.multiple(
            "ray.anyscale.lineage.common.facet_constructor.job_facet_constructor",
            get_anyscale_workload_id=mock.Mock(return_value="resolved-id"),
            get_ray_version=mock.Mock(return_value="2.7.0"),
            get_python_version=mock.Mock(return_value="3.11"),
            get_os_version=mock.Mock(return_value="linux"),
        ):
            facets = (
                job_facet_constructor.construct_anyscale_workload_details_job_facet()
            )

        facet = facets["anyscaleWorkloadDetails"]
        assert facet.type == "job"
        assert facet.name == TEST_WORKLOAD_NAME
        assert facet.id == "resolved-id"
        assert facet.organization_id == TEST_ORG_ID
        assert facet.owner_email == TEST_USER_EMAIL

    def test_construct_ownership_job_facet(self, job_facet_constructor, monkeypatch):
        monkeypatch.setenv("ANYSCALE_USER_EMAIL", TEST_OWNER_EMAIL)

        facets = job_facet_constructor.construct_ownership_job_facet()

        owners = facets["ownership"].owners
        assert len(owners) == 1
        assert owners[0].name == TEST_OWNER_EMAIL

    def test_construct_ownership_job_facet_no_email(
        self, job_facet_constructor, clean_environment
    ):
        facets = job_facet_constructor.construct_ownership_job_facet()

        owners = facets["ownership"].owners
        assert len(owners) == 0


class TestDatasetFacets:
    def test_construct_ownership_dataset_facet(
        self, dataset_facet_constructor, monkeypatch
    ):
        monkeypatch.setenv("ANYSCALE_USER_EMAIL", TEST_OWNER_EMAIL)

        dataset_facet = dataset_facet_constructor.construct_ownership_dataset_facet()

        owners = dataset_facet["ownership"].owners
        assert len(owners) == 1
        assert owners[0].name == TEST_OWNER_EMAIL

    def test_construct_schema_dataset_facet_with_valid_fields(
        self, dataset_facet_constructor
    ):
        schema_fields = [
            {"name": "id", "type": "integer", "description": "Primary key"},
            {"name": "name", "type": "string", "description": "User name"},
        ]

        dataset_facet = dataset_facet_constructor.construct_schema_dataset_facet(
            schema_fields
        )

        schema_fields_result = dataset_facet["schema"].fields
        assert len(schema_fields_result) == 2
        assert schema_fields_result[0].name == "id"
        assert schema_fields_result[1].name == "name"

    def test_construct_schema_dataset_facet_filters_missing_names(
        self, dataset_facet_constructor
    ):
        schema_fields = [
            {"name": "valid_field", "type": "string", "description": "Valid field"},
            {"type": "string", "description": "Missing name"},
            {"name": "", "type": "integer"},
        ]

        dataset_facet = dataset_facet_constructor.construct_schema_dataset_facet(
            schema_fields
        )

        schema_fields_result = dataset_facet["schema"].fields
        assert len(schema_fields_result) == 1
        assert schema_fields_result[0].name == "valid_field"
