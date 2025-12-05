from ray.anyscale.lineage.mlflow_lineage import run_context_provider


def test_run_context_in_context(monkeypatch) -> None:
    # Set TRACKING_ENABLED to True for this test
    monkeypatch.setattr(run_context_provider, "TRACKING_ENABLED", True)

    # Reset the cached class to pick up the monkeypatch
    run_context_provider._AnyscaleRunContextProviderClass = None

    # Instantiate the provider using the factory function
    provider = run_context_provider.AnyscaleRunContextProvider()
    assert provider.in_context() is True


def test_run_context_tags(monkeypatch) -> None:
    monkeypatch.setenv("ANYSCALE_WORKSPACE_ID", "workspace-9")

    # Instantiate the provider using the factory function
    provider = run_context_provider.AnyscaleRunContextProvider()
    tags = provider.tags()

    assert tags["ANYSCALE_WORKSPACE_ID"] == "workspace-9"


def test_run_context_tags_all_variables(monkeypatch) -> None:
    """Test that all supported environment variables are included when set."""
    env_vars = {
        "ANYSCALE_JOB_ID": "job-3",
        "ANYSCALE_PROJECT_ID": "project-5",
        "ANYSCALE_SERVICE_ID": "service-6",
        "ANYSCALE_WORKLOAD_NAME": "test-workload",
        "ANYSCALE_WORKLOAD_TYPE": "job",
        "ANYSCALE_WORKSPACE_ID": "workspace-7",
    }

    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)

    # Instantiate the provider using the factory function
    provider = run_context_provider.AnyscaleRunContextProvider()
    tags = provider.tags()

    assert len(tags) == len(env_vars)
    for key, expected_value in env_vars.items():
        assert tags[key] == expected_value
