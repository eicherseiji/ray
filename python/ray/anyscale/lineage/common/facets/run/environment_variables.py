from openlineage.client.facet_v2 import environment_variables_run

ENVIRONMENT_VARIABLES_RUN_FACET_KEY: str = "environmentVariables"


def create_environment_variables_run_facet(
    env_vars: dict[str, str],
) -> environment_variables_run.EnvironmentVariablesRunFacet:
    return environment_variables_run.EnvironmentVariablesRunFacet(
        environmentVariables=[
            environment_variables_run.EnvironmentVariable(name=name, value=value)
            for name, value in env_vars.items()
        ]
    )
