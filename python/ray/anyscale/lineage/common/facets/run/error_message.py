from typing import Optional

from openlineage.client.facet_v2 import error_message_run

ERROR_MESSAGE_RUN_FACET_KEY: str = "errorMessage"
ERROR_MESSAGE_PROGRAMMING_LANGUAGE: str = "python"


def create_error_message_run_facet(
    message: str, stack_trace: Optional[str] = None
) -> error_message_run.ErrorMessageRunFacet:
    return error_message_run.ErrorMessageRunFacet(
        message=message,
        stackTrace=stack_trace,
        programmingLanguage=ERROR_MESSAGE_PROGRAMMING_LANGUAGE,
    )
