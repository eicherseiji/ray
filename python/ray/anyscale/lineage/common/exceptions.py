class AnyscaleLineageBaseError(Exception):
    """Anyscale Lineage base exception.

    Do not directly raise this exception, use the more specific exceptions instead.
    """


class AnyscaleLineageClientError(AnyscaleLineageBaseError):
    """Anyscale Lineage client exception.

    Use this for raising `openlineage.client.OpenLineageClient` exceptions.
    """


class AnyscaleLineageSDKError(AnyscaleLineageBaseError):
    """Anyscale Lineage SDK exception.

    Use this for raising exceptions from the OpenLineage SDK.
    """


class AnyscaleLineageRayDataError(AnyscaleLineageBaseError):
    """Anyscale Lineage RayData exception.

    Use this for raising exceptions from the Ray Data integration.
    """


class AnyscaleLineageMLflowError(AnyscaleLineageBaseError):
    """Anyscale Lineage MLflow exception.

    Use this for raising exceptions from the MLflow integration.
    """
