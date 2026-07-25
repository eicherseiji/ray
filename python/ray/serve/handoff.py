from dataclasses import dataclass
from typing import Any, Optional

from ray.util.annotations import PublicAPI


@PublicAPI(stability="alpha")
@dataclass
class HandOff:
    """Response-delegation sentinel a deployment returns instead of a response.

    Instead of producing the response itself, a deployment does its request-side
    work and returns a ``HandOff`` naming the ``target`` that will produce the
    response and (optionally) the ``request`` to stage there. Serve honors the
    returned ``HandOff``: it stages the request at the target and delivers the
    response from the target, so the deployment that returned the ``HandOff`` is
    off the response path.

    This is a general Serve primitive: the target can be any deployment/replica
    the proxy can route to (an LLM leaf, a decode worker, etc.).

    Example:
        @serve.deployment
        class Front:
            async def __call__(self, request):
                target = self._pick(request)
                return serve.HandOff(target=target, request=await request.body())

    Attributes:
        target: Replica id the proxy routes the response from.
        request: The (possibly transformed) request Serve stages at the target.
            If ``None``, nothing is staged (the target serves the spliced request).
        key: Rendezvous key; Serve generates one when omitted.
    """

    target: str
    request: Optional[Any] = None
    key: Optional[str] = None
