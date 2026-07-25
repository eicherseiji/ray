"""Serve-native honoring of a ``HandOff`` returned by a deployment handler.

A :class:`~ray.serve.handoff.HandOff` names a target replica and the request to
serve there. Serve honors it so the returning deployment is off the response
path, in one of two ways depending on how the deployment was invoked:

- Unary handle call (``honor_via_handle``): Serve delivers the request to the
  target replica and returns its response as the call's result. The returning
  deployment does no response work. This makes ``return serve.HandOff(...)``
  honest for any ``@serve.deployment`` reached via a ``DeploymentHandle``.

- FastAPI ingress route (``honor`` + ``install_handoff_honoring``): FastAPI
  consumes the route's return value before it reaches the replica return path,
  so honoring wraps the route callable. It stages the request at the target and
  returns the HAProxy wire decision (``{replica_id, handoff_key}``); the leaf
  serves the staged request over the native splice (``x-handoff-key``).

Both share the target-actor lookup; only the delivery mechanism differs.
"""

import functools
import inspect
import pickle
import uuid

import ray
from ray.serve._private.common import RequestMetadata
from ray.serve.handoff import HandOff

# replica_id -> pinned actor handle, memoized per process.
_TARGET_ACTORS: dict = {}


def _target_actor(target: str):
    """Resolve and memoize the replica actor handle for ``target``."""
    actor = _TARGET_ACTORS.get(target)
    if actor is None:
        actor = ray.get_actor(target, namespace="serve")
        _TARGET_ACTORS[target] = actor
    return actor


async def _stage_at_target(target: str, request, key: str) -> None:
    """Stage ``request`` at ``target`` via a pinned call to its ``stage`` method."""
    body = request if isinstance(request, (bytes, bytearray)) else str(request).encode()
    rm = RequestMetadata(request_id=key, internal_request_id=key, call_method="stage")
    await _target_actor(target).handle_request.remote(pickle.dumps(rm), body, key)


async def honor(handoff: HandOff) -> dict:
    """Stage the handoff's request at its target and return the wire decision."""
    key = handoff.key or f"{handoff.target}-{uuid.uuid4().hex[:12]}"
    if handoff.request is not None:
        await _stage_at_target(handoff.target, handoff.request, key)
    return {"replica_id": handoff.target, "handoff_key": key}


async def honor_via_handle(handoff: HandOff, call_method: str = "__call__"):
    """Deliver the handoff's request to its target replica and return the response.

    Used on the unary handle-call path, where the caller awaits a real response
    rather than a wire decision. The target replica serves ``handoff.request`` on
    ``call_method`` and its result is returned unchanged, so the returning
    deployment contributes nothing to the response.
    """
    key = handoff.key or f"{handoff.target}-{uuid.uuid4().hex[:12]}"
    rm = RequestMetadata(
        request_id=key, internal_request_id=key, call_method=call_method
    )
    return await _target_actor(handoff.target).handle_request.remote(
        pickle.dumps(rm), handoff.request
    )


def _honoring_call(call):
    """Wrap a route callable so a ``HandOff`` return is honored, else passed through."""

    @functools.wraps(call)
    async def wrapper(*args, **kwargs):
        result = call(*args, **kwargs)
        if inspect.isawaitable(result):
            result = await result
        if isinstance(result, HandOff):
            # Return a plain dict; FastAPI serializes it to the JSON the proxy reads.
            return await honor(result)
        return result

    return wrapper


def install_handoff_honoring(app) -> None:
    """Make every route on ``app`` honor a ``HandOff`` return value.

    A no-op for routes that return normal responses, so existing ingresses are
    unaffected. Called by ``serve.ingress`` for FastAPI/APIRouter apps.
    """
    from fastapi.routing import APIRoute

    for route in getattr(app, "routes", []):
        if (
            isinstance(route, APIRoute)
            and getattr(route, "dependant", None) is not None
        ):
            call = route.dependant.call
            if call is not None and not getattr(call, "_serve_handoff_wrapped", False):
                wrapped = _honoring_call(call)
                wrapped._serve_handoff_wrapped = True
                route.dependant.call = wrapped
