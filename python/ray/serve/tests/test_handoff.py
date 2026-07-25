"""Replica-level honoring of a serve.HandOff return value.

A deployment that returns serve.HandOff(target, request) is honored by Serve:
the request is delivered to the target replica and its response is returned, so
the returning deployment is off the response path. These tests cover the two
non-FastAPI invocation paths (a handle-called method and a vanilla HTTP ingress);
the FastAPI-ingress path is covered by the ingress-request-router tests.

Honoring is gated by RAY_SERVE_ENABLE_HANDOFF, read at import time, so each test
sets it on the workers via runtime_env.
"""
import sys

import httpx
import pytest
from starlette.requests import Request

import ray
from ray import serve
from ray.serve.handoff import HandOff

HANDOFF_RUNTIME_ENV = {"env_vars": {"RAY_SERVE_ENABLE_HANDOFF": "1"}}


@serve.deployment
class Backend:
    def replica_id(self) -> str:
        return serve.get_replica_context().replica_id.to_full_id_str()

    async def __call__(self, payload):
        return {"served_by": "backend", "payload": payload}


@serve.deployment
class HandleFront:
    def __init__(self, backend):
        self._backend = backend

    async def __call__(self, payload):
        target = await self._backend.replica_id.remote()
        return HandOff(target=target, request=payload)


@serve.deployment
class HTTPFront:
    def __init__(self, backend):
        self._backend = backend

    async def __call__(self, request: Request):
        body = await request.json()
        target = await self._backend.replica_id.remote()
        return HandOff(target=target, request=body)


def test_handle_called_handoff_is_honored(ray_shutdown):
    # A plain deployment returning HandOff, reached via a DeploymentHandle, is
    # delivered from the target; the caller never sees a HandOff object.
    ray.init(namespace="serve", runtime_env=HANDOFF_RUNTIME_ENV)
    handle = serve.run(HandleFront.bind(Backend.bind()))

    result = handle.remote({"n": 1}).result()

    assert result == {"served_by": "backend", "payload": {"n": 1}}
    assert not isinstance(result, HandOff)


def test_vanilla_http_handoff_is_honored(ray_shutdown):
    # A vanilla (non-FastAPI) HTTP ingress returning HandOff delivers the target's
    # response as the HTTP body, not a serialized HandOff.
    ray.init(namespace="serve", runtime_env=HANDOFF_RUNTIME_ENV)
    serve.run(HTTPFront.bind(Backend.bind()))

    resp = httpx.post("http://127.0.0.1:8000/", json={"n": 7}, timeout=15)

    assert resp.status_code == 200
    assert resp.json() == {"served_by": "backend", "payload": {"n": 7}}


def test_handoff_is_gated_off_by_default(ray_shutdown):
    # Without the flag, a returned HandOff is passed through unchanged (no-op), so
    # the feature is opt-in and normal deployments are unaffected.
    ray.init(namespace="serve")
    handle = serve.run(HandleFront.bind(Backend.bind()))

    result = handle.remote({"n": 1}).result()

    assert isinstance(result, HandOff)
    assert result.request == {"n": 1}


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", "-s", __file__]))
