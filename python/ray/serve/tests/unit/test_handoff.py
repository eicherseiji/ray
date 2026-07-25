import pickle
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI

import ray.serve._private.handoff as handoff_module
from ray.serve import HandOff


@pytest.mark.asyncio
async def test_honor_stages_request_and_returns_wire_decision():
    actor = MagicMock()
    actor.handle_request.remote = AsyncMock(return_value=True)
    handoff_module._TARGET_ACTORS.clear()
    with patch.object(handoff_module.ray, "get_actor", return_value=actor):
        wire = await handoff_module.honor(HandOff(target="rep-1", request=b'{"x": 1}'))

    assert wire["replica_id"] == "rep-1"
    assert wire["handoff_key"].startswith("rep-1-")
    pickled_rm, body, key = actor.handle_request.remote.call_args.args
    rm = pickle.loads(pickled_rm)
    assert rm.call_method == "stage"
    assert body == b'{"x": 1}'
    assert key == wire["handoff_key"]


@pytest.mark.asyncio
async def test_honor_without_request_skips_staging():
    handoff_module._TARGET_ACTORS.clear()
    with patch.object(handoff_module.ray, "get_actor") as get_actor:
        wire = await handoff_module.honor(HandOff(target="rep-1"))
    get_actor.assert_not_called()
    assert wire["replica_id"] == "rep-1"
    assert wire["handoff_key"].startswith("rep-1-")


@pytest.mark.asyncio
async def test_honor_uses_provided_key():
    handoff_module._TARGET_ACTORS.clear()
    with patch.object(
        handoff_module.ray,
        "get_actor",
        return_value=MagicMock(handle_request=MagicMock(remote=AsyncMock())),
    ):
        wire = await handoff_module.honor(
            HandOff(target="rep-1", request=b"{}", key="fixed")
        )
    assert wire == {"replica_id": "rep-1", "handoff_key": "fixed"}


@pytest.mark.asyncio
async def test_honor_via_handle_delivers_to_target_and_returns_response():
    # The handle-call path: honoring invokes the target replica with the request
    # and returns its response unchanged, so the returning deployment is off the
    # response path.
    actor = MagicMock()
    actor.handle_request.remote = AsyncMock(return_value="target-response")
    handoff_module._TARGET_ACTORS.clear()
    with patch.object(handoff_module.ray, "get_actor", return_value=actor):
        result = await handoff_module.honor_via_handle(
            HandOff(target="rep-1", request={"prompt": "hi"})
        )

    assert result == "target-response"
    pickled_rm, request = actor.handle_request.remote.call_args.args
    rm = pickle.loads(pickled_rm)
    assert rm.call_method == "__call__"
    assert request == {"prompt": "hi"}


@pytest.mark.asyncio
async def test_install_handoff_honoring_converts_handoff_return():
    app = FastAPI()

    @app.post("/route")
    async def route():
        return HandOff(target="rep-1", request=b"{}")

    @app.post("/plain")
    async def plain():
        return {"ok": True}

    handoff_module.install_handoff_honoring(app)

    routes = {r.path: r for r in app.routes if getattr(r, "path", None)}
    handoff_module._TARGET_ACTORS.clear()
    actor = MagicMock()
    actor.handle_request.remote = AsyncMock(return_value=True)
    with patch.object(handoff_module.ray, "get_actor", return_value=actor):
        handoff_result = await routes["/route"].dependant.call()
    # A HandOff return is converted to the wire decision (and staged).
    assert handoff_result["replica_id"] == "rep-1"
    assert "handoff_key" in handoff_result
    actor.handle_request.remote.assert_awaited_once()

    # A normal return is passed through untouched.
    plain_result = await routes["/plain"].dependant.call()
    assert plain_result == {"ok": True}


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
