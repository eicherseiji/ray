"""LLMRouter: a dedicated router deployment for ingress bypass.

When ingress bypass is enabled, HAProxy calls /internal/route on this
deployment to get a (host, port) pair, then forwards traffic directly
to that LLMServer replica's direct ingress port.
"""

import asyncio
from typing import Dict, List, Tuple

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from ray._common.utils import get_or_create_event_loop
from ray.llm._internal.common.utils.lora_utils import get_base_model_id
from ray.llm._internal.serve.core.configs.llm_config import LLMConfig
from ray.llm._internal.serve.observability.logging import get_logger
from ray.serve.api import router as serve_router
from ray.serve.handle import DeploymentHandle

logger = get_logger(__name__)

router_app = FastAPI()


@serve_router(router_app)
class LLMRouter:
    """Lightweight router deployment for ingress bypass.

    Receives /internal/route POST requests from HAProxy Lua, picks the
    least-loaded LLMServer replica via background-polled queue lengths,
    and returns its direct ingress (host, port).

    The /internal/route handler is a raw ASGI middleware (not a FastAPI
    route) to minimize per-request latency for the Lua TCP round-trip.
    """

    def __init__(self, llm_deployments: List[DeploymentHandle]):
        self._default_serve_handles: Dict[str, DeploymentHandle] = {}
        self._llm_configs: Dict[str, LLMConfig] = {}
        self._rr_counter = 0

        self._init_completed = asyncio.Event()
        get_or_create_event_loop().create_task(self._setup(llm_deployments))

        # Install raw ASGI middleware for /internal/route after the
        # @serve.router wrapper sets self._asgi_app.
        get_or_create_event_loop().create_task(self._install_route_middleware())

    async def _install_route_middleware(self):
        """Replace the ASGI app with a wrapper that intercepts /internal/route."""
        # Wait for @serve.router to set _asgi_app
        while not hasattr(self, "_asgi_app"):
            await asyncio.sleep(0.01)

        original_app = self._asgi_app
        router_self = self
        _log_path = "/tmp/llm_router_debug.log"

        async def route_middleware(scope, receive, send):
            with open(_log_path, "a") as f:
                f.write(
                    f"middleware hit: type={scope.get('type')} "
                    f"method={scope.get('method')} "
                    f"path={scope.get('path')}\n"
                )
            if (
                scope["type"] == "http"
                and scope.get("method") == "POST"
                and scope.get("path") == "/internal/route"
            ):
                await router_self._handle_route_raw(scope, receive, send)
                return
            await original_app(scope, receive, send)

        self._asgi_app = route_middleware
        with open(_log_path, "a") as f:
            f.write("middleware installed\n")

    async def _handle_route_raw(self, scope, receive, send):
        """Raw ASGI handler for /internal/route — no FastAPI overhead."""
        import orjson

        # Read body
        body_parts = []
        while True:
            message = await receive()
            body_parts.append(message.get("body", b""))
            if not message.get("more_body", False):
                break
        body_bytes = b"".join(body_parts)

        try:
            parsed = orjson.loads(body_bytes)
        except Exception:
            await self._send_json(send, {"error": "invalid json"}, 400)
            return

        default_model_id = (
            next(iter(self._llm_configs.keys())) if self._llm_configs else None
        )
        model = parsed.get("model", default_model_id)
        if model and model not in self._llm_configs:
            base = get_base_model_id(model)
            if base not in self._llm_configs:
                model = None
        model_id = model or default_model_id

        if model_id is None:
            await self._send_json(send, {"error": "no model"}, 404)
            return

        try:
            host, port, replica_id = await self._pick_replica(model_id)
        except Exception as e:
            await self._send_json(send, {"error": str(e)}, 503)
            return

        await self._send_json(send, {"host": host, "port": port}, 200)

    async def _send_json(self, send, data, status):
        import orjson

        body = orjson.dumps(data)
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [[b"content-type", b"application/json"]],
            }
        )
        await send({"type": "http.response.body", "body": body})

    async def _setup(self, llm_deployments: List[DeploymentHandle]):
        for handle in llm_deployments:
            llm_config = await handle.llm_config.remote()
            self._default_serve_handles[llm_config.model_id] = handle
            self._llm_configs[llm_config.model_id] = llm_config
        self._init_completed.set()

    async def check_health(self):
        await self._init_completed.wait()

    @router_app.get("/")
    @router_app.get("/health")
    async def health(self):
        return JSONResponse({"status": "ok"})

    async def _pick_replica(self, model_id: str) -> Tuple[str, int, str]:
        """Pick a replica via round-robin.

        Direct ingress bypasses Serve's request tracking, so Serve-side
        queue lengths are always 0. Use simple round-robin for even
        distribution.
        """
        base_model_id = get_base_model_id(model_id)
        handle = self._default_serve_handles.get(base_model_id)
        if handle is None:
            raise RuntimeError(f"No handle for model {model_id}")

        request_router = handle._get_request_router()
        if request_router is None:
            raise RuntimeError(f"Request router not initialized for {model_id}")

        di_replicas = [
            r
            for r in request_router.curr_replicas.values()
            if r.direct_ingress_endpoint is not None
        ]
        if not di_replicas:
            raise RuntimeError(f"No direct-ingress-enabled replicas for {model_id}")

        idx = self._rr_counter % len(di_replicas)
        self._rr_counter += 1
        best = di_replicas[idx]

        return (*best.direct_ingress_endpoint, best.replica_id.unique_id)
