"""LLMRouter: a dedicated router deployment for ingress bypass.

When ingress bypass is enabled, HAProxy calls /internal/route on this
deployment to get a (host, port) pair, then forwards traffic directly
to that LLMServer replica's direct ingress port.
"""

import asyncio
import os
import time
from typing import Dict, List, Tuple

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from ray._common.utils import get_or_create_event_loop
from ray.llm._internal.common.utils.lora_utils import get_base_model_id
from ray.llm._internal.serve.core.configs.llm_config import LLMConfig
from ray.llm._internal.serve.observability.logging import get_logger
from ray.serve.api import router as serve_router
from ray.serve.handle import DeploymentHandle

_PERF_LOG_PATH = "/tmp/llm_router_perf.log"
_perf_log_fh = None


def _perf_log(line: str) -> None:
    global _perf_log_fh
    if _perf_log_fh is None:
        _perf_log_fh = open(_PERF_LOG_PATH, "a", buffering=1)
        _perf_log_fh.write(f"# pid={os.getpid()} start={time.time()}\n")
    _perf_log_fh.write(line)


logger = get_logger(__name__)

router_app = FastAPI()


@serve_router(router_app)
class LLMRouter:
    """Lightweight router deployment for ingress bypass.

    Receives /internal/route POST requests from HAProxy Lua, picks an
    LLMServer replica via round-robin, and returns its direct ingress
    (host, port).

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
        while not hasattr(self, "_asgi_app"):
            await asyncio.sleep(0.01)

        original_app = self._asgi_app
        router_self = self

        async def route_middleware(scope, receive, send):
            if (
                scope["type"] == "http"
                and scope.get("method") == "POST"
                and scope.get("path") == "/internal/route"
            ):
                mw_start = time.perf_counter_ns()

                async def send_wrapped(msg):
                    if msg.get("type") == "http.response.body" and not msg.get(
                        "more_body", False
                    ):
                        _perf_log(
                            f"mw\t{time.time():.6f}\ttotal={time.perf_counter_ns() - mw_start}\n"
                        )
                    await send(msg)

                await router_self._handle_route_raw(scope, receive, send_wrapped)
                return
            await original_app(scope, receive, send)

        self._asgi_app = route_middleware
        logger.info("LLMRouter route middleware installed")

    async def _handle_route_raw(self, scope, receive, send):
        """Raw ASGI handler for /internal/route — no FastAPI overhead."""
        import orjson

        t0 = time.perf_counter_ns()

        # Read body
        body_parts = []
        while True:
            message = await receive()
            body_parts.append(message.get("body", b""))
            if not message.get("more_body", False):
                break
        body_bytes = b"".join(body_parts)

        t1 = time.perf_counter_ns()

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

        t2 = time.perf_counter_ns()

        await self._send_json(send, {"host": host, "port": port}, 200)

        t3 = time.perf_counter_ns()
        _perf_log(
            f"{time.time():.6f}\tbody={t1 - t0}\tpick={t2 - t1}\tsend={t3 - t2}\ttotal={t3 - t0}\n"
        )

    async def _send_json(self, send, data, status):
        import orjson

        body = orjson.dumps(data)
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [
                    [b"content-type", b"application/json"],
                    [b"content-length", str(len(body)).encode()],
                ],
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
        """Least-loaded pick via the request router's choose_replicas.

        Measured at c=32/8r today on this box:
        - choose_replicas (this code): 125ms TTFT
        - reference bg-poller min cache (PR #62298 verbatim): 323ms TTFT
          (min() stalls on ties; 32 concurrent picks in the 50ms poll
          window all select the same replica)
        - bg-poller + local-increment tie-break: 141ms TTFT
          (still worse because 50ms cache lag misses fast arrivals)
        """
        base_model_id = get_base_model_id(model_id)
        handle = self._default_serve_handles.get(base_model_id)
        if handle is None:
            raise RuntimeError(f"No handle for model {model_id}")

        request_router = handle._get_request_router()
        if request_router is None:
            raise RuntimeError(f"Request router not initialized for {model_id}")

        direct_ingress_replicas = [
            r
            for r in request_router.curr_replicas.values()
            if r.direct_ingress_endpoint is not None
        ]
        if not direct_ingress_replicas:
            raise RuntimeError(f"No direct-ingress-enabled replicas for {model_id}")

        replica_tiers = await request_router.choose_replicas(
            candidate_replicas=direct_ingress_replicas,
            pending_request=None,
        )
        for tier in replica_tiers:
            for replica in tier:
                endpoint = replica.direct_ingress_endpoint
                if endpoint is not None:
                    return (*endpoint, replica.replica_id.unique_id)

        # Fallback: pure round-robin if choose_replicas returned no tiers.
        idx = self._rr_counter % len(direct_ingress_replicas)
        self._rr_counter += 1
        best = direct_ingress_replicas[idx]
        return (*best.direct_ingress_endpoint, best.replica_id.unique_id)
