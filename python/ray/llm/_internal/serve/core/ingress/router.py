"""LLMRouter: a dedicated router deployment for ingress bypass.

When ingress bypass is enabled, HAProxy calls /internal/route on this
deployment to get a (host, port) pair, then forwards traffic directly
to that LLMServer replica's direct ingress port.
"""

import asyncio
from typing import Dict, List, Tuple

from fastapi import FastAPI, Request
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
    least-loaded LLMServer replica via the deployment handle's request
    router, and returns its direct ingress (host, port).
    """

    def __init__(self, llm_deployments: List[DeploymentHandle]):
        self._default_serve_handles: Dict[str, DeploymentHandle] = {}
        self._llm_configs: Dict[str, LLMConfig] = {}
        self._di_load_cache: Dict[str, float] = {}
        self._load_poller_started = False
        self._rr_counter = 0

        self._init_completed = asyncio.Event()
        get_or_create_event_loop().create_task(self._setup(llm_deployments))

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

    @router_app.post("/internal/route")
    async def route(self, request: Request):
        import time

        import orjson

        t0 = time.monotonic()
        body = await request.body()
        t_body = time.monotonic()

        try:
            parsed = orjson.loads(body)
        except Exception:
            return JSONResponse({"error": "invalid json"}, status_code=400)

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
            return JSONResponse({"error": "no model"}, status_code=404)

        t_pre_pick = time.monotonic()
        try:
            host, port, replica_id = await self._pick_replica(model_id)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=503)
        t_post_pick = time.monotonic()

        route_msg = (
            f"ROUTE request_id={request.headers.get('x-request-id', '?')} "
            f"model={model_id} → {host}:{port} "
            f"body={1000*(t_body-t0):.1f}ms "
            f"pick={1000*(t_post_pick-t_pre_pick):.1f}ms "
            f"total={1000*(t_post_pick-t0):.1f}ms"
        )
        logger.info(route_msg)
        print(route_msg, flush=True)

        return JSONResponse({"host": host, "port": port})

    async def _start_load_poller(self):
        """Background task: poll replica queue lengths every 50ms.

        Refreshes the replica list from the request router each iteration
        so scaling events are reflected.
        """
        while True:
            for handle in self._default_serve_handles.values():
                request_router = handle._get_request_router()
                if request_router is None:
                    continue
                for r in request_router.curr_replicas.values():
                    if r.direct_ingress_endpoint is None:
                        continue
                    try:
                        load = await r.get_queue_len(deadline_s=0.5)
                        self._di_load_cache[r.replica_id] = load
                    except Exception:
                        pass
            await asyncio.sleep(0.05)

    async def _pick_replica(self, model_id: str) -> Tuple[str, int, str]:
        """Pick the least-loaded replica using background-polled queue lengths.

        No per-request RPCs — routing decisions use cached load data,
        keeping the routing path fast and preserving request delivery
        timing patterns. Falls back to round-robin when cache is empty.

        Returns (host, port, replica_id).
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

        # Start background load poller on first call.
        if not self._load_poller_started:
            self._load_poller_started = True
            asyncio.get_event_loop().create_task(self._start_load_poller())

        # If cache has data, pick least-loaded. Otherwise round-robin.
        if self._di_load_cache:
            best = min(
                di_replicas,
                key=lambda r: self._di_load_cache.get(r.replica_id, float("inf")),
            )
        else:
            idx = self._rr_counter % len(di_replicas)
            self._rr_counter += 1
            best = di_replicas[idx]

        return (*best.direct_ingress_endpoint, best.replica_id.unique_id)
