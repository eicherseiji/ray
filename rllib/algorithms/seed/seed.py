import time
from typing import Optional

import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.rllib.algorithms.appo import APPO
from ray.rllib.algorithms.infinite_appo.infinite_appo import InfiniteAPPOConfig
from ray.rllib.algorithms.infinite_appo.utils import (
    BatchDispatcher,
    MetricsActor,
    WeightsServerActor,
)
from ray.rllib.algorithms.seed.utils.seed_env_runner import SEEDEnvRunner
from ray.rllib.algorithms.seed.utils.seed_inference import SEEDInference
from ray.rllib.utils.annotations import OverrideToImplementCustomLogic, override
from ray.tune.execution.placement_groups import PlacementGroupFactory


class SEEDConfig(InfiniteAPPOConfig):
    """The config class for SEED."""

    def __init__(self, algo_class=None):
        """Initializes a SEEDConfig instance."""
        super().__init__(algo_class=algo_class or SEED)

        # SEED-specific configs
        self.env_runner_cls = SEEDEnvRunner

        # Distributed inference
        self.inference_batch_size = 1
        self.n_inference_processes = 1
        self.inference_processes_scheduling_strategy = "SPREAD"
        self.inference_num_cpus_per_process = 1
        self.inference_num_gpus_per_process = 0

        # ZMQ-based Router-Dealer communication pattern
        self._zmq_asyncio = False
        self._router_channel_max_num_actors = 1_000
        self._max_outbound_messages = 100_000
        self._max_inbound_messages = 100_000

        # Override some of AlgorithmConfig's default values
        self.num_gpu_loader_threads = 4

        # TODO (sven): APPO Learner.
        self.circular_buffer_num_batches = 1
        self.circular_buffer_iterations_per_batch = 1

    def zeromq(
        self,
        *,
        zmq_asyncio,
        router_channel_max_num_actors,
        max_outbound_messages,
        max_inbound_messages,
    ) -> "SEEDConfig":
        """Sets parameters for the ZMQ-based Router-Dealer communication pattern"""
        if zmq_asyncio is not NotProvided:
            self._zmq_asyncio = zmq_asyncio
        if router_channel_max_num_actors is not NotProvided:
            self._router_channel_max_num_actors = router_channel_max_num_actors
        if max_outbound_messages is not NotProvided:
            self._max_outbound_messages = max_outbound_messages
        if max_inbound_messages is not NotProvided:
            self._max_inbound_messages = max_inbound_messages

        return self

    @override(AlgorithmConfig)
    def resources(
        self,
        inference_num_cpus_per_process: Optional[int] = NotProvided,
        inference_num_gpus_per_process: Optional[int] = NotProvided,
        inference_processes_scheduling_strategy: Optional[str] = NotProvided,
        **kwargs,
    ) -> "SEEDConfig":
        """Sets the resources-related configuration."""
        super().resources(**kwargs)
        if inference_num_cpus_per_process is not NotProvided:
            self.inference_num_cpus_per_process = inference_num_cpus_per_process
        if inference_num_gpus_per_process is not NotProvided:
            self.inference_num_gpus_per_process = inference_num_gpus_per_process
        if inference_processes_scheduling_strategy is not NotProvided:
            self.inference_processes_scheduling_strategy = (
                inference_processes_scheduling_strategy
            )

        return self

    def inference(
        self,
        *,
        inference_batch_size: Optional[int] = NotProvided,
        n_inference_processes: Optional[int] = NotProvided,
    ) -> "SEEDConfig":
        """Sets the inference-related configuration."""
        if inference_batch_size is not NotProvided:
            self.inference_batch_size = inference_batch_size
        if n_inference_processes is not NotProvided:
            self.n_inference_processes = n_inference_processes

        return self

    @override(InfiniteAPPOConfig)
    def training(
        self,
        **kwargs,
    ) -> "SEEDConfig":
        """Sets the training-related configuration."""
        super().training(**kwargs)
        return self

    @override(InfiniteAPPOConfig)
    def validate(self) -> None:
        super().validate()

        # TODO (kamil): zmq _router_channel_max_num_actors -> should be larger than
        #  number of env runners per inference process

        # Max batch size.
        # TODO (sven): Figure out settings for evaluation env runners.
        if (
            not self.in_evaluation
            and self.inference_batch_size
            > (self.num_env_runners or 1) * self.num_envs_per_env_runner
        ):
            raise ValueError(
                f"`inference_batch_size` ({self.inference_batch_size}) must be <= the "
                f"product of `num_env_runners` ({self.num_env_runners}) and "
                f"`num_envs_per_env_runner` ({self.num_envs_per_env_runner})!"
            )

        # "env runners / inference" -> Should be and integer.
        # TODO (sven): Figure out settings for evaluation env runners.
        if not self.in_evaluation:
            ratio = (self.num_env_runners or 1) / self.n_inference_processes
            if ratio != int(ratio):
                raise ValueError(
                    f"The ratio of `num_env_runners` ({self.num_env_runners}) / "
                    f"`n_inference_processes` ({self.n_inference_processes}) must be "
                    "a whole number (int)!"
                )


class SEED(Algorithm):
    """SEED Algorithm class."""

    @OverrideToImplementCustomLogic
    @override(APPO)
    @classmethod
    def default_resource_request(cls, config: SEEDConfig):
        pg_factory = APPO.default_resource_request(config)

        config = cls.get_default_config().update_from_dict(config)

        inference_bundles = [
            {
                "CPU": config.inference_num_cpus_per_process,
                "GPU": config.inference_num_gpus_per_process,
            }
            for _ in range(config.n_inference_processes)
        ]

        seed_bundles = (
            pg_factory.bundles
            + inference_bundles
            + [
                # 1 metrics actor + n weights servers + m batch dispatchers +
                # o env runner state aggregators.
                {"CPU": 1}
                for _ in range(
                    1
                    + config["num_weights_server_actors"]
                    + config["num_batch_dispatchers"]
                    + config["num_env_runner_state_aggregators"]
                )
            ]
        )
        return PlacementGroupFactory(
            bundles=seed_bundles,
            strategy=config["placement_strategy"],
        )

    @classmethod
    @override(Algorithm)
    def get_default_config(cls) -> SEEDConfig:
        return SEEDConfig()

    @override(Algorithm)
    def setup(self, config: SEEDConfig):
        super().setup(config)

        # SEED inference: internal attributes
        self._env_runners_per_inference = None
        self._inference_actors = {}

        # initially there should be "num_env_runners" EnvRunners
        assert (
            self.config.num_env_runners
            == self.env_runner_group.num_healthy_remote_env_runners()
        ), (
            f"expected {self.config.num_env_runners} EnvRunners, "
            f"but got {self.env_runner_group.num_healthy_remote_env_runners()}"
        )

        self._env_runners_per_inference = (
            self.config.num_env_runners // self.config.n_inference_processes
        )
        env_runner_ids = self.env_runner_group.healthy_env_runner_ids()

        # run the setup for "n_inference_processes" subgroup of EnvRunners
        for j, i in enumerate(
            range(0, len(env_runner_ids), self._env_runners_per_inference)
        ):
            subgroup_ids = env_runner_ids[i : i + self._env_runners_per_inference]
            env_runners_subgroup = {
                k: v
                for k, v in self.env_runner_group._worker_manager.actors().items()
                if k in subgroup_ids
            }
            self._inference_actors[j] = self._make_inference_actor(
                env_runners=env_runners_subgroup
            )

            print(
                f"SEEDInference: {self._inference_actors[j]}; connected to "
                f"n={len(env_runners_subgroup)} EnvRunners"
            )
            for k, v in env_runners_subgroup.items():
                print(f"    EnvRunner with id: {k}; ActorHandle: {v}")

        # Create metrics actor (last CPU bundle in pg).
        self.metrics_actor = MetricsActor.remote()

        # Create env runner state aggregator actors.
        # self.env_runner_state_aggregators = [
        #    EnvRunnerStateAggregator.remote(
        #        config=self.config,
        #        spaces=self.env_runner_group.get_spaces(),
        #    )
        #    for _ in range(self.config.num_env_runner_state_aggregators)
        # ]

        # Create weights server actors (next last n CPU-actors in pg).
        self.weights_server_actors = [
            WeightsServerActor.remote()
            for _ in range(self.config.num_weights_server_actors)
        ]
        for aid, actor in enumerate(self.weights_server_actors):
            actor.set_peers.remote(
                self.weights_server_actors[:aid] + self.weights_server_actors[aid + 1 :]
            )
        # Create batch dispatcher actors (next last n CPU-actors in pg).
        self.batch_dispatcher_actors = [
            BatchDispatcher.remote(sync_freq=self.config.pipeline_sync_freq)
            for _ in range(self.config.num_batch_dispatchers)
        ]

        # Setup all Learners' knowledge of important actors.
        learners = list(self.learner_group._worker_manager.actors().values())
        for lid, learner in enumerate(learners):
            ray.get(
                learner.set_other_actors.remote(
                    metrics_actor=self.metrics_actor,
                    weights_server_actors=self.weights_server_actors,
                    batch_dispatchers=self.batch_dispatcher_actors,
                )
            )
        self.aggregator_actors = [
            res.get()
            for res in self.learner_group.foreach_learner(
                func=lambda learner: learner.aggregator_actors,
            ).result_or_errors
        ]

        # Add agg. actors to inference actors.
        for inf_actor in self._inference_actors.values():
            inf_actor.set_aggregator_actors.remote(self.aggregator_actors)
            inf_actor.set_weights_server_actors.remote(self.weights_server_actors)

        # Start the inference actors.
        ray.get(
            [
                inf_act.start_infinite_inference.remote()
                for inf_act in self._inference_actors.values()
            ]
        )

        # Start the env runner actors' infinite sampling loops.
        self.env_runner_group.foreach_env_runner(
            "start_infinite_sample",
            local_env_runner=False,
        )

        # Add agg. actors, weights server actors and correct sync_freq to env runners.
        # agg = self.aggregator_actors[:]
        # er_agg = self.env_runner_state_aggregators[:]
        # ws = self.weights_server_actors[:]
        # sync_freq = self.config.pipeline_sync_freq

        # def _setup_er(env_runner, agg=agg, er_agg=er_agg, ws=ws, sync_freq=sync_freq):
        #    env_runner.set_aggregator_actors(aggregator_actor_refs=agg)
        #    env_runner.set_env_runner_state_aggregators(er_agg)
        #    env_runner.set_weights_server_actors(weights_server_actors=ws)
        #    env_runner.sync_freq = sync_freq

        # self.env_runner_group.foreach_env_runner(_setup_er)

        # Set metrics actor and learner on all batch dispatchers.
        for i in range(self.config.num_batch_dispatchers):
            self.batch_dispatcher_actors[i].set_other_actors.remote(
                metrics_actor=self.metrics_actor,
                learners=learners,
            )

    def _make_inference_actor(self, env_runners: dict):
        _inference_actor = SEEDInference.options(
            num_cpus=self.config.inference_num_cpus_per_process,
            num_gpus=self.config.inference_num_gpus_per_process,
            # scheduling_strategy=self.config.inference_processes_scheduling_strategy,
        ).remote(
            config=self.config,
            metrics=self.metrics,
            env_runners=env_runners,
        )

        _ready = False
        while not _ready:
            status = ray.get(_inference_actor.is_initialized.remote())
            if status:
                break
            else:
                time.sleep(2)

        _ready = False
        _ = ray.get(_inference_actor.setup.remote())
        while not _ready:
            status = ray.get(_inference_actor.is_ready.remote())
            if status:
                break
            else:
                time.sleep(2)

        return _inference_actor

    @override(Algorithm)
    def training_step(self):
        # Get results from metrics actor.
        metrics = ray.get(self.metrics_actor.get.remote())
        self.metrics.merge_and_log_n_dicts([metrics])
