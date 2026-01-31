from typing import Optional

import numpy as np

import ray
from ray.rllib.algorithms import Algorithm, AlgorithmConfig
from ray.rllib.algorithms.algorithm_config import NotProvided
from ray.rllib.algorithms.appo import APPO, APPOConfig
from ray.rllib.algorithms.infinite_appo.infinite_appo_multi_agent_env_runner import (
    InfiniteAPPOMultiAgentEnvRunner,
)
from ray.rllib.algorithms.infinite_appo.utils import (
    BatchDispatcher,
    EnvRunnerStateAggregator,
    MetricsActor,
    WeightsServerActor,
)
from ray.rllib.core import ALL_MODULES
from ray.rllib.utils.annotations import override
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    LEARNER_RESULTS,
    NUM_ENV_STEPS_SAMPLED_LIFETIME,
    NUM_ENV_STEPS_TRAINED_LIFETIME,
)
from ray.tune import PlacementGroupFactory


class InfiniteAPPOConfig(APPOConfig):
    def __init__(self, algo_class=None):
        super().__init__(algo_class=algo_class or InfiniteAPPO)

        self.enable_gpu_pre_loading = False
        self.num_weights_server_actors = 1
        self.num_batch_dispatchers = 1
        self.num_env_runner_state_aggregators = 1
        self.num_aggregator_actors_per_inf_appo_learner = 2
        self.pipeline_sync_freq = 10

        # Defaults overriding APPOConfig settings.
        self.num_aggregator_actors_per_learner = 0
        self.env_runner_cls = InfiniteAPPOMultiAgentEnvRunner
        self.num_gpu_loader_threads = 0

    @override(APPOConfig)
    def validate(self) -> None:
        super().validate()

        if self.num_aggregator_actors_per_learner != 0:
            raise ValueError(
                "`num_aggregator_actors_per_learner` must be 0! Use "
                "`config.training(num_aggregator_actors_per_inf_appo_learner=...) "
                "instead."
            )

        if self.enable_gpu_pre_loading and self.num_gpus_per_learner == 0:
            raise ValueError(
                "Can't set `enable_gpu_pre_loading=True` w/o setting "
                "`num_gpus_per_learner` > 0!"
            )

    @override(APPOConfig)
    def get_default_learner_class(self):
        if self.framework_str == "torch":
            from ray.rllib.algorithms.infinite_appo.torch import (
                infinite_appo_torch_learner,
            )

            return infinite_appo_torch_learner.InfiniteAPPOTorchLearner
        else:
            raise ValueError(
                f"The framework {self.framework_str} is not supported. "
                "Use `framework='torch'`."
            )

    @override(APPOConfig)
    def training(
        self,
        *,
        enable_gpu_pre_loading: Optional[bool] = NotProvided,
        num_aggregator_actors_per_inf_appo_learner: Optional[int] = NotProvided,
        num_weights_server_actors: Optional[int] = NotProvided,
        num_batch_dispatchers: Optional[int] = NotProvided,
        num_env_runner_state_aggregators: Optional[int] = NotProvided,
        pipeline_sync_freq: Optional[int] = NotProvided,
        **kwargs,
    ):
        """Sets the training related configuration.

        Args:
            enable_gpu_pre_loading: Whether the train batch on the aggregator actors
                should be already pre-loaded to the Learner's GPU. If True, each
                aggregator actor occupies 0.01 of the Learner's GPU.
            num_aggregator_actors_per_inf_appo_learner: The number of aggregator actors
                to use per Learner.
            num_weights_server_actors: The number of weights server actors to use. These
                receive weight updates from the Learners and distribute those to the
                EnvRunners.
            num_batch_dispatchers: The number of batch dispatcher actors to use.
                These actors receive train batch references from the aggregator actors
                and make sure a full set of train batches is sent out to all Learners in
                parallel. This ensures a continuous task-flow into the Learner pipeline
                and increases GPU utilization.
            num_env_runner_state_aggregators: The number of env runner state aggregators
                to use. These actors collect EnvRunner states, merge them into one
                global state and redistribute this merged state back to all EnvRunners.
            pipeline_sync_freq: The frequency, with which each actor type (EnvRunner,
                aggregator actor, etc..) sends a ray.remote "sync" request, which
                returns None, to the next layer of actors in order to avoid queueing
                in case one actor layer poses a bottleneck. Increase this value for
                a higher throughput, but at the cost of larger off-policiness.
        """
        super().training(**kwargs)

        if enable_gpu_pre_loading is not NotProvided:
            self.enable_gpu_pre_loading = enable_gpu_pre_loading
        if num_aggregator_actors_per_inf_appo_learner is not NotProvided:
            self.num_aggregator_actors_per_inf_appo_learner = (
                num_aggregator_actors_per_inf_appo_learner
            )
        if num_weights_server_actors is not NotProvided:
            self.num_weights_server_actors = num_weights_server_actors
        if num_batch_dispatchers is not NotProvided:
            self.num_batch_dispatchers = num_batch_dispatchers
        if num_env_runner_state_aggregators is not NotProvided:
            self.num_env_runner_state_aggregators = num_env_runner_state_aggregators
        if pipeline_sync_freq is not NotProvided:
            self.pipeline_sync_freq = pipeline_sync_freq

        return self


class InfiniteAPPO(APPO):
    @override(Algorithm)
    @classmethod
    def default_resource_request(cls, config):
        pg_factory = APPO.default_resource_request(config)

        # Loop through learner bundles and add the `InfiniteAPPOAggregatorActors`.
        # Note that we don't have to change the GPUs here as agg. actors don't take
        # their own GPU, they take a slice (0.01) from the Learner's GPU. Thus, the
        # number of GPUs per bundle remains the same (normally 1.0).
        num_learners = config["num_learners"]
        learner_bundles = []
        for i in range(num_learners):
            lb = pg_factory.bundles[-num_learners + i].copy()
            if "CPU" not in lb:
                lb["CPU"] = 0
            lb["CPU"] += config["num_aggregator_actors_per_inf_appo_learner"]
            learner_bundles.append(lb)

        infinite_appo_bundles = (
            pg_factory.bundles[:-num_learners]
            + learner_bundles
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
            bundles=infinite_appo_bundles,
            strategy=config["placement_strategy"],
        )

    @classmethod
    @override(APPO)
    def get_default_config(cls) -> AlgorithmConfig:
        return InfiniteAPPOConfig()

    @override(APPO)
    def setup(self, config: AlgorithmConfig):
        super().setup(config=config)

        # Create metrics actor (last CPU bundle in pg).
        self.metrics_actor = MetricsActor.remote()

        # Create env runner state aggregator actors.
        self.env_runner_state_aggregators = [
            EnvRunnerStateAggregator.remote(
                config=self.config,
                spaces=self.env_runner_group.get_spaces(),
            )
            for _ in range(self.config.num_env_runner_state_aggregators)
        ]

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
        for learner in learners:
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

        # Add agg. actors, weights server actors and correct sync_freq to env runners.
        agg = self.aggregator_actors[:]
        er_agg = self.env_runner_state_aggregators[:]
        ws = self.weights_server_actors[:]
        sync_freq = self.config.pipeline_sync_freq

        def _setup_er(env_runner, agg=agg, er_agg=er_agg, ws=ws, sync_freq=sync_freq):
            env_runner.set_aggregator_actors(aggregator_actor_refs=agg)
            env_runner.set_env_runner_state_aggregators(er_agg)
            env_runner.set_weights_server_actors(weights_server_actors=ws)
            env_runner.sync_freq = sync_freq

        self.env_runner_group.foreach_env_runner(_setup_er)

        # Set metrics actor and learner on all batch dispatchers.
        for i in range(self.config.num_batch_dispatchers):
            self.batch_dispatcher_actors[i].set_other_actors.remote(
                metrics_actor=self.metrics_actor,
                learners=learners,
            )

        self._env_runners_started = False
        self._env_runners_pending_failure_checks = set()

    @override(APPO)
    def training_step(self):
        # Kick of sampling, aggregating, and training, if not done yet.
        if not self._env_runners_started:
            self.env_runner_group.foreach_env_runner(
                "start_infinite_sample",
                local_env_runner=False,
            )
            self._env_runners_started = True

        # Pull previous `ping` command results.
        health_check_results = self.env_runner_group.fetch_ready_async_reqs()
        for env_runner_id, _ in health_check_results:
            self._env_runners_pending_failure_checks.remove(env_runner_id)
        # Check a random subset of (max 50) EnvRunners for failures.
        env_runner_ids_to_check = set(
            map(
                int,
                np.random.choice(
                    range(1, self.config.num_env_runners + 1),
                    max(min(self.config.num_env_runners // 10, 50), 1),
                    replace=False,
                ),
            )
        )
        check = env_runner_ids_to_check - self._env_runners_pending_failure_checks
        if check:
            self.env_runner_group.foreach_env_runner_async(
                func="ping",
                remote_worker_ids=list(check),
            )
            self._env_runners_pending_failure_checks.update(env_runner_ids_to_check)

        # Update all global timestep counters on all batch dispatchers.
        timesteps = {
            NUM_ENV_STEPS_SAMPLED_LIFETIME: self.metrics.peek(
                (ENV_RUNNER_RESULTS, NUM_ENV_STEPS_SAMPLED_LIFETIME),
                default=0,
            ),
            NUM_ENV_STEPS_TRAINED_LIFETIME: self.metrics.peek(
                (LEARNER_RESULTS, ALL_MODULES, NUM_ENV_STEPS_TRAINED_LIFETIME),
                default=0,
            ),
        }
        for batch_dispatcher in self.batch_dispatcher_actors:
            batch_dispatcher.set_timesteps.remote(timesteps)

        # Get results from metrics actor.
        metrics = ray.get(self.metrics_actor.get.remote())
        self.metrics.merge_and_log_n_dicts([metrics])

        # Get env runner states to update the local env runner, if necessary.
        if self.env_runner:
            env_runner_states = ray.get(
                self.env_runner_state_aggregators[0].get_connector_states.remote()
            )
            self.env_runner.set_state(env_runner_states)
