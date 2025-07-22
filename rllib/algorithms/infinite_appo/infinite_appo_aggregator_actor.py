import random

import tree  # pip install dm_tree

import ray
from ray.rllib.algorithms.utils import AggregatorActor
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
from ray.rllib.utils.framework import try_import_torch

torch, _ = try_import_torch()


class InfiniteAPPOAggregatorActor(AggregatorActor):
    def __init__(
        self,
        *,
        config,
        rl_module_spec,
        sync_freq,
    ):
        super().__init__(config=config, rl_module_spec=rl_module_spec)

        # Remove NumpyToTensor piece from connector, if we don't do GPU pre-loading.
        if not self.config.enable_gpu_pre_loading:
            self._device = torch.device("cpu")
            self._learner_connector.remove("NumpyToTensor")

        self.sync_freq = sync_freq
        self._batch_dispatchers = None
        self._metrics_actor = None
        self._learner_index = None

        self._num_batches_produced = 0
        self._ts = 0
        self._episodes = []
        self._env_runner_metrics = MetricsLogger()

    def set_other_actors(self, *, batch_dispatchers, metrics_actor, learner_index):
        self._batch_dispatchers = batch_dispatchers
        self._metrics_actor = metrics_actor
        self._learner_index = learner_index

    # Synchronization helper method.
    def sync(self):
        return None

    def push_episodes(self, episodes, env_runner_metrics):
        self._env_runner_metrics.merge_and_log_n_dicts([env_runner_metrics])

        # Make sure we count how many timesteps we already have and only produce a
        # batch, once we have enough episode data.

        # TODO (sven): Fix this logic for all algos. For infinite APPO, this should NOT
        #  be done here as it already happened on the EnvRunners.
        # Only for SEED algo: Numpy'ize episodes, if necessary.
        if type(self.config).__name__ == "SEEDConfig" and self.config.episodes_to_numpy:
            for eps in episodes:
                eps.to_numpy()

        self._episodes.extend(episodes)

        env_steps = sum(len(e) for e in episodes)
        self._ts += env_steps

        # If we have enough episodes collected, pass them through the connector
        # to create a single train batch.
        if self._ts >= self.config.train_batch_size_per_learner:
            batch = self._learner_connector(
                episodes=self._episodes,
                rl_module=self._module,
                metrics=self.metrics,
            )
            batch_env_steps = sum(len(e) for e in self._episodes)
            self._ts = 0
            self._episodes = []

            # Pre-load onto the GPU using IPC.
            if self.config.enable_gpu_pre_loading:
                batch = tree.map_structure(
                    lambda s: _SharedCUDA(
                        s.untyped_storage()._share_cuda_(),
                        dtype=s.dtype,
                        shape=s.shape,
                    ),
                    batch,
                )

            self.metrics.log_value(
                "num_env_steps_aggregated_lifetime",
                batch_env_steps,
                reduce="sum",
                with_throughput=True,
            )

            # Forward results to a Learner actor.
            batch_dispatch_actor = random.choice(self._batch_dispatchers)
            batch_dispatch_actor.add_batch.remote(
                batch_ref={"train_batch": batch},
                batch_env_steps=batch_env_steps,
                learner_index=self._learner_index,
            )

            self._num_batches_produced += 1

            if self._num_batches_produced % 10 == 0:
                self._metrics_actor.add.remote(
                    env_runner_metrics=self._env_runner_metrics.reduce(),
                    aggregator_metrics=self.metrics.reduce(),
                )

            # Sync with one of the dispatcher actors.
            if self._num_batches_produced % self.sync_freq == 0:
                ray.get(batch_dispatch_actor.sync.remote())


class _SharedCUDA:
    def __init__(self, handle, *, shape, dtype):
        self.handle = handle
        self.dtype = dtype
        self.shape = shape
