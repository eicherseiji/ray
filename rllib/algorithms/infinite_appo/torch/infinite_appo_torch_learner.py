import random

import torch
import tree

import ray
from ray.rllib.algorithms.appo.torch.appo_torch_learner import APPOTorchLearner
from ray.rllib.algorithms.appo.utils import CircularBuffer
from ray.rllib.algorithms.infinite_appo.infinite_appo_aggregator_actor import (
    InfiniteAPPOAggregatorActor,
)
from ray.rllib.core import COMPONENT_RL_MODULE
from ray.rllib.core.learner.torch.torch_learner import TorchLearner
from ray.rllib.core.learner.training_data import TrainingData
from ray.rllib.policy.sample_batch import MultiAgentBatch, SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.metrics import NUM_ENV_STEPS_TRAINED_LIFETIME


class InfiniteAPPOTorchLearner(APPOTorchLearner):
    @override(APPOTorchLearner)
    def build(self) -> None:
        super().build()

        self._num_batches = 0
        self._timesteps = {NUM_ENV_STEPS_TRAINED_LIFETIME: 0}

        # Create child aggregator actors and place all of them on the same device
        # through picking the same placement bundle index as this very Learner actor.
        self.aggregator_actors = [
            ray.remote(
                num_cpus=1,
                # Provide each agg. actor with access to the GPU, only so that it can
                # preload the train batches to the GPU. The agg. actor doesn't have to
                # do any heavy lifting on that GPU, so 0.01 seems a good choice here.
                # The Learner would still have 90+x% of the GPU for computations.
                num_gpus=0.01
                * float(
                    self.config.enable_gpu_pre_loading
                    and self.config.num_gpus_per_learner > 0
                ),
            )(InfiniteAPPOAggregatorActor)
            .options(
                placement_group=self._placement_group,
                placement_group_bundle_index=(
                    -1
                    if self._placement_group is None
                    else
                    # main process
                    1
                    # env runners
                    + self.config.num_env_runners
                    # eval env runners
                    + self.config.get_evaluation_config_object().num_env_runners
                    # Learners
                    + self._learner_index
                ),
            )
            .remote(
                config=self.config,
                rl_module_spec=self._module_spec,
                sync_freq=self.config.pipeline_sync_freq,
            )
            for _ in range(self.config.num_aggregator_actors_per_inf_appo_learner)
        ]

        # Stop the Learner thread again and delete it.
        self._learner_thread.stopped = True
        # Make sure learner thread gets out of its `step()` method (waiting for
        # circular buffer to return an item).
        self._learner_thread_in_queue.add("dummy")
        del self._learner_thread

        # Recreate the circular buffer with K-1 (b/c we use the incoming batch right
        # away for 1 update, only then add it to the buffer).
        self._learner_thread_in_queue = CircularBuffer(
            num_batches=self.config.circular_buffer_num_batches,
            iterations_per_batch=self.config.circular_buffer_iterations_per_batch - 1,
        )

    # Synchronization helper method.
    def set_other_actors(
        self, *, metrics_actor, weights_server_actors, batch_dispatchers
    ):
        self._metrics_actor = metrics_actor
        self._weights_server_actors = weights_server_actors

        for agg in self.aggregator_actors:
            ray.get(
                agg.set_other_actors.remote(
                    batch_dispatchers=batch_dispatchers,
                    metrics_actor=metrics_actor,
                    learner_index=self._learner_index,
                )
            )

    @override(APPOTorchLearner)
    def _compute_off_policyness(self, batch):
        # TODO (sven): Investigate, why this call is slowing things down in the
        #  distributed Learner setup.
        pass

    @override(APPOTorchLearner)
    def update(self, batch_and_env_steps, timesteps, send_weights=False):
        if timesteps is not None:
            self._timesteps = timesteps

        # Reduce metrics (and sync them from GPU, if applicable), then send reduced
        # metrics to metrics actor.
        reduced_metrics = None
        if self._num_batches >= 10:
            reduced_metrics = self.metrics.reduce()
            self._num_batches = 0

        batch, env_steps = batch_and_env_steps

        # Get tensors directly from GPU memory.
        if self.config.enable_gpu_pre_loading:
            batch = tree.map_structure(self._map_from_gpu_memory, batch)

        # Convert to MABatch.
        batch = MultiAgentBatch(
            policy_batches={pid: SampleBatch(b) for pid, b in batch.items()},
            env_steps=env_steps,
        )

        # Load the batch to the GPU.
        if not self.config.enable_gpu_pre_loading:
            batch = batch.to_device(self._device, pin_memory=False)

        # If buffer is full, pull K batches from it and perform an update on each.
        if (
            self.config.circular_buffer_iterations_per_batch == 1
            or self._learner_thread_in_queue.filled
        ):
            for i in range(self.config.circular_buffer_iterations_per_batch):
                # Don't sample the very first batch, but use the one we just received.
                # This saves an entire sampling step AND makes sure that new batches
                # are consumed right away (at least once) before we even add them to
                # the circular buffer.
                if i > 0:
                    batch = self._learner_thread_in_queue.sample()
                TorchLearner.update(
                    self,
                    training_data=TrainingData(batch=batch),
                    timesteps=self._timesteps,
                    _no_metrics_reduce=True,
                )
                self._num_batches += 1
                self._timesteps[NUM_ENV_STEPS_TRAINED_LIFETIME] += (
                    batch.env_steps() * self.config.num_learners
                )

        if self.config.circular_buffer_iterations_per_batch > 1:
            self._learner_thread_in_queue.add(batch)

        # Figure out, whether we need to send our weights to a weights server.
        if send_weights and self._weights_server_actors:
            learner_state = self.get_state(
                # Only return the state of those RLModules that are trainable.
                components=[
                    COMPONENT_RL_MODULE + "/" + mid
                    for mid in self.module.keys()
                    if self.should_module_be_updated(mid)
                ],
                # Inference-only, b/c this is for the EnvRunners.
                inference_only=True,
            )
            learner_state[COMPONENT_RL_MODULE] = ray.put(
                learner_state[COMPONENT_RL_MODULE]
            )
            random.choice(self._weights_server_actors).put.remote(
                learner_state, broadcast=True
            )

        if reduced_metrics is not None:
            self._metrics_actor.add.remote(learner_metrics=reduced_metrics)

    def _map_from_gpu_memory(self, s):
        storage = torch.UntypedStorage._new_shared_cuda(*s.handle)
        tensor = torch.tensor([], dtype=s.dtype, device=self._device)
        tensor.set_(storage, 0, s.shape)
        return tensor
