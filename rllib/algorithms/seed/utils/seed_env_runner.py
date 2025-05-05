import threading
import time

import numpy as np

from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.core import DEFAULT_MODULE_ID
from ray.rllib.env import INPUT_ENV_SPACES
from ray.rllib.env.env_runner import EnvRunner
from ray.rllib.env.single_agent_env_runner import SingleAgentEnvRunner
from ray.rllib.utils.annotations import override
from ray.rllib.utils.checkpoints import Checkpointable
from ray.rllib.utils.metrics import TIMERS
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
from ray.rllib.utils.typing import StateDict


class SEEDEnvRunner(EnvRunner):
    """SEED: EnvRunner that implements ZMQ Router-Dealer communication pattern."""

    @override(EnvRunner)
    def __init__(self, *, config: AlgorithmConfig, **kwargs):
        super().__init__(config=config)

        self.worker_index: int = kwargs.get("worker_index")
        self.num_workers: int = kwargs.get("num_workers", self.config.num_env_runners)
        self.tune_trial_id: str = kwargs.get("tune_trial_id")

        self._callbacks = None
        self.metrics: MetricsLogger = MetricsLogger()
        self.dealer_channel = None
        self.env = None
        self.make_env()

        self._t0 = time.time()
        self._interval = 20

        # This should be the default.
        self._needs_initial_reset: bool = True

        self._sampling_thread = threading.Thread(
            name="sampling_thread",
            target=self._sampling_thread_func,
        )

    def start_zmq(self, dealer_channel):
        self.dealer_channel = dealer_channel

    def is_ready(self):
        if self.env is not None and self.dealer_channel is not None:
            return True
        else:
            return False

    @override(EnvRunner)
    def assert_healthy(self):
        """Checks that self.__init__() has been completed properly.

        Ensures that the instance has an environment defined.

        Raises:
            AssertionError: If the EnvRunner Actor has NOT been properly initialized.
        """
        assert self.env

    def start_infinite_sample(self):
        self._sampling_thread.start()

    def _sampling_thread_func(self):
        iteration = 0
        while True:
            with self.metrics.log_time((TIMERS, "mean_sample_time")):
                self.sample()
            iteration += 1

    @override(EnvRunner)
    def sample(self):
        # Receive the message from the RouterChannel (on the inference actors).
        with self.metrics.log_time((TIMERS, "mean_zeromq_read_time")):
            # Note: The very first actions received are dummy actions to be
            # discarded.
            actions = self.dealer_channel.read()
        self.metrics.log_value(
            key="messages_received_lifetime",
            value=1,
            reduce="sum",
            with_throughput=True,
        )

        # Compute an environment step.
        if self._needs_initial_reset:
            observation, info = self.env.reset()
            reward = np.zeros(shape=(self.num_envs,))
            terminated = truncated = np.zeros(shape=(self.num_envs,)).astype(bool)
            self._needs_initial_reset = False
        else:
            actions = np.frombuffer(actions, dtype=np.int32)
            observation, reward, terminated, truncated, info = self.env.step(actions)

        # Send the new state to the RouterChannel.
        self.dealer_channel.write(
            (
                observation.tobytes(),
                reward.tobytes(),
                terminated.tobytes(),
                truncated.tobytes(),
            )
        )

    @override(EnvRunner)
    def make_env(self):
        return SingleAgentEnvRunner.make_env(self)

    @override(EnvRunner)
    def make_module(self):
        raise NotImplementedError("SEEDEnvRunner doesn't have a module!")

    @override(EnvRunner)
    def stop(self):
        # Close our env object via gymnasium's API.
        self.env.close()

    def get_spaces(self):
        return {
            INPUT_ENV_SPACES: (self.env.observation_space, self.env.action_space),
            DEFAULT_MODULE_ID: (
                self.config.observation_space or self.env.single_observation_space,
                self.env.single_action_space,
            ),
        }

    @override(Checkpointable)
    def set_state(self, state: StateDict) -> None:
        pass

    @override(Checkpointable)
    def get_state(self, **kwargs) -> StateDict:
        return {}
