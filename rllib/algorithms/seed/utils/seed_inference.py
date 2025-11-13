import logging
import threading
import time
from collections import defaultdict
from typing import DefaultDict, Dict, List

import numpy as np
import torch
import tree  # pip install dm_tree

import ray
from ray.rllib.core import (
    COMPONENT_RL_MODULE,
    DEFAULT_MODULE_ID,
    Columns,
)
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.env import INPUT_ENV_SPACES
from ray.rllib.env.single_agent_env_runner import SingleAgentEnvRunner
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.rllib.utils.metrics import TIMERS, WEIGHTS_SEQ_NO
from ray.rllib.utils.typing import EpisodeID, ResultDict
from ray.util.anyscale.zmq_channel import RouterChannel

logger = logging.getLogger(__name__)


@ray.remote
class SEEDInference:
    """SEED: ZMQ communication pattern PoC"""

    def __init__(
        self,
        *,
        config,
        metrics,
        env_runners,
    ):
        self.config = config
        self.metrics = metrics
        self.env_runners = env_runners

        # TODO (sven): No need for local EnvRunner, use get_spaces API to get the
        #  actual env's spaces from the remote env runners and use these in the call
        #  to build the module and connectors.
        self._env_runner = SingleAgentEnvRunner(config=config)
        self.env = self._env_runner.env

        self.num_envs = self.env.num_envs

        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._env_to_module = config.build_env_to_module_connector(
            self.env,
            device=_device,
        )
        self.module = None
        self.make_module()
        self._module_to_env = config.build_module_to_env_connector(self.env)
        self.module.to(_device)

        # Dict mapping EnvRunner ActorID to lists (env vector indices) of ongoing
        # episodes.
        self._running_episodes: Dict[ray.ActorID, List[SingleAgentEpisode]] = {}
        self._episodes_for_aggregators = []
        self._done_episodes_for_metrics: List[SingleAgentEpisode] = []
        self._ongoing_episodes_for_metrics: DefaultDict[
            EpisodeID, List[SingleAgentEpisode]
        ] = defaultdict(list)
        self._inference_batch_count = 0
        self._rollout_ts_count = 0

        # Dicts mapping EnvRunner ActorID to lists (env vector indices) of previously
        # computed actions and model outputs.
        # These buffered values need to be passed into the `.add_env_step()` method
        # call of the episodes.
        self._buffered_actions = defaultdict(list)
        self._buffered_extra_model_outputs = defaultdict(list)

        self._episodes_for_next_inference_batch = []
        self._env_runners_for_actions = []

        self._inference_batch_size = max(
            self.config.inference_batch_size,
            (
                (self.config.num_env_runners // self.config.n_inference_processes)
                * self.num_envs
            ),
        )

        # ZMQ setup attributes
        self._router_channel = None

        # Aggregator actors.
        self._aggregator_actor_refs = None
        self._curr_agg_idx = 0

        # EnvRunner
        # self._env_runner_state_aggregator_refs
        self._weights_server_actors = None

        # logging
        self._t0 = None

    def is_initialized(self):
        if (
            self.module is not None
            and self._env_to_module is not None
            and self._module_to_env is not None
        ):
            return True
        else:
            return False

    def is_ready(self):
        if self._router_channel is not None:
            return True
        else:
            return False

    def setup(self):
        # Set up ZMQ router channel.
        self._router_channel = RouterChannel(
            _asyncio=self.config._zmq_asyncio,
            max_num_actors=self.config._router_channel_max_num_actors,
            max_outbound_messages=self.config._max_outbound_messages,
            max_inbound_messages=self.config._max_inbound_messages,
        )
        # Set up ZMQ dealer channels for the EnvRunners.
        _dealer_channels = {
            aid: self._router_channel.create_dealer(
                actor=actor,
                _asyncio=self.config._zmq_asyncio,
            )
            for aid, actor in self.env_runners.items()
        }
        merged = {
            k: (self.env_runners[k], _dealer_channels[k])
            for k in self.env_runners.keys()
        }
        for k, v in merged.items():
            er = v[0]
            dealer_ch = v[1]
            er.start_zmq.remote(dealer_channel=dealer_ch)

        self._inference_thread = threading.Thread(target=self._inference_thread_func)

        _ready = False
        while not _ready:
            if self._check_env_runners_ready():
                break
            else:
                time.sleep(2)

        # logging
        self._t0 = time.time()

    def set_aggregator_actors(self, aggregator_actor_refs):
        # `aggregator_actor_refs` must be list of lists.
        # Outer index is the Learner index.
        # Inner index is the aggregator index (for that Learner).

        # Create a flat list of aggregator actors.
        self._aggregator_actor_refs = []

        # Shuffle inner index (aggregator indices per Learner).
        for learner_idx in range(len(aggregator_actor_refs)):
            np.random.shuffle(aggregator_actor_refs[learner_idx])

        # Shuffle Learner sequence.
        learner_seq = list(range(len(aggregator_actor_refs)))
        np.random.shuffle(learner_seq)
        for agg_idx, agg_0 in enumerate(aggregator_actor_refs[learner_seq[0]]):
            self._aggregator_actor_refs.extend(
                [agg_0]
                + [
                    aggregator_actor_refs[learner_seq[i]][agg_idx]
                    for i in range(1, len(aggregator_actor_refs))
                ]
            )

    def set_env_runner_state_aggregators(self, env_runner_state_aggregator_refs):
        self._env_runner_state_aggregator_refs = env_runner_state_aggregator_refs

    def set_weights_server_actors(self, weights_server_actors):
        self._weights_server_actors = weights_server_actors

    def start_infinite_inference(self):
        self._inference_thread.start()

    def _check_env_runners_ready(self):
        ready = {}
        for aid, er in self.env_runners.items():
            status = ray.get(er.is_ready.remote())
            ready[aid] = status
            if not all(ready.values()):
                return False
        return True

    def _inference_thread_func(self):
        # Send initial (dummy) actions to all the EnvRunner actors.
        for aid, actor in self.env_runners.items():
            _action = np.random.rand(5).astype(np.float32)
            self._router_channel.write(
                actor=actor,
                message=_action.tobytes(),
            )

        # Make "iterations" comparable to non-SEED algorithms' training_steps, so
        # we can apply the same `config.broadcast_interval` settings here in SEED.
        iteration = -1
        weights_pulled = False

        while True:
            # Whenever rollout_ts_count is back to 0 (has been reset), we increase
            # `iteration` by one.
            if self._rollout_ts_count == 0:
                iteration += 1
                weights_pulled = False

            # Pull new weights.
            # TODO (sven): and merged connector states, every n times.
            if (
                iteration % self.config.broadcast_interval == 0
                and self._weights_server_actors
                and not weights_pulled
                # and self._env_runner_state_aggregator_refs
            ):
                weights_pulled = True
                # Push our connector states down to one env runner state aggregator.
                # env_runner_agg = np.random.choice(
                #    self._env_runner_state_aggregator_refs
                # )
                # env_runner_agg.merge_connector_states.remote(
                #    self.get_state(
                #        components=[
                #            COMPONENT_ENV_TO_MODULE_CONNECTOR,
                #            COMPONENT_MODULE_TO_ENV_CONNECTOR,
                #        ]
                #    ),
                #    broadcast=True,
                # )
                # Get and set weights.
                with self.metrics.log_time((TIMERS, "get_and_set_weights")):
                    learner_state = ray.get(
                        np.random.choice(self._weights_server_actors).get.remote()
                    )
                    if learner_state is not None:
                        assert isinstance(
                            learner_state[COMPONENT_RL_MODULE], ray.ObjectRef
                        )
                        self.module.set_state(
                            ray.get(learner_state[COMPONENT_RL_MODULE])[
                                DEFAULT_MODULE_ID
                            ]
                        )
                        self._weights_seq_no = learner_state[WEIGHTS_SEQ_NO]
                    # Get and set new merged env runner states.
                    # env_runner_state = ray.get(env_runner_agg.get_connector_states.remote())
                    # self.set_state(state=env_runner_state)

            with self.metrics.log_time((TIMERS, "process_single_vector_env_step")):
                self._process_single_vector_env_step()

    def make_module(self):
        try:
            module_spec: RLModuleSpec = self.config.get_rl_module_spec(
                env=self.env.unwrapped, spaces=self.get_spaces(), inference_only=True
            )
            # Build the module from its spec.
            self.module = module_spec.build()

            # TODO (sven): Move the RLModule to our device.
            # TODO (sven): In order to make this framework-agnostic, we should maybe
            #  make the RLModule.build() method accept a device OR create an additional
            #  `RLModule.to()` override.
            # self.module.to(self._device)

        # If `AlgorithmConfig.get_rl_module_spec()` is not implemented, this env runner
        # will not have an RLModule, but might still be usable with random actions.
        except NotImplementedError:
            self.module = None

    def get_spaces(self):
        return {
            INPUT_ENV_SPACES: (self.env.observation_space, self.env.action_space),
            DEFAULT_MODULE_ID: (
                self._env_to_module.observation_space,
                self.env.single_action_space,
            ),
        }

    def _process_single_vector_env_step(self):
        # receive the message from the RouterChannel
        with self.metrics.log_time((TIMERS, "mean_zeromq_read_time")):
            (
                observations,
                rewards,
                terminateds,
                truncateds,
            ), env_runner = self._router_channel.read()

        # TODO (sven): Support arbitrary obs spaces.
        observations = np.frombuffer(observations, dtype=np.float32).reshape(
            self.env.observation_space.shape
        )
        num_observations = len(observations)
        rewards = np.frombuffer(rewards, dtype=np.float64)
        terminateds = np.frombuffer(terminateds, dtype=bool)
        truncateds = np.frombuffer(truncateds, dtype=bool)

        # Create list of empty episodes if this Env actor has never sent
        # anything before.
        if env_runner._actor_id not in self._running_episodes:
            self._running_episodes[env_runner._actor_id] = [
                SingleAgentEpisode(
                    observation_space=self.env.single_observation_space,
                    action_space=self.env.single_action_space,
                )
                for _ in range(self.num_envs)
            ]
        episodes_for_inference_batch = self._running_episodes[env_runner._actor_id][:]

        # Add observations to the running episodes.
        for vec_idx, observation in enumerate(observations):
            episode = episodes_for_inference_batch[vec_idx]
            if not episode.is_reset:
                episode.add_env_reset(observation=observation)
                assert rewards[vec_idx] == 0.0
            else:
                episode.add_env_step(
                    observation=observation,
                    action=self._buffered_actions[env_runner._actor_id][vec_idx],
                    reward=float(rewards[vec_idx]),
                    terminated=bool(terminateds[vec_idx]),
                    truncated=bool(truncateds[vec_idx]),
                    extra_model_outputs=self._buffered_extra_model_outputs[
                        env_runner._actor_id
                    ][vec_idx],
                )

                # Start a new episode, if this one has been terminated.
                if terminateds[vec_idx] or truncateds[vec_idx]:
                    self._done_episodes_for_metrics.append(episode)
                    self._episodes_for_aggregators.append(episode)
                    self._running_episodes[env_runner._actor_id][
                        vec_idx
                    ] = SingleAgentEpisode(
                        observation_space=self.env.single_observation_space,
                        action_space=self.env.single_action_space,
                    )

        # TODO (sven): After adding a step to the episodes, send the latest steps
        #  also to the aggregator actors, so they can themselves build episodes
        #  and use these to build train batches.

        # Increase the inference batch counter.
        self._inference_batch_count += num_observations
        self._rollout_ts_count += num_observations
        self._episodes_for_next_inference_batch.extend(episodes_for_inference_batch)
        self._env_runners_for_actions.append(env_runner)

        # If we have "rollout_fragment_length", cut current episodes and send all chunks
        # to aggregator actor.
        if self._aggregator_actor_refs and self._rollout_ts_count >= (
            self.config.rollout_fragment_length * self.config.num_envs_per_env_runner
        ):
            for actor_id, episodes in self._running_episodes.copy().items():
                self._running_episodes[actor_id] = tree.map_structure(
                    lambda s: (
                        s.cut(len_lookback_buffer=self.config.episode_lookback_horizon)
                    ),
                    episodes,
                )
                for eps in episodes:
                    if len(eps) > 0:
                        assert not eps.is_done
                        self._ongoing_episodes_for_metrics[eps.id_].append(eps)
                        self._episodes_for_aggregators.append(eps)

            agg_actor = self._aggregator_actor_refs[
                self._curr_agg_idx % len(self._aggregator_actor_refs)
            ]
            agg_actor.push_episodes.remote(
                self._episodes_for_aggregators,
                env_runner_metrics=self.get_metrics(),
            )
            self.metrics.log_value(
                key="num_env_steps_sampled_lifetime",
                value=sum(map(len, self._episodes_for_aggregators)),
                reduce="sum",
                with_throughput=True,
            )
            self._curr_agg_idx += 1
            self._episodes_for_aggregators = []
            self._rollout_ts_count = 0

        # If we have enough samples for an inference batch, create it and perform
        # a forward pass.
        if self._inference_batch_count >= self._inference_batch_size:
            # Env-to-module connector.
            shared_data = {}
            batch = self._env_to_module(
                episodes=self._episodes_for_next_inference_batch,
                explore=self.config.explore,
                rl_module=self.module,
                shared_data=shared_data,
                metrics=self.metrics,
            )
            # Compute actions.
            with self.metrics.log_time((TIMERS, "model_forward_pass")):
                module_output = self.module.forward_exploration(batch)

            # Module-to-env connector.
            to_env = self._module_to_env(
                batch=module_output,
                episodes=self._episodes_for_next_inference_batch,
                explore=self.config.explore,
                rl_module=self.module,
                shared_data=shared_data,
                metrics=self.metrics,
            )
            all_actions = to_env.pop(Columns.ACTIONS)
            all_actions_for_env = to_env.pop(Columns.ACTIONS_FOR_ENV, all_actions)

            for idx, env_runner_for_action in enumerate(self._env_runners_for_actions):
                actions = all_actions_for_env[
                    idx * self.num_envs : (idx + 1) * self.num_envs
                ]
                # Send all computed actions back to their respective EnvRunners.
                self._router_channel.write(
                    actor=env_runner_for_action,
                    message=actions.tobytes(),
                )
                self._buffered_actions[env_runner_for_action._actor_id] = actions

                extra_model_output = [
                    {k: v[idx * self.num_envs + vec_idx] for k, v in to_env.items()}
                    for vec_idx in range(self.num_envs)
                ]
                # extra_model_output[WEIGHTS_SEQ_NO] = self._weights_seq_no
                self._buffered_extra_model_outputs[
                    env_runner_for_action._actor_id
                ] = extra_model_output

            self._episodes_for_next_inference_batch = []
            self._inference_batch_count = 0
            self._env_runners_for_actions = []

    def get_metrics(self) -> ResultDict:
        # Compute per-episode metrics (only on already completed episodes).
        for eps in self._done_episodes_for_metrics:
            assert eps.is_done
            episode_length = len(eps)
            episode_return = eps.get_return()
            episode_duration_s = eps.get_duration_s()
            # Don't forget about the already returned chunks of this episode.
            if eps.id_ in self._ongoing_episodes_for_metrics:
                for eps2 in self._ongoing_episodes_for_metrics[eps.id_]:
                    episode_length += len(eps2)
                    episode_return += eps2.get_return()
                    episode_duration_s += eps2.get_duration_s()
                del self._ongoing_episodes_for_metrics[eps.id_]

            SingleAgentEnvRunner._log_episode_metrics(
                self, episode_length, episode_return, episode_duration_s
            )

        # Now that we have logged everything, clear cache of done episodes.
        self._done_episodes_for_metrics.clear()

        # Return reduced metrics.
        return self.metrics.reduce()
