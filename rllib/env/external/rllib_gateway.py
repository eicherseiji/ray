import logging
import pickle
import socket
import threading
import time

import numpy as np

from ray.rllib.core import COMPONENT_RL_MODULE, Columns
from ray.rllib.env.external.rllink import (
    RLlink,
    get_rllink_message,
    send_rllink_message,
)
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.rllib.utils.metrics import WEIGHTS_SEQ_NO
from ray.util.annotations import DeveloperAPI

logger = logging.getLogger("ray.rllib")


@DeveloperAPI
class RLlibGateway:
    """Gateway class for external, non-pythoninc, simulators to connect to RLlib.

    As long as there is a path to bind python code into your simulator's language, for
    example C++, you should be able to use the simulator very easily in connection with
    an RLlib experiment.

    You should use the gateway as follows in your C++ code:

    .. code-block:: c++

        #include <pybind11/embed.h>
        #include <pybind11/stl.h>

        namespace py = pybind11;

        int main(int argc, char** argv)
        {
            // Proper interpreter init (RAII-safe).
            py::scoped_interpreter guard{};

            // Import RLlibGateway class.
            py::object rllib_gateway_class = py::module_::import(
                "ray.rllib.env.external.rllib_gateway"
            ).attr("RLlibGateway");
            py::object rllib = rllib_gateway_class();

            // Assuming, you have a CartPole simulator class, create it and reset.
            CartPole env;
            env.reset();
            float total_reward = 0.0;
            int eps = 0;

            // Endless loop through an infinite number of episodes.
            while (true)
            {
                // Send previous reward (result of the previous action taken) and
                // current observation to get_action. If the episode has just been
                // reset, the gateway won't log it (for example, set it to 0.0).
                try
                {
                    py::gil_scoped_acquire gil;
                    py::object action = rllib.attr("get_action")(
                        env.reward,
                        env.observation
                    );
                    // Apply the locally computed action in the simulation.
                    env.step(action.cast<int>());
                }
                catch (const py::error_already_set& e)
                {
                    std::cerr << "[Python error in get_action]\n" << e.what() << std::endl;
                    break;
                }

                // Send last reward and last observation to episode_done().
                if (env.terminated || env.truncated)
                {
                    try {
                        py::gil_scoped_acquire gil;
                        rllib.attr("episode_done")(
                            env.reward,
                            env.observation,
                            env.truncated
                        );
                    }
                    catch (const py::error_already_set& e) {
                        std::cerr << "[Python error in get_action (episode done)]\n" << e.what() << std::endl;
                        break;
                    }
                    // Reset episode to start a new one.
                    env.reset();
                    // Report episode's total return.
                    std::cout << "Episode " << eps << " return: " << total_reward << "\n";
                    total_reward = 0.0f;
                    eps += 1;
                }
                total_reward += env.reward;
            }
            return 0;
        }

    The gateway automatically tries to connect to the given address and port, where
    an RLlib EnvRunner should be listening as a service.

    Once connected to an RLlib EnvRunner, the gateway receives the RLlib algo config
    and the current state of the EnvRunner (model weights and connector states).
    It then constructs the local RLModule and connector pipelines (env-to-module
    and module-to-env), through which it's enabled to compute actions locally.

    As a user of the gateway, its `get_action` and `episode_done` APIs are the two
    methods you need to call from within your simulator's code (for example C++),
    always passing it the previously received reward and the current observation. In
    case of `episode_done`, you have to specify also whether the episode is truncated
    or not.
    Directly a simulator-episode reset, the reward passed into the first `get_action`
    call should be 0.0 and the observation passed in should be the reset/first
    observation with which the episode starts.

    The gateway also takes care of frequently sending batches of recorded episodes
    back to the connected `EnvRunner` for model updating purposes and then waits for
    the latest model weights and connector states.
    """

    def __init__(
        self,
        address: str = "localhost",
        port: int = 5556,
        log_level: str = "WARNING",
    ):
        """Initializes a RLlibGateway instance.

        Args:
            address: The address under which to connect to the RLlib EnvRunner.
            port: The post to connect to.
            log_level: The log level
        """
        logger.setLevel(log_level)

        # The open socket connection to an RLlib EnvRunner.
        self._address = address
        self._port = port
        self._sock = None
        self._should_exit = False

        # The RLlib config from the ray cluster.
        self._config = None
        # RLlib SingleAgentEpisode collection buckets.
        self._episodes = []
        # The timesteps sampled thus far.
        self._timesteps = 0
        # EnvToModule connector pipeline.
        self._env_to_module = None
        # ModuleToEnv connector pipeline.
        self._module_to_env = None
        # The RLModule for action computations.
        self._rl_module = None
        self._weights_seq_no = 0

        self._prev_action = None
        self._prev_extra_model_outputs = None

        self._is_initializing = False
        self._is_initialized = False

        self._connection_thread = None
        self._try_connecting_and_initializing()

    @property
    def is_initializing(self):
        """Returns True, if this Gateway is in the process of initializing."""
        return self._is_initializing

    @property
    def is_initialized(self):
        """Returns True, if this Gateway has an RLModule and connectors."""
        return self._is_initialized

    @property
    def timesteps(self):
        return self._timesteps

    def get_action(
        self,
        prev_reward,
        prev_observation,
    ):
        """Computes and returns a new action, given an observation.

        Args:
            prev_reward: The reward received after the previously computed action
                (returned from this method in the previous call).
            prev_observation: The current observation, from which the action should be
                computed. Note that first, `observation`, the previously returned
                action, `prev_reward`, and `terminated/truncated` are logged with the running
                episode through `Episode.add_env_step()`, then the env-to-module
                connector creates the inference forward batch for the RLModule based on
                this running episode.
        """
        return self._step_helper(prev_reward, prev_observation)

    def episode_done(self, final_reward, final_observation, truncated: bool):
        """Logs the last step in an episode and starts a new one.

        Args:
            final_reward: The final reward received in the episode.
            final_observation: The final observation in the episode.
            truncated: Whether the episode is truncated. If True,
                `final_observation` is the observation right before the truncation point
                and `final_reward` is the last reward that the agent receives in the
                episode. A truncated episode's final observation should still be used to
                compute value function estimates at the truncation point.
        """
        # Forward to `self.get_action()` with the correct terminated/truncated args.
        self._step_helper(
            final_reward,
            final_observation,
            terminated=not truncated,
            truncated=truncated,
        )

    def cleanup(self):
        self._should_exit = True
        if self._sock is not None:
            # Try closing and invalidating socket.
            try:
                self._sock.close()
                self._sock = None
            except Exception:
                time.sleep(1)
            time.sleep(2)

    def _try_connecting_and_initializing(self):
        assert not self.is_initializing

        if self._connection_thread is None:
            self._connection_thread = threading.Thread(
                target=self._connect_to_server_thread_func,
                args=(self._address, self._port),
            )
            self._connection_thread.start()

    def _connect_to_server_thread_func(self, address, port):
        # Try initializing the Gateway.
        while not self._should_exit:
            self._is_initializing = False
            # Try connecting to (RLlib) server.
            while not self._should_exit:
                try:
                    logger.info(f"Trying to connect to {address}:{port} ...")
                    _sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    _sock.settimeout(60.0)
                    _sock.connect((address, port))
                    logger.info(f"Connected to server at {address}:{port} ...")
                    self._is_initializing = True
                    self._sock = _sock
                    break

                except ConnectionRefusedError:
                    time.sleep(5)
            # We should exit.
            else:
                break

            # Send ping-pong.
            msg_type, msg_body = self._try_send_receive_rllink_msg(
                {"type": RLlink.PING.name},
            )
            # Error -> Retry connecting to server.
            if msg_type != RLlink.PONG:
                continue
            logger.info("\tPING/PONG ok ...")

            # Request config.
            msg_type, msg_body = self._try_send_receive_rllink_msg(
                {"type": RLlink.GET_CONFIG.name}
            )
            # Error -> Retry connecting to server.
            if msg_type != RLlink.SET_CONFIG:
                continue
            # TODO (sven): Make AlgorithmConfig msgpack'able by making it a
            #  Checkpointable with a pickle-independent state.
            self._config = pickle.loads(msg_body["config"])
            # Create the RLModule and connector pipelines.
            self._env_to_module = self._config.build_env_to_module_connector()
            rl_module_spec = self._config.get_rl_module_spec()
            self._rl_module = rl_module_spec.build()
            self._module_to_env = self._config.build_module_to_env_connector()
            logger.info("\tGET_CONFIG ok (built connectors and module) ...")

            # Request EnvRunner state (incl. model weights).
            msg_type, msg_body = self._try_send_receive_rllink_msg(
                {"type": RLlink.GET_STATE.name}
            )
            # Error -> Retry connecting to server.
            if msg_type != RLlink.SET_STATE:
                continue
            self._set_state(msg_body["state"])
            logger.info("\tSET_STATE ok ...")

            # Set this Gateway to `initialized` and return from the thread.
            self._is_initialized = True
            self._is_initializing = False
            self._connection_thread = None
            return

        self.cleanup()

    def _step_helper(
        self,
        prev_reward,
        prev_observation,
        terminated: bool = False,
        truncated: bool = False,
    ):
        # Error, if user requests action before initialization.
        if not self.is_initialized:
            raise RuntimeError(
                "Gateway not initialized yet! Needs to connect to RLlib "
                "server before actions can be computed."
            )
        # If we are in the process of initializing, just wait a bit.
        elif self.is_initializing:
            while self.is_initializing:
                time.sleep(0.01)

        # C++ may send observation tensors as std::vector<float> (which get translated
        # into python lists).
        if isinstance(prev_observation, list):
            prev_observation = np.array(prev_observation, np.float32)

        # Episode logging.
        if len(self._episodes) == 0 or self._episodes[-1].is_done:
            self._episodes.append(SingleAgentEpisode())
            self._episodes[-1].add_env_reset(observation=prev_observation)
        else:
            # Log timestep to current episode.
            self._episodes[-1].add_env_step(
                observation=prev_observation,
                action=self._prev_action,
                reward=prev_reward,
                terminated=terminated,
                truncated=truncated,
                extra_model_outputs=self._prev_extra_model_outputs,
            )
            self._timesteps += 1

            # TODO (sven): If enough timesteps have been collected, send out episodes
            #  through socket to RLlib server for training.
            # We collected enough samples -> Send them to server.
            if self._timesteps >= self._config.get_rollout_fragment_length():
                assert sum(map(len, self._episodes)) == (
                    self._config.get_rollout_fragment_length()
                )

                # Send the data to the server.
                # On-policy: Block until response received back from server. Note that
                # this may halt the simulation calling this function (`get_action`) for
                # a while.
                while self.is_initializing:
                    time.sleep(0.01)
                msg_type, msg_body = self._try_send_receive_rllink_msg(
                    {
                        "type": RLlink.EPISODES_AND_GET_STATE.name,
                        "episodes": [e.get_state() for e in self._episodes],
                        "timesteps": self._timesteps,
                    },
                )
                # We are forced to sample on-policy. Have to wait for a response
                # with the state (weights) in it.
                if msg_type != RLlink.SET_STATE:
                    logger.warning(
                        "Can't SET_STATE, connection error to RLlib "
                        f"server! Trying to reconnect and reinitialize "
                        f"...\nmsg={msg_body}"
                    )
                    # Restart the connection and re-initialization thread.
                    self._try_connecting_and_initializing()
                else:
                    self._set_state(msg_body["state"])

                self._timesteps = 0
                self._episodes = [
                    eps.cut(len_lookback_buffer=self._config.episode_lookback_horizon)
                    for eps in self._episodes
                    if not eps.is_done
                ]

        # Model forward pass.
        shared_data = {}
        to_module = self._env_to_module(
            episodes=[self._episodes[-1]],
            rl_module=self._rl_module,
            explore=True,
            shared_data=shared_data,
        )
        if self._config.explore:
            model_outs = self._rl_module.forward_exploration(to_module)
        else:
            model_outs = self._rl_module.forward_inference(to_module)
        # Add `module_outs` to `batch`.
        to_module.update(model_outs)
        to_env = self._module_to_env(
            episodes=[self._episodes[-1]],
            batch=to_module,
            rl_module=self._rl_module,
            explore=True,
            shared_data=shared_data,
        )
        # Extract the action that should be applied in the env.
        self._prev_action = to_env.pop(Columns.ACTIONS)
        action_for_env = to_env.pop(Columns.ACTIONS_FOR_ENV, self._prev_action)[0]
        self._prev_action = self._prev_action[0]

        extra_model_output = {k: v[0] for k, v in to_env.items()}
        extra_model_output[WEIGHTS_SEQ_NO] = self._weights_seq_no

        # Store action for next timestep's logging into the episode.
        self._prev_extra_model_outputs = extra_model_output

        # And return the action.
        return action_for_env

    def _set_state(self, msg_body):
        # TODO (sven): Add once our EnvRunner publishes these (right now, it doesn't
        #  even have its own connectors, for simplicity).
        # self._env_to_module.set_state(msg_body[COMPONENT_ENV_TO_MODULE_CONNECTOR])
        # self._module_to_env.set_state(msg_body[COMPONENT_MODULE_TO_ENV_CONNECTOR])
        self._rl_module.set_state(msg_body[COMPONENT_RL_MODULE])
        self._weights_seq_no = msg_body[WEIGHTS_SEQ_NO]

    def _try_send_receive_rllink_msg(self, msg):
        try:
            send_rllink_message(self._sock, msg)
            msg_type, msg_body = get_rllink_message(self._sock)
        except ConnectionError as e:
            msg_type = e.__class__
            msg_body = str(e)
            # Try closing and invalidating socket.
            try:
                self._sock.close()
                self._sock = None
            except Exception:
                time.sleep(1)
            time.sleep(2)

        return msg_type, msg_body
