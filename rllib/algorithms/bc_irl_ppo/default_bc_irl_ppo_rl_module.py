import gymnasium as gym
import numpy as np

from enum import Enum

from ray.rllib.core.columns import Columns
from ray.rllib.core.models.base import Encoder, Model
from ray.rllib.core.models.configs import MLPEncoderConfig, MLPHeadConfig
from ray.rllib.core.models.specs.typing import SpecType
from ray.rllib.core.rl_module import RLModule
from ray.rllib.utils.annotations import override


class DefaultBCIRLRewardModelType(str, Enum):
    """Defines the default reward model types.

    ACTION: Action, current state, and next state input.
    NEXT_STATE: Next state input.
    CURR_NEXT_STATE: Current state, and next state input.
    """

    ACTION = "action"
    NEXT_STATE = "next_state"
    CURR_NEXT_STATE = "curr_next_state"


class DefaultBCIRLRewardRLModule(RLModule):
    @override(RLModule)
    def setup(self):
        # If an action reward model should be used or a state-only one.
        try:
            self.reward_type = DefaultBCIRLRewardModelType(
                self.model_config["reward_type"]
            )
        except ValueError:
            raise ValueError(
                f"Invalid reward model type: {self.reward_type.value}. "
                f"Allowed types: {[t.value for t in DefaultBCIRLRewardModelType]}"
            )

        # Set the latent space dimension.
        self.latent_dims = self.model_config["fcnet_hiddens"][-1]
        # Configure the reward-function encoder.
        self.rf_encoder = self._build_rf_encoder(framework=self.framework)
        # Configure the reward-function head.
        self.rf = self._build_rf_head(framework=self.framework)

    @override(RLModule)
    def get_initial_state(self) -> dict:
        """Defines the initial state for stateful RLModules."""
        return {}

    @override(RLModule)
    def input_specs_train(self) -> SpecType:
        """Defines the input specs for the train forward pass."""
        # Note, the reward function inputs actual state, action and, next state.
        if self.reward_type == DefaultBCIRLRewardModelType.ACTION:
            return [
                Columns.OBS,
                Columns.ACTIONS,
                Columns.NEXT_OBS,
            ]
        elif self.reward_type == DefaultBCIRLRewardModelType.CURR_NEXT_STATE:
            return [Columns.OBS, Columns.NEXT_OBS]
        elif self.reward_type == DefaultBCIRLRewardModelType.NEXT_STATE:
            return [Columns.NEXT_OBS]

    @override(RLModule)
    def output_specs_train(self) -> SpecType:
        """Defines the output specs for the train forward pass."""
        return [
            Columns.REWARDS,
        ]

    def _build_rf_encoder(self, framework: str) -> Encoder:

        if self.reward_type == DefaultBCIRLRewardModelType.ACTION:
            # The input dimension reserved for the actions is either the number of
            # actions for discrete spaces (one-hot encoded) or the first shape dimension
            # for 1-dimensional Box spaces.
            required_action_dim = (
                self.action_space.shape[0]
                if isinstance(self.action_space, gym.spaces.Box)
                else self.action_space.n
            )
        else:
            required_action_dim = 0

        # Encoder input for the reward model contains state, action, and next state. We
        # need to infer the shape for the input from the state and action spaces.
        if (
            isinstance(self.observation_space, gym.spaces.Box)
            and len(self.observation_space.shape) == 1
        ):
            input_space = gym.spaces.Box(
                -np.inf,
                np.inf,
                (
                    self.observation_space.shape[0]
                    * (
                        1
                        + int(
                            self.reward_type
                            == DefaultBCIRLRewardModelType.CURR_NEXT_STATE
                        )
                    )
                    + required_action_dim,
                ),
                dtype=np.float32,
            )
        # Other observations spaces are at this moment not implemented.
        else:
            ValueError("The observation space is not supported by RLlib's BC-IRL-PPO.")

        self.rf_encoder_hiddens = self.model_config["fcnet_hiddens"][:-1]
        self.rf_encoder_activation = self.model_config["fcnet_activation"]

        # Now define the encoder of the reward-function.
        self.rf_encoder_config = MLPEncoderConfig(
            input_dims=input_space.shape,
            hidden_layer_dims=self.rf_encoder_hiddens,
            hidden_layer_activation=self.rf_encoder_activation,
            output_layer_dim=self.latent_dims,
            output_layer_activation=self.rf_encoder_activation,
        )

        # Return the built reward-function encoder.
        return self.rf_encoder_config.build(framework=framework)

    def _build_rf_head(self, framework: str) -> Model:
        """Builds the reward-function head."""

        self.rf_head_hiddens = self.model_config["head_fcnet_hiddens"]
        self.rf_head_activation = self.model_config["head_fcnet_activation"]

        # Define the head for the reward-function here. Note, the encoder
        # needs to be defined later because at initialization
        self.rf_head_config = MLPHeadConfig(
            input_dims=(self.latent_dims,),
            hidden_layer_dims=self.rf_head_hiddens,
            hidden_layer_activation=self.rf_head_activation,
            output_layer_activation="linear",
            output_layer_dim=1,
        )
        return self.rf_head_config.build(framework=framework)
