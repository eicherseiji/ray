import gymnasium as gym
import numpy as np

from ray.rllib.algorithms.seed.seed import SEEDConfig
from ray.rllib.connectors.env_to_module.frame_stacking import FrameStackingEnvToModule
from ray.rllib.connectors.learner.frame_stacking import FrameStackingLearner
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.env.wrappers.atari_wrappers import wrap_atari_for_new_api_stack
from ray.rllib.utils.test_utils import add_rllib_example_script_args, seed_testing_args
from ray.tune.registry import register_env

parser = add_rllib_example_script_args(
    default_reward=20.0,
    default_timesteps=10000000,
)
parser = seed_testing_args(parser)
parser.add_argument(
    "--num-batch-dispatchers",
    type=int,
    default=1,
    help="Number of batch dispatch actors (layer in between "
    "AggregatorActor and Learner).",
)
parser.add_argument(
    "--num-env-runner-state-aggregators",
    type=int,
    default=1,
    help="Number of EnvRunner state aggregator actors.",
)
parser.add_argument(
    "--num-weights-server-actors",
    type=int,
    default=1,
    help="Number of weights server actors.",
)
parser.add_argument(
    "--sync-freq",
    type=int,
    default=10,
    help="Synchronization frequency between the different actor types of the pipeline.",
)
parser.add_argument(
    "--num-aggregator-actors-per-learner2",
    type=int,
    default=1,
)
parser.set_defaults(
    enable_new_api_stack=True,
    env="ale_py:ALE/Pong-v5",
    num_envs_per_env_runner=5,
)
args = parser.parse_args()
args.create_env_on_local_worker = False


def _make_env_to_module_connector(env):
    return FrameStackingEnvToModule(num_frames=4)


def _make_learner_connector(input_observation_space, input_action_space):
    return FrameStackingLearner(num_frames=4)


def _env_creator(cfg):
    return wrap_atari_for_new_api_stack(
        gym.make(args.env, **cfg, **{"render_mode": "rgb_array"}),
        dim=64,
        framestack=None,
    )


register_env("env", _env_creator)


config = (
    SEEDConfig()
    .resources(
        inference_num_cpus_per_process=args.inference_num_cpus_per_process,
        inference_num_gpus_per_process=args.inference_num_gpus_per_process,
        inference_processes_scheduling_strategy="SPREAD",
    )
    .environment(
        "env",
        env_config={
            # Make analogous to old v4 + NoFrameskip.
            "frameskip": 1,
            "full_action_space": False,
            "repeat_action_probability": 0.0,
        },
        clip_rewards=True,
        observation_space=gym.spaces.Box(-1.0, 1.0, (64, 64, 4), np.float32),
    )
    .env_runners(
        env_to_module_connector=_make_env_to_module_connector,
    )
    .inference(
        inference_batch_size=args.inference_batch_size,
        n_inference_processes=args.n_inference_processes,
    )
    .training(
        learner_connector=_make_learner_connector,
        num_aggregator_actors_per_inf_appo_learner=args.num_aggregator_actors_per_learner2,
        num_weights_server_actors=args.num_weights_server_actors,
        num_batch_dispatchers=args.num_batch_dispatchers,
        num_env_runner_state_aggregators=args.num_env_runner_state_aggregators,
        pipeline_sync_freq=args.sync_freq,
        train_batch_size_per_learner=500,
        target_network_update_freq=2,
        lr=0.0005 * ((args.num_learners or 1) ** 0.5),
        vf_loss_coeff=1.0,
        entropy_coeff=[[0, 0.01], [3000000, 0.0]],  # <- crucial parameter to finetune
        # Only update connector states and model weights every n training_step calls.
        broadcast_interval=5,
        circular_buffer_num_batches=4,
        circular_buffer_iterations_per_batch=2,
    )
    .rl_module(
        model_config=DefaultModelConfig(
            vf_share_layers=True,
            conv_filters=[(16, 4, 2), (32, 4, 2), (64, 4, 2), (128, 4, 2)],
            conv_activation="relu",
            head_fcnet_hiddens=[256],
        )
    )
)


if __name__ == "__main__":
    from ray.rllib.utils.test_utils import run_rllib_example_script_experiment

    run_rllib_example_script_experiment(config, args)
