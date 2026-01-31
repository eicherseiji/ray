from ray.rllib.algorithms.infinite_appo import InfiniteAPPOConfig
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.utils.test_utils import add_rllib_example_script_args

parser = add_rllib_example_script_args(default_timesteps=2000000)
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
# Use `parser` to add your own custom command line options to this script
# and (if needed) use their values to set up `config` below.
args = parser.parse_args()


config = (
    InfiniteAPPOConfig()
    .framework(torch_skip_nan_gradients=True)
    .environment("CartPole-v1")
    .training(
        num_aggregator_actors_per_inf_appo_learner=(
            args.num_aggregator_actors_per_learner2
        ),
        num_weights_server_actors=args.num_weights_server_actors,
        num_batch_dispatchers=args.num_batch_dispatchers,
        num_env_runner_state_aggregators=args.num_env_runner_state_aggregators,
        pipeline_sync_freq=args.sync_freq,
        vf_loss_coeff=0.005,
        entropy_coeff=0.0,
        # Only update connector states and model weights every n training_step calls.
        broadcast_interval=1,
    )
    .rl_module(
        model_config=DefaultModelConfig(vf_share_layers=True),
    )
)


if __name__ == "__main__":
    from ray.rllib.utils.test_utils import run_rllib_example_script_experiment

    run_rllib_example_script_experiment(config, args)
