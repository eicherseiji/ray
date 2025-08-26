from ray.rllib.algorithms.seed import SEEDConfig
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.utils.test_utils import add_rllib_example_script_args, seed_testing_args

parser = add_rllib_example_script_args(
    default_timesteps=2000000,
    default_reward=450,
    default_iters=200,
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
# Use `parser` to add your own custom command line options to this script
# and (if needed) use their values to set up `config` below.
args = parser.parse_args()


config = (
    SEEDConfig()
    .framework(torch_skip_nan_gradients=True)
    .environment(
        "CartPole-v1",
    )
    .training(
        num_aggregator_actors_per_inf_appo_learner=(
            args.num_aggregator_actors_per_learner2
        ),
        num_weights_server_actors=args.num_weights_server_actors,
        num_batch_dispatchers=args.num_batch_dispatchers,
        num_env_runner_state_aggregators=args.num_env_runner_state_aggregators,
        pipeline_sync_freq=args.sync_freq,
        circular_buffer_iterations_per_batch=2,
        vf_loss_coeff=0.05,
        entropy_coeff=0.0,
    )
    .resources(
        num_cpus_for_main_process=args.num_cpus_for_main_process,
        inference_num_cpus_per_process=args.inference_num_cpus_per_process,
        inference_num_gpus_per_process=args.inference_num_gpus_per_process,
    )
    .inference(
        inference_batch_size=args.inference_batch_size,
        n_inference_processes=args.n_inference_processes,
    )
    .rl_module(
        model_config=DefaultModelConfig(vf_share_layers=True),
    )
)


if __name__ == "__main__":
    from ray.rllib.utils.test_utils import run_rllib_example_script_experiment

    run_rllib_example_script_experiment(config, args)
