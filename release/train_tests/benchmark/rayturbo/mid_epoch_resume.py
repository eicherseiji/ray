import json
import logging
import os
import pprint
import time

import ray.train
from ray._private.test_utils import safe_write_to_results_json
from ray.train.torch import TorchTrainer
from ray.train.v2._internal.util import date_str

from config import cli_to_config, DataloaderType
from constants import DatasetKey
from image_classification.factory import ImageClassificationFactory
from logger_utils import ContextLoggerAdapter
from ray_dataloader_factory import RayDataLoaderFactory
from runner import VanillaTorchRunner
from train_benchmark import METRICS_OUTPUT_PATH


logger = ContextLoggerAdapter(logging.getLogger(__name__))


class TorchRunnerWithDataCheckpointing(VanillaTorchRunner):
    def _setup(self):
        super()._setup()
        self._should_inject_error = True

    def _save_training_state(self, local_dir: str):
        super()._save_training_state(local_dir)

        # If we're running the baseline of skipping batches upon resume,
        # disable data checkpointing.
        if self.benchmark_config.skip_batches_upon_resume:
            logger.info(
                "Running the batch skipping upon resume baseline. "
                "Data checkpointing is disabled. Skipping data checkpoint save."
            )
            return

        ds = ray.train.get_dataset_shard(DatasetKey.TRAIN)
        if ray.train.get_context().get_world_rank() == 0:
            with open(os.path.join(local_dir, "data_state.json"), "w") as f:
                json.dump(ds.state_dict(), f)
            logger.info(f"Saved data state dict: {ds.state_dict()}")

    def _load_training_state(self, local_dir: str):
        super()._load_training_state(local_dir)

        # If we restored, don't inject another error.
        self._should_inject_error = False

        if self.benchmark_config.skip_batches_upon_resume:
            logger.info(
                "Running the batch skipping upon resume baseline. "
                "Data checkpointing is disabled. Skipping data checkpoint load."
            )
            return

        with open(os.path.join(local_dir, "data_state.json"), "r") as f:
            data_state_dict = json.load(f)
        # HACK: This first call to `get_dataset_shard` loads the dataset shard
        # with the state dict.
        # Subsequent calls to `get_dataset_shard` return the same cached shard,
        # so it's okay if we don't pass in the state dict again.
        ray.train.get_dataset_shard(DatasetKey.TRAIN, state_dict=data_state_dict)

        logger.info(f"Loaded data state dict: {data_state_dict}")

    def _checkpoint(self, *args, **kwargs):
        super()._checkpoint(*args, **kwargs)

        if (
            self._global_rows_processed_this_epoch >= 750_000
            and self._should_inject_error
        ):
            raise RuntimeError(
                f"[Mid-epoch error] global_rows_processed={self._global_rows_processed_this_epoch}"
            )


def train_fn_per_worker(config):
    factory = config["factory"]

    runner = TorchRunnerWithDataCheckpointing(factory)
    runner.run()

    metrics = runner.get_metrics()
    if ray.train.get_context().get_world_rank() == 0:
        with open(METRICS_OUTPUT_PATH, "w") as f:
            json.dump(metrics, f)


def main():
    config = cli_to_config()
    print("\nBenchmark config:\n" + pprint.pformat(config.__dict__, indent=2))

    factory = ImageClassificationFactory(config)

    if config.dataloader_type != DataloaderType.RAY_DATA:
        raise ValueError(
            "This test only runs with Ray Data: --dataloader_type=ray_data"
        )

    # If we're running the skip batches baseline,
    # disable data checkpointing.
    running_skip_batches_baseline = config.skip_batches_upon_resume
    if running_skip_batches_baseline:
        print("Running the skip batches baseline. Data checkpointing is disabled!")

    dataset_checkpoint_configs = (
        {
            DatasetKey.TRAIN: ray.train.DatasetCheckpointConfig(
                id_column="generated_id", generate_id_column=True
            ),
        }
        if not running_skip_batches_baseline
        else {}
    )

    dataloader_factory = factory.get_dataloader_factory()
    assert isinstance(dataloader_factory, RayDataLoaderFactory)
    # NOTE: The dataset map tasks must pass the generated ID column
    # through all the way to the end of the pipeline.
    datasets = dataloader_factory.get_ray_datasets()
    data_config = ray.train.DataConfig(
        dataset_checkpoint_configs=dataset_checkpoint_configs
    )

    # Hard code some configurations for the test:
    config.max_failures = 1
    config.checkpoint_every_n_steps = 100

    start_time = time.perf_counter()
    trainer = TorchTrainer(
        train_loop_per_worker=train_fn_per_worker,
        train_loop_config={"factory": factory},
        scaling_config=ray.train.ScalingConfig(
            num_workers=config.num_workers,
            use_gpu=not config.mock_gpu,
            resources_per_worker={"MOCK_GPU": 1} if config.mock_gpu else None,
        ),
        run_config=ray.train.RunConfig(
            storage_path=f"{os.environ['ANYSCALE_ARTIFACT_STORAGE']}/rayturbo_train_benchmark/",
            name=f"mid_epoch_resume-{config.task}-{date_str(include_ms=True)}",
            failure_config=ray.train.FailureConfig(max_failures=config.max_failures),
        ),
        datasets=datasets,
        dataset_config=data_config,
    )
    trainer.fit()
    end_time = time.perf_counter()
    e2e_time = end_time - start_time

    with open(METRICS_OUTPUT_PATH, "r") as f:
        metrics = json.load(f)

    metrics["e2e_time"] = e2e_time

    # Update the metrics with the time to first batch, which is the main metric
    # to track for mid-epoch resume.
    if running_skip_batches_baseline:
        # For the skip batches baseline, the time to first batch is the total time
        # it takes to skip batches. Note that "iter_first_batch" time is already
        # included in "iter_skip_batch".
        metrics["time_to_first_batch_s"] = metrics["train/iter_skip_batch-total"]
    else:
        # Take the max first batch iteration recorded.
        # This will be the first batch iteration time after restarting
        # and loading the data checkpoint.
        metrics["time_to_first_batch_s"] = metrics["train/iter_first_batch-max"]

    # Write to release test result file.
    safe_write_to_results_json(metrics)

    final_metrics_str = (
        f"\nTotal training time: {e2e_time} seconds\n"
        "Final metrics:\n" + "-" * 80 + "\n" + pprint.pformat(metrics) + "\n" + "-" * 80
    )
    print(final_metrics_str)


if __name__ == "__main__":
    # Workers need to access the working directory module.
    ray.init(runtime_env={"working_dir": os.path.dirname(os.path.dirname(__file__))})
    main()
