import argparse
import os
import time
import numpy
from typing import Optional
from ray.data.datasource import WriteResult
from benchmark import Benchmark, BenchmarkMetric
import ray
from ray.data import DataContext
from ray.anyscale.data.checkpoint import CheckpointBackend, CheckpointConfig

from ray.data._internal.datasource.parquet_datasink import ParquetDatasink


def _parse_checkpoint_config(args: argparse.Namespace) -> Optional[CheckpointConfig]:
    backend_str = args.checkpoint_backend.upper()
    if backend_str == "NONE":
        return None
    elif backend_str == "FILE_STORAGE":
        backend = CheckpointBackend.FILE_STORAGE
    elif backend_str == "CLOUD_OBJECT_STORAGE":
        backend = CheckpointBackend.CLOUD_OBJECT_STORAGE
    else:
        raise ValueError(f"Unknown checkpoint backend: {backend_str}")

    if args.generated_id_column:
        id_column: Optional[str] = None
        generated_id_column: Optional[str] = args.generated_id_column
    else:
        id_column: Optional[str] = "id"
        generated_id_column: Optional[str] = None

    return CheckpointConfig(
        id_column=id_column,
        generated_id_column=generated_id_column,
        checkpoint_path=args.checkpoint_output_path,
        override_backend=backend,
    )


def run_dataset(
    checkpoint_config: Optional[CheckpointConfig],
    input_data_path: str,
    transform_sleep_s: float,
    inference_sleep_s: float,
    inference_batch_size: int,
    inference_concurrency: int,
    data_output_path: str,
    num_output_files: int,
    num_rows: Optional[int] = None,
) -> int:
    ctx = DataContext.get_current()
    ctx.checkpoint_config = checkpoint_config
    ctx.checkpoint_enabled_override = False

    # Make read_parquet and transform fuse.
    ctx._enable_read_files_fusion_override = True

    ds = ray.data.read_parquet(input_data_path)
    if not num_rows:
        num_rows = ds.count()

    def transform(batch):
        time.sleep(transform_sleep_s)
        return batch

    ds = ds.map_batches(transform, batch_size=None)

    class Inference:
        INFER_RESULT_DIMENSION = 16

        def __call__(self, batch):
            time.sleep(inference_sleep_s)
            batch["inference"] = numpy.random.random(
                (len(batch["data"]), self.INFER_RESULT_DIMENSION)
            )
            # Remove the data column to make the write op run faster.
            # We want the Inference op to be the main bottleneck of the pipeline.
            del batch["data"]
            return batch

    ds = ds.map_batches(
        Inference,
        batch_size=inference_batch_size,
        concurrency=inference_concurrency,
        num_gpus=1,
    )

    # Patch `on_write_complete` to get the WriteResult.
    # TODO(hchen): make `write_parquet` expose the WriteResult directly.
    num_rows_written = None
    original_on_write_complete = ParquetDatasink.on_write_complete

    def patched_on_write_complete(self, write_result: WriteResult[None]):
        nonlocal num_rows_written
        num_rows_written = write_result.num_rows
        return original_on_write_complete(self, write_result)

    ParquetDatasink.on_write_complete = patched_on_write_complete

    try:
        ds.write_parquet(
            data_output_path,
            min_rows_per_file=num_rows // num_output_files,
        )
        return int(num_rows_written)
    finally:
        ParquetDatasink.on_write_complete = original_on_write_complete


def run_checkpoints_benchmark(
    benchmark: Benchmark,
    checkpoint_config: Optional[CheckpointConfig],
    input_data_path: str,
    transform_sleep_s: float,
    inference_sleep_s: float,
    inference_batch_size: int,
    inference_concurrency: int,
    data_output_path: str,
    num_output_files: int,
    benchmark_name: str = "",
    num_rows: Optional[int] = None,
):
    def run():
        start_time = time.time()
        print(f"[{benchmark_name}] Running dataset from scratch")
        if checkpoint_config is not None:
            # Keep the checkpoint files. We'll test loading them in the second run.
            checkpoint_config.delete_checkpoint_on_success = False
            print(f"checkpoint_path = {checkpoint_config.checkpoint_path}")

        num_rows_written = run_dataset(
            checkpoint_config,
            input_data_path,
            transform_sleep_s,
            inference_sleep_s,
            inference_batch_size,
            inference_concurrency,
            data_output_path,
            num_output_files,
            num_rows=num_rows,
        )
        runtime = time.time() - start_time
        print(f"[{benchmark_name}] dataset finished in {runtime:.2f} seconds")

        benchmark_results = {
            BenchmarkMetric.RUNTIME: runtime,
            BenchmarkMetric.THROUGHPUT: num_rows_written // runtime,
            "num_rows_written": num_rows_written,
        }

        if checkpoint_config is not None:
            print(f"[{benchmark_name}] Rerunning dataset with full checkpoint")
            start_time = time.time()
            num_rows_written = run_dataset(
                checkpoint_config,
                input_data_path,
                transform_sleep_s,
                inference_sleep_s,
                inference_batch_size,
                inference_concurrency,
                data_output_path,
                num_output_files,
                num_rows=num_rows,
            )
            assert num_rows_written == 0
            runtime = time.time() - start_time
            # TODO: capture checkpoint loading time.
            benchmark_results["runtime_with_full_checkpoint"] = time.time() - start_time
            print(
                f"[{benchmark_name}] dataset with full checkpoint finished in {runtime:.2f} seconds"
            )
        return benchmark_results

    benchmark.run_fn(benchmark_name, run)


def clean_up_output_files(
    checkpoint_config: Optional[CheckpointConfig],
    data_output_path: str,
):
    print("Cleaning up output files")
    output_paths = [data_output_path]
    if checkpoint_config is not None:
        assert checkpoint_config.checkpoint_path is not None
        output_paths.append(checkpoint_config.checkpoint_path)
    for checkpoint_path in output_paths:
        print(f"Cleaning up {checkpoint_path}")
        if checkpoint_path.startswith("s3://"):
            import boto3

            s3 = boto3.client("s3")
            bucket, key = checkpoint_path[len("s3://") :].split("/", 1)
            s3.delete_object(Bucket=bucket, Key=key)
        else:
            if not os.path.exists(checkpoint_path):
                continue
            import shutil

            shutil.rmtree(checkpoint_path)


# This benchmark is triggered by `run_checkpoint_benchmark.py` in CI.
# This is only used for manual run.
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--checkpoint_backend", type=str, default="None")
    _ = parser.add_argument("--input_data_path", type=str)
    _ = parser.add_argument("--data_output_path", type=str)
    _ = parser.add_argument("--checkpoint_output_path", type=str)
    _ = parser.add_argument("--inference_concurrency", type=int)
    _ = parser.add_argument("--inference_batch_size", type=int)
    _ = parser.add_argument("--inference_sleep_s", type=float)
    _ = parser.add_argument("--transform_sleep_s", type=float, default=0.001)
    _ = parser.add_argument("--num_output_files", type=int, default=50)
    _ = parser.add_argument("--generated_id_column", type=str, default=None)
    args = parser.parse_args()

    checkpoint_config = _parse_checkpoint_config(args)
    try:
        benchmark = Benchmark()
        run_checkpoints_benchmark(
            benchmark,
            checkpoint_config,
            args.input_data_path,
            args.transform_sleep_s,
            args.inference_sleep_s,
            args.inference_batch_size,
            args.inference_concurrency,
            args.data_output_path,
            args.num_output_files,
        )
        benchmark.write_result()
    finally:
        clean_up_output_files(
            checkpoint_config,
            args.data_output_path,
        )
