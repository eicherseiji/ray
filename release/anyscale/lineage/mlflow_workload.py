#!/usr/bin/env python3
"""Lineage integration test for MLflow APIs.

This test validates that lineage tracking works correctly for MLflow APIs.
It tests model logging, loading, artifact operations, and model registration.
"""

import gc
import os
import tempfile
import time
import uuid
from typing import Any, Dict

import numpy as np
from benchmark import Benchmark

import ray


def is_smoke_test() -> bool:
    """Check if running in smoke test mode for faster CI validation."""
    return os.environ.get("IS_SMOKE_TEST", "0") == "1"


# Local storage for artifacts (use unique path per run)
# Note: file:// prefix is required to activate the Anyscale MLflow artifact repository plugin
ARTIFACT_BASE = f"file:///mnt/cluster_storage/lineage-test/mlflow/{uuid.uuid4().hex}"

# File-based MLflow tracking (no server needed)
MLFLOW_TRACKING_URI = "file:///mnt/cluster_storage/lineage-test/mlruns"


def get_or_create_experiment(experiment_name: str) -> str:
    """Get existing experiment or create new one with ARTIFACT_BASE as artifact location."""
    import mlflow

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment:
        return experiment.experiment_id
    return mlflow.create_experiment(experiment_name, artifact_location=ARTIFACT_BASE)


def test_log_model() -> Dict[str, Any]:
    """Test mlflow.sklearn.log_model with lineage tracking."""
    gc.collect()
    import mlflow
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Create and train simple sklearn model (smaller for smoke test)
    n_samples = 50 if is_smoke_test() else 100
    X, y = make_classification(n_samples=n_samples, n_features=4, random_state=42)
    model = LogisticRegression(random_state=42)
    model.fit(X, y)

    # Log model to MLflow
    experiment_name = f"lineage-test-{uuid.uuid4().hex[:8]}"
    experiment_id = get_or_create_experiment(experiment_name)
    mlflow.set_experiment(experiment_id=experiment_id)

    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(model, "sklearn-model")
        run_id = run.info.run_id
        artifact_uri = run.info.artifact_uri

    print(f"log_model: run_id={run_id}, artifact_uri={artifact_uri}")

    return {
        "operation": "log_model",
        "run_id": run_id,
        "artifact_uri": artifact_uri,
        "model_path": "sklearn-model",
        "experiment_name": experiment_name,
    }


def test_load_model_from_uri(run_id: str) -> Dict[str, Any]:
    """Test mlflow.pyfunc.load_model from artifact URI."""
    gc.collect()
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    model_uri = f"runs:/{run_id}/sklearn-model"
    print(f"load_model_from_uri: loading from {model_uri}")

    loaded_model = mlflow.pyfunc.load_model(model_uri)

    # Verify model works
    X_test = np.array([[0.1, 0.2, 0.3, 0.4]])
    prediction = loaded_model.predict(X_test)

    print(f"load_model_from_uri: prediction shape={prediction.shape}")

    return {
        "operation": "load_model_uri",
        "model_uri": model_uri,
        "prediction_shape": list(prediction.shape),
    }


def test_log_artifact() -> Dict[str, Any]:
    """Test mlflow.log_artifact with lineage tracking."""
    gc.collect()
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Create temp file with explicit name to log
    artifact_content = "test artifact content for lineage validation"
    temp_dir = tempfile.mkdtemp()
    artifact_file = os.path.join(temp_dir, "test_artifact.txt")
    with open(artifact_file, "w") as f:
        f.write(artifact_content)

    experiment_name = f"lineage-artifact-test-{uuid.uuid4().hex[:8]}"
    experiment_id = get_or_create_experiment(experiment_name)
    mlflow.set_experiment(experiment_id=experiment_id)

    with mlflow.start_run() as run:
        mlflow.log_artifact(artifact_file)
        run_id = run.info.run_id
        artifact_uri = run.info.artifact_uri

    print(f"log_artifact: run_id={run_id}, artifact_uri={artifact_uri}")

    return {
        "operation": "log_artifact",
        "run_id": run_id,
        "artifact_uri": artifact_uri,
        "artifact_path": "test_artifact.txt",
    }


def test_download_artifacts(run_id: str) -> Dict[str, Any]:
    """Test mlflow.artifacts.download_artifacts with lineage tracking."""
    gc.collect()
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    print(f"download_artifacts: downloading from run_id={run_id}")

    download_path = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path="test_artifact.txt",
    )

    print(f"download_artifacts: downloaded to {download_path}")

    return {
        "operation": "download_artifacts",
        "run_id": run_id,
        "download_path": download_path,
    }


def test_register_model(run_id: str) -> Dict[str, Any]:
    """Test mlflow.register_model with lineage tracking."""
    gc.collect()
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    model_uri = f"runs:/{run_id}/sklearn-model"
    model_name = f"test-registered-model-{uuid.uuid4().hex[:8]}"

    print(f"register_model: registering {model_uri} as {model_name}")

    result = mlflow.register_model(model_uri, model_name)

    print(f"register_model: version={result.version}, source={result.source}")

    return {
        "operation": "register_model",
        "model_name": model_name,
        "model_version": result.version,
        "source": result.source,
    }


def test_load_model_from_registry(model_name: str, version: str) -> Dict[str, Any]:
    """Test mlflow.pyfunc.load_model from model registry."""
    gc.collect()
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    model_uri = f"models:/{model_name}/{version}"
    print(f"load_model_from_registry: loading from {model_uri}")

    loaded_model = mlflow.pyfunc.load_model(model_uri)

    # Verify model works
    X_test = np.array([[0.1, 0.2, 0.3, 0.4]])
    prediction = loaded_model.predict(X_test)

    print(f"load_model_from_registry: prediction shape={prediction.shape}")

    return {
        "operation": "load_model_registry",
        "model_uri": model_uri,
        "prediction_shape": list(prediction.shape),
    }


def main():
    """Run MLflow lineage integration tests."""
    # IMPORTANT: ray.init() must be called BEFORE importing MLflow.
    # For lineage tracking to work with the MLflow plugin, we rely on Ray's
    # session logging configuration for event and application logging.
    # MLflow is imported inside test functions to ensure this ordering.
    # TODO(GA): Move log files out of Ray session logging directory so this
    # requirement is no longer needed.
    ray.init()

    try:
        # Create artifact and tracking directories
        os.makedirs(ARTIFACT_BASE, exist_ok=True)
        os.makedirs("/mnt/cluster_storage/lineage-test/mlruns", exist_ok=True)

        benchmark = Benchmark()

        print(f"\nTest artifact base path: {ARTIFACT_BASE}")
        print(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")
        print(f"Smoke test mode: {is_smoke_test()}\n")

        # Test log_model
        print("Running log_model test...")
        benchmark.run_fn("log_model", test_log_model)
        run_id = benchmark.result["log_model"]["run_id"]

        # Test load_model from URI
        print("\nRunning load_model (from URI) test...")
        benchmark.run_fn(
            "load_model_uri", lambda rid=run_id: test_load_model_from_uri(rid)
        )

        # Test log_artifact
        print("\nRunning log_artifact test...")
        benchmark.run_fn("log_artifact", test_log_artifact)
        artifact_run_id = benchmark.result["log_artifact"]["run_id"]

        # Test download_artifacts
        print("\nRunning download_artifacts test...")
        benchmark.run_fn(
            "download_artifacts",
            lambda arid=artifact_run_id: test_download_artifacts(arid),
        )

        # Test register_model (skip in smoke test for faster execution)
        if not is_smoke_test():
            print("\nRunning register_model test...")
            benchmark.run_fn(
                "register_model", lambda rid=run_id: test_register_model(rid)
            )
            model_name = benchmark.result["register_model"]["model_name"]
            model_version = benchmark.result["register_model"]["model_version"]

            # Test load_model from registry
            print("\nRunning load_model (from registry) test...")
            benchmark.run_fn(
                "load_model_registry",
                lambda mn=model_name, mv=model_version: test_load_model_from_registry(
                    mn, mv
                ),
            )
        else:
            print("\nSkipping register_model and load_model_registry (smoke test)")

        # Allow time for async event emission
        print("\nWaiting for async event emission...")
        time.sleep(2)

        benchmark.write_result()

        print("\nAll MLflow tests PASSED")

    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
