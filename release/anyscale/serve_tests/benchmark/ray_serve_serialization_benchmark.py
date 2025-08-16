import time
import pickle
import statistics
import os
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import argparse

from pydantic import BaseModel, Field

from ray.anyscale.serve._private.serialization import (
    RPCSerializer,
    SerializationMethod,
)


@dataclass
class BenchmarkResult:
    """Store results for a single benchmark run."""

    serialization_method: str
    workload_type: str
    data_size_bytes: int
    avg_serialize_time_ms: float
    avg_deserialize_time_ms: float
    p50_serialize_time_ms: float
    p95_serialize_time_ms: float
    p99_serialize_time_ms: float
    p50_deserialize_time_ms: float
    p95_deserialize_time_ms: float
    p99_deserialize_time_ms: float
    throughput_rps: float
    success_rate: float
    error_message: Optional[str] = None
    # Raw timing data for box plots
    serialize_times_ms: List[float] = field(default_factory=list)
    deserialize_times_ms: List[float] = field(default_factory=list)


class WorkloadDataGenerator:
    """Generate different types of workloads for benchmarking."""

    @staticmethod
    def pydantic_model_data() -> BaseModel:
        """Pydantic model data - common in modern Python APIs and data validation."""

        class Address(BaseModel):
            street: str
            city: str
            state: str
            zip_code: str
            country: str = "USA"

        class Preferences(BaseModel):
            theme: str = "light"
            notifications: bool = True
            language: str = "en"
            timezone: str = "UTC"

        class User(BaseModel):
            id: int
            username: str
            email: str
            first_name: str
            last_name: str
            age: Optional[int] = None
            is_active: bool = True
            created_at: datetime
            updated_at: datetime
            address: Optional[Address] = None
            preferences: Preferences
            tags: List[str] = Field(default_factory=list)
            metadata: Dict[str, Any] = Field(default_factory=dict)

        # Create a realistic user object
        return User(
            id=12345,
            username="alice_johnson",
            email="alice.johnson@example.com",
            first_name="Alice",
            last_name="Johnson",
            age=28,
            is_active=True,
            created_at=datetime(2023, 1, 15, 10, 30, 0),
            updated_at=datetime(2024, 1, 15, 14, 45, 30),
            address=Address(
                street="123 Main St, Apt 4B",
                city="San Francisco",
                state="CA",
                zip_code="94105",
                country="USA",
            ),
            preferences=Preferences(
                theme="dark",
                notifications=True,
                language="en",
                timezone="America/Los_Angeles",
            ),
            tags=["premium", "beta_tester", "early_adopter"],
            metadata={
                "source": "web_registration",
                "campaign": "spring_2023",
                "referrer": "google_ads",
                "device": "desktop",
                "browser": "chrome",
            },
        )

    @staticmethod
    def small_json_data() -> Dict[str, Any]:
        """Small JSON data (~100 bytes)."""
        return {
            "user_id": 12345,
            "name": "John Doe",
            "email": "john.doe@example.com",
            "status": "active",
            "timestamp": "2024-01-01T00:00:00Z",
        }

    @staticmethod
    def large_json_data() -> Dict[str, Any]:
        """Large JSON data (~1MB)."""
        base_data = {
            "metadata": {
                "version": "1.0",
                "timestamp": "2024-01-01T00:00:00Z",
                "source": "benchmark",
            },
            "records": [],
        }

        # Generate ~1MB of JSON data
        for i in range(1000):
            base_data["records"].append(
                {
                    "id": i,
                    "name": f"Record {i}",
                    "value": i * 1.5,
                    "tags": [f"tag_{j}" for j in range(5)],
                    "metadata": {
                        "created_at": "2024-01-01T00:00:00Z",
                        "updated_at": "2024-01-01T00:00:00Z",
                        "version": 1,
                    },
                }
            )

        return base_data

    @staticmethod
    def complex_python_object() -> Any:
        """Complex Python object with nested structures."""

        class CustomClass:
            def __init__(self, value):
                self.value = value
                self.nested = {"key": value, "list": [1, 2, 3]}

        return {
            "custom_objects": [CustomClass(i) for i in range(100)],
            "nested_dict": {
                "level1": {"level2": {"level3": {"data": list(range(1000))}}}
            },
            "mixed_types": [1, "string", 3.14, True, None, [1, 2, 3]],
            "numpy_array": np.random.rand(1000).tolist(),
            "tuple_data": tuple(range(100)),
        }

    @staticmethod
    def binary_data() -> bytes:
        """Binary data (~100KB)."""
        return os.urandom(100 * 1024)

    @staticmethod
    def large_string() -> str:
        """Large string data (~1MB) - common in text processing, documents, logs."""
        # Generate a large string that resembles real text data
        base_text = """
        This is a sample text document that contains multiple paragraphs of content.
        It simulates real-world text processing scenarios such as document analysis,
        natural language processing, chat message handling, and log file processing.

        The text includes various formatting elements, punctuation, and realistic
        sentence structures that you might find in actual applications dealing with
        textual data. This helps ensure the serialization benchmark reflects
        real-world performance characteristics.

        Common use cases for large string serialization include:
        - Document processing and analysis
        - Chat message storage and retrieval
        - Log file aggregation and processing
        - Natural language processing pipelines
        - Web scraping and content extraction
        - Email and message handling systems
        - Configuration file management
        - Template rendering and processing
        """

        # Repeat and modify the base text to create ~1MB of content
        paragraphs = []
        for i in range(200):  # This will create approximately 1MB of text
            modified_text = base_text.replace("sample text", f"sample text #{i}")
            modified_text = modified_text.replace("document", f"document-{i:04d}")
            paragraphs.append(modified_text)

        return "\n".join(paragraphs)

    @staticmethod
    def small_string() -> str:
        """Small string data (~1KB) - common in API responses, messages, logs."""
        return """
        This is a small string workload that represents typical text data
        found in API responses, log messages, chat messages, and configuration data.
        It's designed to test serialization performance for common text-based
        communication scenarios in web services and microservices architectures.

        Common use cases include:
        - API response messages
        - Log entries and system messages
        - Chat messages and notifications
        - Configuration strings
        - Error messages and status updates
        - Short document snippets
        - Metadata and descriptions

        This workload helps benchmark serialization performance for the most
        common text data size range in production systems.
        """

    @staticmethod
    def numpy_arrays() -> np.ndarray:
        """NumPy array workload - common in ML/data science."""
        # Medium-sized float array - good balance of size and performance testing
        return np.random.rand(1000, 100).astype(np.float32)

    @staticmethod
    def pandas_dataframes() -> pd.DataFrame:
        """Pandas DataFrame workload - common in data processing."""
        # Realistic DataFrame with mixed data types
        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=5000, freq="1min"),
                "sensor_id": np.random.randint(1, 100, 5000),
                "temperature": np.random.normal(20, 5, 5000),
                "humidity": np.random.normal(50, 10, 5000),
                "status": np.random.choice(["OK", "WARNING", "ERROR"], 5000),
            }
        )

    @staticmethod
    def tensor_like_data() -> np.ndarray:
        """Tensor-like data structures common in ML."""
        # Image batch tensor - common in computer vision
        return np.random.rand(16, 224, 224, 3).astype(np.float32)


class SerializationBenchmark:
    """Main benchmark class."""

    def __init__(self, num_iterations: int = 1000):
        self.num_iterations = num_iterations
        self.results: List[BenchmarkResult] = []
        self.workload_generators = {
            "small_json": WorkloadDataGenerator.small_json_data,
            "large_json": WorkloadDataGenerator.large_json_data,
            "complex_python": WorkloadDataGenerator.complex_python_object,
            "binary_data": WorkloadDataGenerator.binary_data,
            "small_string": WorkloadDataGenerator.small_string,
            "large_string": WorkloadDataGenerator.large_string,
            "pydantic_model": WorkloadDataGenerator.pydantic_model_data,
            "numpy_arrays": WorkloadDataGenerator.numpy_arrays,
            "pandas_dataframes": WorkloadDataGenerator.pandas_dataframes,
            "tensor_like": WorkloadDataGenerator.tensor_like_data,
        }

        self.serialization_methods = [
            SerializationMethod.CLOUDPICKLE,
            SerializationMethod.PICKLE,
            SerializationMethod.MSGPACK,
            SerializationMethod.ORJSON,
            SerializationMethod.NOOP,
        ]

    def _get_data_size(self, data: Any) -> int:
        """Estimate data size in bytes."""
        try:
            # Try to serialize with pickle to get size estimate
            return len(pickle.dumps(data))
        except Exception:
            return 0

    def _get_serializer(self, method: str):
        """Get the appropriate serializer for the method."""
        return RPCSerializer.get_cached_serializer(method, method)

    def _run_single_benchmark(
        self, serialization_method: str, workload_type: str, data_generator: Callable
    ) -> BenchmarkResult:
        """Run a single benchmark test."""
        print(f"  Testing {serialization_method} with {workload_type}...")

        # Generate test data
        try:
            test_data = data_generator()
            data_size = self._get_data_size(test_data)
        except Exception as e:
            return BenchmarkResult(
                serialization_method=serialization_method,
                workload_type=workload_type,
                data_size_bytes=0,
                avg_serialize_time_ms=0,
                avg_deserialize_time_ms=0,
                p50_serialize_time_ms=0,
                p95_serialize_time_ms=0,
                p99_serialize_time_ms=0,
                p50_deserialize_time_ms=0,
                p95_deserialize_time_ms=0,
                p99_deserialize_time_ms=0,
                throughput_rps=0,
                success_rate=0,
                error_message=f"Data generation failed: {str(e)}",
            )

        # Create serializer
        try:
            serializer = self._get_serializer(serialization_method)
        except Exception as e:
            return BenchmarkResult(
                serialization_method=serialization_method,
                workload_type=workload_type,
                data_size_bytes=data_size,
                avg_serialize_time_ms=0,
                avg_deserialize_time_ms=0,
                p50_serialize_time_ms=0,
                p95_serialize_time_ms=0,
                p99_serialize_time_ms=0,
                p50_deserialize_time_ms=0,
                p95_deserialize_time_ms=0,
                p99_deserialize_time_ms=0,
                throughput_rps=0,
                success_rate=0,
                error_message=f"Serializer creation failed: {str(e)}",
            )

        # Run benchmark
        serialize_times = []
        deserialize_times = []
        successful_runs = 0
        errors = []

        for i in range(self.num_iterations):
            try:
                # Serialize
                start_time = time.perf_counter()
                serialized_data = serializer.dumps_request(test_data)
                serialize_time = (time.perf_counter() - start_time) * 1000
                serialize_times.append(serialize_time)

                # Deserialize
                start_time = time.perf_counter()
                _ = serializer.loads_request(serialized_data)
                deserialize_time = (time.perf_counter() - start_time) * 1000
                deserialize_times.append(deserialize_time)

                successful_runs += 1

            except Exception as e:
                errors.append(str(e))
                continue

        # Calculate metrics
        if not serialize_times or not deserialize_times:
            return BenchmarkResult(
                serialization_method=serialization_method,
                workload_type=workload_type,
                data_size_bytes=data_size,
                avg_serialize_time_ms=0,
                avg_deserialize_time_ms=0,
                p50_serialize_time_ms=0,
                p95_serialize_time_ms=0,
                p99_serialize_time_ms=0,
                p50_deserialize_time_ms=0,
                p95_deserialize_time_ms=0,
                p99_deserialize_time_ms=0,
                throughput_rps=0,
                success_rate=0,
                error_message=f"All iterations failed. Sample errors: {errors[:3]}",
            )

        avg_serialize_time = statistics.mean(serialize_times)
        avg_deserialize_time = statistics.mean(deserialize_times)
        total_time = sum(serialize_times) + sum(deserialize_times)
        throughput = (successful_runs * 1000) / total_time if total_time > 0 else 0

        return BenchmarkResult(
            serialization_method=serialization_method,
            workload_type=workload_type,
            data_size_bytes=data_size,
            avg_serialize_time_ms=avg_serialize_time,
            avg_deserialize_time_ms=avg_deserialize_time,
            p50_serialize_time_ms=statistics.median(serialize_times),
            p95_serialize_time_ms=statistics.quantiles(serialize_times, n=20)[18]
            if len(serialize_times) > 20
            else max(serialize_times),
            p99_serialize_time_ms=statistics.quantiles(serialize_times, n=100)[98]
            if len(serialize_times) > 100
            else max(serialize_times),
            p50_deserialize_time_ms=statistics.median(deserialize_times),
            p95_deserialize_time_ms=statistics.quantiles(deserialize_times, n=20)[18]
            if len(deserialize_times) > 20
            else max(deserialize_times),
            p99_deserialize_time_ms=statistics.quantiles(deserialize_times, n=100)[98]
            if len(deserialize_times) > 100
            else max(deserialize_times),
            throughput_rps=throughput,
            success_rate=successful_runs / self.num_iterations,
            error_message=f"Errors: {len(errors)}" if errors else None,
            # Store raw timing data for box plots
            serialize_times_ms=serialize_times.copy(),
            deserialize_times_ms=deserialize_times.copy(),
        )

    def run_all_benchmarks(self) -> List[BenchmarkResult]:
        """Run all benchmark combinations."""
        print("Starting Ray Serve Serialization Benchmark...")
        print(f"Running {self.num_iterations} iterations per test...")
        print("=" * 60)

        for workload_type, data_generator in self.workload_generators.items():
            print(f"\nTesting workload: {workload_type}")

            for method in self.serialization_methods:
                result = self._run_single_benchmark(
                    method, workload_type, data_generator
                )
                self.results.append(result)

        print("\nBenchmark completed!")
        return self.results


class ReportGenerator:
    """Generate comprehensive performance report."""

    def __init__(self, results: List[BenchmarkResult]):
        self.results = results
        self.df = pd.DataFrame(
            [
                {
                    "method": r.serialization_method,
                    "workload": r.workload_type,
                    "data_size_kb": r.data_size_bytes / 1024,
                    "avg_serialize_ms": r.avg_serialize_time_ms,
                    "avg_deserialize_ms": r.avg_deserialize_time_ms,
                    "p95_serialize_ms": r.p95_serialize_time_ms,
                    "p95_deserialize_ms": r.p95_deserialize_time_ms,
                    "throughput_rps": r.throughput_rps,
                    "success_rate": r.success_rate,
                    "total_time_ms": r.avg_serialize_time_ms
                    + r.avg_deserialize_time_ms,
                    "error_msg": r.error_message,
                }
                for r in results
            ]
        )

    def _format_latency(self, latency_ms: float) -> str:
        """Format latency with appropriate units (ms or μs)."""
        if latency_ms < 0.1:  # Less than 0.1ms, show in microseconds
            latency_us = latency_ms * 1000
            if latency_us < 0.1:
                return f"{latency_us * 1000:.1f}ns"
            return f"{latency_us:.1f}μs"
        else:
            return f"{latency_ms:.2f}ms"

    def generate_performance_charts(
        self, output_path: str = "/tmp/ray_serve_serialization_benchmark.png"
    ):
        """Generate performance visualization charts with box plots by workload type."""
        plt.style.use("seaborn-v0_8")

        # Get successful results only
        successful_df = self.df[self.df["success_rate"] > 0.8]

        if successful_df.empty:
            print("No successful results to plot")
            return

        workloads = successful_df["workload"].unique()
        n_workloads = len(workloads)

        # Create figure with subplots: one row per workload, 2 columns (latency, throughput)
        fig, axes = plt.subplots(n_workloads, 2, figsize=(12, 5 * n_workloads))

        # Handle case where there's only one workload
        if n_workloads == 1:
            axes = axes.reshape(1, -1)

        colors = ["lightblue", "lightgreen", "lightcoral", "lightyellow", "lightpink"]

        for i, workload in enumerate(workloads):
            workload_data = successful_df[successful_df["workload"] == workload]

            if workload_data.empty:
                continue

            methods = workload_data["method"].tolist()

            # Latency box plot - use raw timing data
            latency_data = []
            serialize_data = []
            deserialize_data = []

            for _, row in workload_data.iterrows():
                # Get the actual BenchmarkResult object to access raw timing data
                result = next(
                    r
                    for r in self.results
                    if r.serialization_method == row["method"]
                    and r.workload_type == row["workload"]
                )

                if result.serialize_times_ms and result.deserialize_times_ms:
                    # Use raw timing data for box plots
                    total_times = [
                        s + d
                        for s, d in zip(
                            result.serialize_times_ms, result.deserialize_times_ms
                        )
                    ]
                    latency_data.append(total_times)
                    serialize_data.append(result.serialize_times_ms)
                    deserialize_data.append(result.deserialize_times_ms)
                else:
                    # Fallback to synthetic data if raw data not available
                    synthetic_latency = [
                        row["total_time_ms"] * 0.7,  # Lower whisker
                        row["total_time_ms"] * 0.85,  # Q1
                        row["total_time_ms"],  # Median (avg)
                        row["total_time_ms"] * 1.15,  # Q3
                        row["total_time_ms"] * 1.3,  # Upper whisker
                    ]
                    latency_data.append(synthetic_latency)

            # Calculate throughput data from raw timing data
            throughput_data = []
            for j, (_, row) in enumerate(workload_data.iterrows()):
                if j < len(latency_data):
                    # Convert latency to throughput (requests per second)
                    latency_seconds = [
                        t / 1000.0 for t in latency_data[j]
                    ]  # Convert ms to seconds
                    throughput_values = [
                        1.0 / t if t > 0 else 0 for t in latency_seconds
                    ]
                    throughput_data.append(throughput_values)
                else:
                    # Fallback to synthetic data
                    base_throughput = row["throughput_rps"]
                    synthetic_throughput = [
                        base_throughput * 0.8,  # Lower whisker
                        base_throughput * 0.9,  # Q1
                        base_throughput,  # Median
                        base_throughput * 1.1,  # Q3
                        base_throughput * 1.2,  # Upper whisker
                    ]
                    throughput_data.append(synthetic_throughput)

            # Create latency box plot
            if latency_data:
                bp1 = axes[i, 0].boxplot(
                    latency_data, labels=methods, patch_artist=True
                )
                axes[i, 0].set_title(
                    f'{workload.replace("_", " ").title()} - Latency Distribution'
                )
                axes[i, 0].set_ylabel("Total Latency (ms)")
                axes[i, 0].tick_params(axis="x", rotation=45)

                # Color the boxes
                for patch, color in zip(bp1["boxes"], colors[: len(bp1["boxes"])]):
                    patch.set_facecolor(color)

            # Create throughput box plot
            if throughput_data:
                bp2 = axes[i, 1].boxplot(
                    throughput_data, labels=methods, patch_artist=True
                )
                axes[i, 1].set_title(
                    f'{workload.replace("_", " ").title()} - Throughput Distribution'
                )
                axes[i, 1].set_ylabel("Requests per Second")
                axes[i, 1].tick_params(axis="x", rotation=45)

                # Color the boxes
                for patch, color in zip(bp2["boxes"], colors[: len(bp2["boxes"])]):
                    patch.set_facecolor(color)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    def generate_recommendations(self) -> Dict[str, str]:
        """Generate performance-based recommendations."""
        recommendations = {}

        # Filter successful results
        successful_results = self.df[self.df["success_rate"] > 0.8]

        if successful_results.empty:
            return {"general": "No reliable results found. Check compatibility issues."}

        # Best overall performance
        best_overall = successful_results.loc[
            successful_results["total_time_ms"].idxmin()
        ]
        recommendations[
            "best_overall"
        ] = f"{best_overall['method']} (avg latency: {self._format_latency(best_overall['total_time_ms'])})"

        # Best for each workload
        for workload in successful_results["workload"].unique():
            workload_data = successful_results[
                successful_results["workload"] == workload
            ]
            if not workload_data.empty:
                best_for_workload = workload_data.loc[
                    workload_data["total_time_ms"].idxmin()
                ]
                recommendations[
                    f"best_for_{workload}"
                ] = f"{best_for_workload['method']} (latency: {self._format_latency(best_for_workload['total_time_ms'])})"

        # High throughput recommendation
        best_throughput = successful_results.loc[
            successful_results["throughput_rps"].idxmax()
        ]
        recommendations[
            "highest_throughput"
        ] = f"{best_throughput['method']} ({best_throughput['throughput_rps']:.0f} RPS)"

        # Most reliable (highest success rate)
        most_reliable = successful_results.loc[
            successful_results["success_rate"].idxmax()
        ]
        recommendations[
            "most_reliable"
        ] = f"{most_reliable['method']} ({most_reliable['success_rate']:.1%} success rate)"

        return recommendations

    def generate_report(self) -> str:
        """Generate comprehensive text report."""
        report = []
        report.append("=" * 80)
        report.append("RAY SERVE SERIALIZATION BENCHMARK REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total test combinations: {len(self.results)}")
        report.append("")

        # Executive Summary
        report.append("EXECUTIVE SUMMARY")
        report.append("-" * 40)
        recommendations = self.generate_recommendations()
        for key, value in recommendations.items():
            report.append(f"{key.replace('_', ' ').title()}: {value}")
        report.append("")

        # Detailed Results
        report.append("DETAILED RESULTS")
        report.append("-" * 40)

        # Group by workload
        for workload in self.df["workload"].unique():
            workload_data = self.df[self.df["workload"] == workload]
            report.append(f"\n{workload.upper()} WORKLOAD:")
            report.append(
                f"{'Method':<12} {'Latency':<12} {'Throughput(RPS)':<15} {'Success Rate':<12} {'Status'}"
            )
            report.append("-" * 65)

            for _, row in workload_data.iterrows():
                status = (
                    "✓"
                    if row["success_rate"] > 0.8
                    else "✗"
                    if row["success_rate"] == 0
                    else "⚠"
                )
                if status == "✗":
                    continue
                latency_str = self._format_latency(row["total_time_ms"])
                report.append(
                    f"{row['method']:<12} {latency_str:<12} {row['throughput_rps']:<15.0f} {row['success_rate']:<12.1%} {status}"
                )

        # Compatibility Matrix
        report.append("\n\nCOMPATIBILITY MATRIX")
        report.append("-" * 40)
        compatibility_pivot = self.df.pivot(
            index="method", columns="workload", values="success_rate"
        )
        report.append(compatibility_pivot.to_string())

        # Performance Summary
        report.append("\n\nPERFORMANCE SUMMARY")
        report.append("-" * 40)
        summary_stats = (
            self.df.groupby("method")
            .agg(
                {
                    "total_time_ms": ["mean", "min", "max"],
                    "throughput_rps": ["mean", "min", "max"],
                    "success_rate": "mean",
                }
            )
            .round(2)
        )
        report.append(summary_stats.to_string())

        return "\n".join(report)


def main():
    """Main benchmark execution."""
    parser = argparse.ArgumentParser(description="Ray Serve Serialization Benchmark")
    parser.add_argument(
        "--output-path",
        type=str,
        default="/tmp/ray_serve_serialization_benchmark",
        help="Output path prefix for generated files (default: /tmp/ray_serve_serialization_benchmark)",
    )
    args = parser.parse_args()

    print("Ray Serve Serialization Benchmark")
    print("=" * 50)
    print(f"Output path: {args.output_path}")
    print()

    # Run benchmarks
    benchmark = SerializationBenchmark(
        num_iterations=1000
    )  # Reduced for faster execution
    results = benchmark.run_all_benchmarks()

    # Generate report
    report_generator = ReportGenerator(results)

    # Generate charts
    try:
        # Update the chart generation to use the output path
        chart_path = f"{args.output_path}.png"
        report_generator.generate_performance_charts(chart_path)
        print(f"\nPerformance charts saved as: {chart_path}")
    except Exception as e:
        print(f"Could not generate charts: {e}")

    # Generate text report
    report_text = report_generator.generate_report()

    # Save report
    report_path = f"{args.output_path}_report.txt"
    with open(report_path, "w") as f:
        f.write(report_text)

    print(f"\nBenchmark report saved as '{report_path}'")
    print("\nReport Preview:")
    print("=" * 50)
    print(report_text)


if __name__ == "__main__":
    main()
