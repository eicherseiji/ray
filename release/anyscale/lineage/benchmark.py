import json
import os
import time


class Benchmark:
    """Benchmark runner for lineage release tests."""

    def __init__(self):
        self.result = {}

    def run_fn(self, name, fn, *args, **kwargs):
        """Run a function and record its runtime."""
        print(f"Running case: {name}")
        start_time = time.perf_counter()
        fn_output = fn(*args, **kwargs)
        duration = time.perf_counter() - start_time

        self.result[name] = {"time": duration}
        if isinstance(fn_output, dict):
            self.result[name].update(fn_output)

        print(f"Result of case {name}: {self.result[name]}")

    def write_result(self):
        """Write results to JSON file."""
        test_output_json = os.environ.get("TEST_OUTPUT_JSON", "./result.json")
        with open(test_output_json, "w") as f:
            json.dump(self.result, f)
        print(f"Finished benchmark, metrics exported to '{test_output_json}':")
        print(json.dumps(self.result, indent=4))
