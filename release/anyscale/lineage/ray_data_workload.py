#!/usr/bin/env python3
"""Lineage integration test for Ray Data file format APIs.

This test validates that lineage tracking works correctly for Ray Data APIs.
It tests various file format read/write operations (CSV, Parquet, JSON, Text, Images).
"""

import gc
import os
import time
import uuid
from typing import Any, Dict

import numpy as np
from benchmark import Benchmark

import ray


def is_smoke_test() -> bool:
    """Check if running in smoke test mode for faster CI validation."""
    return os.environ.get("IS_SMOKE_TEST", "0") == "1"


# Local storage for test outputs (use unique path per run)
OUTPUT_BASE = f"/mnt/cluster_storage/lineage-test/ray-data/{uuid.uuid4().hex}"


def test_csv_roundtrip() -> Dict[str, Any]:
    """Test CSV read/write with lineage tracking."""
    gc.collect()
    num_records = 100 if is_smoke_test() else 1000
    ds = ray.data.range(num_records)
    ds = ds.map(lambda x: {"id": x["id"], "value": x["id"] * 2})

    output_path = f"{OUTPUT_BASE}/csv/"
    ds.write_csv(output_path)

    ds_read = ray.data.read_csv(output_path)
    count = ds_read.count()

    return {"records": count, "format": "csv", "path": output_path}


def test_parquet_roundtrip() -> Dict[str, Any]:
    """Test Parquet read/write with lineage tracking."""
    gc.collect()
    num_records = 100 if is_smoke_test() else 1000
    ds = ray.data.range(num_records)
    ds = ds.map(lambda x: {"id": x["id"], "value": str(x["id"])})

    output_path = f"{OUTPUT_BASE}/parquet/"
    ds.write_parquet(output_path)

    ds_read = ray.data.read_parquet(output_path)
    count = ds_read.count()

    return {"records": count, "format": "parquet", "path": output_path}


def test_json_roundtrip() -> Dict[str, Any]:
    """Test JSON read/write with lineage tracking."""
    gc.collect()
    num_records = 10 if is_smoke_test() else 100
    ds = ray.data.range(num_records)
    ds = ds.map(lambda x: {"id": x["id"], "name": f"item_{x['id']}"})

    output_path = f"{OUTPUT_BASE}/json/"
    ds.write_json(output_path)

    ds_read = ray.data.read_json(output_path)
    count = ds_read.count()

    return {"records": count, "format": "json", "path": output_path}


def test_text_roundtrip() -> Dict[str, Any]:
    """Test Text read with lineage tracking.

    Note: Ray Data doesn't have a native write_text() API, so we write using
    write_csv() with include_header=False (which produces plain text lines) and
    read back using read_text(). The write will emit CSV format lineage,
    while the read will emit Text format lineage.
    """
    gc.collect()
    import pyarrow.csv as pa_csv

    # Create text data
    num_records = 10 if is_smoke_test() else 100
    ds = ray.data.range(num_records)
    ds = ds.map(lambda x: {"text": f"Line {x['id']}: This is sample text content."})

    output_path = f"{OUTPUT_BASE}/text/"
    # Write as CSV without header (produces plain text lines)
    # Note: This will emit CSV format in lineage for the write operation
    # Use arrow_csv_args_fn to pass WriteOptions with include_header=False
    ds.write_csv(
        output_path,
        arrow_csv_args_fn=lambda: {
            "write_options": pa_csv.WriteOptions(include_header=False)
        },
    )

    # Read back as text - this will emit Text format in lineage
    ds_read = ray.data.read_text(output_path)
    count = ds_read.count()

    return {"records": count, "format": "text", "path": output_path}


def test_images_roundtrip() -> Dict[str, Any]:
    """Test Images read/write with lineage tracking.

    Creates synthetic images, writes them to local storage, and reads them back.
    """
    gc.collect()
    from PIL import Image

    num_images = 3 if is_smoke_test() else 10

    # Generate synthetic image data (simple colored images)
    def generate_synthetic_images(count: int = 10):
        images = []
        for i in range(count):
            # Create a simple colored image (64x64 RGB)
            color = [(i * 25) % 256, (i * 50) % 256, (i * 75) % 256]
            img_array = np.full((64, 64, 3), color, dtype=np.uint8)
            images.append({"image": img_array, "label": f"image_{i}"})
        return images

    # Create dataset from synthetic images
    image_data = generate_synthetic_images(num_images)

    # Write images to local filesystem as PNG files
    image_files_path = f"{OUTPUT_BASE}/image_files/"
    os.makedirs(image_files_path, exist_ok=True)

    for i, item in enumerate(image_data):
        img = Image.fromarray(item["image"])
        img.save(os.path.join(image_files_path, f"img_{i:04d}.png"))

    # Read images back using Ray Data
    ds_read = ray.data.read_images(image_files_path)
    count = ds_read.count()

    return {
        "records": count,
        "format": "images",
        "path": image_files_path,
    }


def main():
    """Run lineage integration tests."""
    ray.init()

    try:
        # Create output directory
        os.makedirs(OUTPUT_BASE, exist_ok=True)

        benchmark = Benchmark()

        print(f"Test output base path: {OUTPUT_BASE}")
        print(f"Smoke test mode: {is_smoke_test()}\n")

        # Run success test cases for each file format
        print("Running CSV roundtrip test...")
        benchmark.run_fn("csv_roundtrip", test_csv_roundtrip)

        print("Running Parquet roundtrip test...")
        benchmark.run_fn("parquet_roundtrip", test_parquet_roundtrip)

        print("Running JSON roundtrip test...")
        benchmark.run_fn("json_roundtrip", test_json_roundtrip)

        print("Running Text roundtrip test...")
        benchmark.run_fn("text_roundtrip", test_text_roundtrip)

        print("Running Images roundtrip test...")
        benchmark.run_fn("images_roundtrip", test_images_roundtrip)

        # Allow time for async event emission
        print("Waiting for async event emission...")
        time.sleep(2)

        benchmark.write_result()

    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
