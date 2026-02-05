import pytest

import ray
from ray.anyscale.data._internal.logical.rules.combine_downloads import CombineDownloads
from ray.data._internal.logical.interfaces import LogicalPlan
from ray.data._internal.logical.operators.input_data_operator import InputData
from ray.data._internal.logical.operators.one_to_one_operator import Download
from ray.data.context import DataContext
from ray.data.expressions import download
from ray.data.tests.conftest import *  # noqa


def test_combine_sequential_downloads():
    ctx = DataContext.get_current()

    # Create a chain: InputData -> Download -> Download -> Download
    source = InputData([])
    download1 = Download(source, ["uri1"], ["bytes1"])
    download2 = Download(download1, ["uri2"], ["bytes2"])
    download3 = Download(download2, ["uri3"], ["bytes3"])

    plan = LogicalPlan(download3, ctx)
    rule = CombineDownloads()
    optimized_plan = rule.apply(plan)

    # Should have only 1 Download operator with all 3 columns
    download_ops = [
        op for op in optimized_plan.dag.post_order_iter() if isinstance(op, Download)
    ]

    assert len(download_ops) == 1
    combined = download_ops[0]
    assert combined.uri_column_names == ["uri1", "uri2", "uri3"]
    assert combined.output_bytes_column_names == ["bytes1", "bytes2", "bytes3"]
    assert combined.ray_remote_args == {}


def test_single_download_unchanged():
    """A single download should not be modified."""
    ctx = DataContext.get_current()

    source = InputData([])
    download = Download(source, ["uri1", "uri2"], ["bytes1", "bytes2"])

    plan = LogicalPlan(download, ctx)
    rule = CombineDownloads()
    optimized_plan = rule.apply(plan)

    # Plan should be unchanged
    assert optimized_plan is plan

    download_ops = [
        op for op in optimized_plan.dag.post_order_iter() if isinstance(op, Download)
    ]

    assert len(download_ops) == 1
    assert download_ops[0] is download


def test_different_ray_remote_args_not_combined():
    ctx = DataContext.get_current()

    source = InputData([])
    download1 = Download(source, ["uri1"], ["bytes1"], {"num_cpus": 1})
    download2 = Download(download1, ["uri2"], ["bytes2"], {"num_cpus": 2})

    plan = LogicalPlan(download2, ctx)
    rule = CombineDownloads()
    optimized_plan = rule.apply(plan)

    download_ops = [
        op for op in optimized_plan.dag.post_order_iter() if isinstance(op, Download)
    ]

    # Should have 2 separate Download operators due to different resources
    assert len(download_ops) == 2
    assert download_ops[0].ray_remote_args == {"num_cpus": 1}
    assert download_ops[1].ray_remote_args == {"num_cpus": 2}


def test_same_ray_remote_args_are_combined():
    ctx = DataContext.get_current()

    source = InputData([])
    download1 = Download(source, ["uri1"], ["bytes1"], {"num_cpus": 2})
    download2 = Download(download1, ["uri2"], ["bytes2"], {"num_cpus": 2})
    download3 = Download(download2, ["uri3"], ["bytes3"], {"num_cpus": 2})

    plan = LogicalPlan(download3, ctx)
    rule = CombineDownloads()
    optimized_plan = rule.apply(plan)

    download_ops = [
        op for op in optimized_plan.dag.post_order_iter() if isinstance(op, Download)
    ]

    assert len(download_ops) == 1
    combined = download_ops[0]
    assert combined.uri_column_names == ["uri1", "uri2", "uri3"]
    assert combined.output_bytes_column_names == ["bytes1", "bytes2", "bytes3"]
    assert combined.ray_remote_args == {"num_cpus": 2}


def test_combine_downloads_correctness(ray_start_10_cpus_shared, tmp_path):
    path1 = tmp_path / "file1.txt"
    path1.write_bytes("spam".encode())
    path2 = tmp_path / "file2.txt"
    path2.write_bytes("ham".encode())

    ds = (
        ray.data.from_items([{"uri1": str(path1), "uri2": str(path2)}])
        .with_column("bytes1", download("uri1"))
        .with_column("bytes2", download("uri2"))
    )
    results = ds.take_all()

    assert len(results) == 1
    result = results[0]
    assert result["bytes1"] == b"spam"
    assert result["bytes2"] == b"ham"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
