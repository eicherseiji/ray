load("@rules_python//python:defs.bzl", "py_test")

py_test(
    name = "test_zmq_channel",
    size = "small",
    srcs = ["test_zmq_channel.py"],
    tags = [
        "exclusive",
        "no_main",
        "no_windows",
        "small_size_python_tests",
        "team:core",
    ],
    deps = [
        "//:ray_lib",
        "//python/ray/tests:conftest",
    ],
)
