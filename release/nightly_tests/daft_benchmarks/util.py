import ray


def warmup_ray():
    """Launch some tasks to warmup Ray.

    On a fresh Ray cluster, it can take a minute or longer to schedule the first task.
    To ensure benchmarks compare data processing speed and not cluster startup overhead,
    this function launches a several tasks as warmup.
    """

    @ray.remote
    def warmup():
        pass

    ray.get([warmup.remote() for _ in range(64)])
