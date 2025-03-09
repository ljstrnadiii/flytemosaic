import dask.array as da
from distributed import Client
from flytekit import Resources, task
from flytekitplugins.dask import Dask, Scheduler, WorkerGroup


@task(
    cache=True,
    cache_version="1.0",
    interruptible=True,
    retries=2,
    requests=Resources(cpu="2", mem="4Gi"),
    task_config=Dask(
        scheduler=Scheduler(
            limits=Resources(cpu="4", mem="16Gi"),
            requests=Resources(cpu="4", mem="16Gi"),
        ),
        workers=WorkerGroup(
            number_of_workers=16,
            limits=Resources(cpu="2", mem="8Gi"),
            requests=Resources(cpu="2", mem="8Gi"),
        ),
    ),
)
def dask_cluster_example() -> float:
    with Client() as client:
        client.wait_for_workers(8)
        # take the mean of a 4Tb array using the dask cluster
        mean = da.random.uniform(size=(1024, 1024, 1024, 1024)).astype("float32").mean()
        return float(mean.compute())
