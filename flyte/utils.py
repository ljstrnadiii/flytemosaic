import base64
import io
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import flytekit
import fsspec
import xarray as xr
from distributed import Client
from flytekit.core.context_manager import FlyteContextManager
from flytekitplugins.dask.task import Dask, Scheduler, WorkerGroup
from matplotlib import pyplot as plt
from union import Resources, task, workflow
from yarl import URL

if TYPE_CHECKING:
    import s3fs

PROJECT_DIR = "flytemosaic"


def get_default_bucket() -> URL:
    """
    Get a deterministic default bucket for the project.

    Returns
    -------
    URL
        The default bucket for the project. It will be deterministic remotely,
        but random locally.
    """
    ctx = FlyteContextManager.current_context()
    parsed_url = urlparse(ctx.file_access.raw_output_prefix)
    if parsed_url.scheme == "":
        # local case will be random
        # TODO: perhaps make it deterministic somehow so we can more easily rerun
        # locally?
        return URL(parsed_url.path)
    else:
        return URL(f"{parsed_url.scheme}://{parsed_url.netloc}") / PROJECT_DIR


@task
def default_bucket_usage_task() -> dict[str, list[int]]:
    bucket = get_default_bucket()
    fs: s3fs.S3FileSystem = fsspec.get_filesystem_class(bucket.scheme)()
    bucket_name = str(bucket)
    top_level = fs.du(bucket_name, withdirs=True, total=False, maxdepth=1)
    dir_sizes = {}
    for subdir in top_level:
        sizes = fs.du(subdir, withdirs=True, total=False)
        dir_sizes[subdir] = [len(sizes), sum(sizes.values())]
    return dir_sizes


@task
def rm_project_bucket_task(password: str) -> None:
    if password != "flytemosaic":
        raise ValueError("Password is incorrect. Aborting.")
    bucket = get_default_bucket()
    fs: s3fs.S3FileSystem = fsspec.get_filesystem_class(bucket.scheme)()
    fs.rm(str(bucket), recursive=True)


@workflow
def clean_project_bucket_workflow(password: str) -> None:
    """
    This workflow will clean the project bucket, but first report the sizes.

    Parameters
    ----------
    password : str
        The password to confirm the deletion of the project bucket. Use
        "flytemosaic" to confirm.
    """
    (
        default_bucket_usage_task()
        >> rm_project_bucket_task(password=password)
        >> default_bucket_usage_task()
    )


@task(
    cache=True,
    cache_version="0.0.1",
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
    enable_deck=True,
)
def plot_mosaic_task(store: str, factor: int) -> None:
    with Client():
        ds = xr.open_zarr(store)
        ds = (
            ds["variables"]
            .isel(band=[2, 1, 0], time=0)
            .coarsen(x=factor, y=factor, boundary="trim")
            .mean()
            .compute()
        )
        fig, ax = plt.subplots()
        ds.plot.imshow(ax=ax, robust=True)  # Ensure the plot is on the figure

        # Save to a BytesIO buffer
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", bbox_inches="tight")  # Save as PNG
        buffer.seek(0)
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        html_str = f'<img src="data:image/png;base64,{encoded}" />'
        flytekit.Deck("Coarsened", html_str)
