import base64
import io
from urllib.parse import urlparse

import flytekit
import xarray as xr
from distributed import Client
from flytekit.core.context_manager import FlyteContextManager
from flytekitplugins.dask.task import Dask, Scheduler, WorkerGroup
from matplotlib import pyplot as plt
from union import Resources, task
from yarl import URL

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
        return URL(parsed_url.path)
    else:
        return URL(f"{parsed_url.scheme}://{parsed_url.netloc}") / PROJECT_DIR


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
