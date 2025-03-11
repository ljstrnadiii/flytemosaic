import datetime
from functools import partial

import dask
import flytekit
import fsspec
import geopandas as gpd
import pandas as pd
from distributed import Client, LocalCluster
from flytekit import Resources, task
from flytekit.exceptions.user import FlyteUserException
from flytekitplugins.deck.renderer import FrameProfilingRenderer
from more_itertools import chunked
from shapely import box
from union import map_task, workflow

from flyte.utils import get_default_bucket
from flytemosaic.datasets import DatasetEnum, get_dataset_protocol

_EPHEMERAL_STORAGE = 32 * 1024**3
_SCRAPE_CONCURRENCY = 32  # be nice to UMD's servers


@task(
    cache=True,
    cache_version="0.0.3",
    requests=Resources(cpu="2", mem="8Gi"),
    enable_deck=True,
    deck_fields=None,
)
def get_required_scenes_task(
    bbox: list[float],
    start: datetime.datetime,
    end: datetime.datetime,
    dataset_enum: DatasetEnum,
) -> gpd.GeoDataFrame:
    dp = get_dataset_protocol(dataset_enum=dataset_enum)
    gdf = dp.get_required_scenes_gdf(
        geo=box(*bbox),
        start=start.replace(tzinfo=None),
        end=end.replace(tzinfo=None),
    )
    # in case an implemented dataset protocol returns duplicates
    gdf = gdf.drop_duplicates()
    flytekit.Deck(
        "Summary",
        FrameProfilingRenderer().to_html(df=pd.DataFrame(gdf.drop(columns=["geometry"]))),
    )
    return gdf


@task(cache=True, cache_version="0.0.1", requests=Resources(cpu="2", mem="8Gi"))
def partition_gdf_task(gdf: gpd.GeoDataFrame, dataset_enum: DatasetEnum) -> list[gpd.GeoDataFrame]:
    sp = get_dataset_protocol(dataset_enum=dataset_enum).scene_protocol
    batch_size = int(_EPHEMERAL_STORAGE / (sp.max_bytes_per_file * 2))
    return [gdf.iloc[i : i + batch_size] for i in range(0, len(gdf), batch_size)]


@task(
    cache=True,
    cache_version="0.0.1",
    secret_requests=[
        flytekit.Secret(key="glad_user"),
        flytekit.Secret(key="glad_password"),
    ],
    requests=Resources(cpu="2", mem="8Gi", ephemeral_storage=str(_EPHEMERAL_STORAGE)),
    enable_deck=True,
    deck_fields=None,
    retries=3,
)
def scrape_and_upload_batch_task(
    gdf: gpd.GeoDataFrame, dataset_enum: DatasetEnum
) -> gpd.GeoDataFrame:
    ctx = flytekit.current_context()
    scene_source = get_dataset_protocol(dataset_enum=dataset_enum).scene_protocol

    try:
        gdf = scene_source.scrape_tifs_uploads_cogs_batch(
            gdf=gdf,
            workdir=ctx.working_directory,
            bucket=get_default_bucket(),
            user=ctx.secrets.get(key="glad_user"),
            password=ctx.secrets.get(key="glad_password"),
        )
    except Exception as e:
        raise FlyteUserException(f"Failed to scrape and upload batch: {e}") from e

    flytekit.Deck(
        "Summary",
        FrameProfilingRenderer().to_html(df=pd.DataFrame(gdf.drop(columns=["geometry"]))),
    )
    return gdf


@task(
    cache=True,
    cache_version="0.0.2",
    requests=Resources(cpu="4", mem="8Gi"),
    enable_deck=True,
    deck_fields=None,
)
def determine_scenes_to_ingest(
    gdf: gpd.GeoDataFrame, dataset_enum: DatasetEnum
) -> gpd.GeoDataFrame:
    # determine remote paths
    scene_source = get_dataset_protocol(dataset_enum=dataset_enum).scene_protocol
    bucket = get_default_bucket()
    dst_urls = [scene_source.src_url_to_dst_url(src_url, bucket) for src_url in gdf["url"]]
    fs = fsspec.get_filesystem_class(bucket.scheme)()
    results = []
    with LocalCluster(n_workers=16) as cluster, Client(cluster):
        for chunk in chunked(dst_urls, int(2**14)):
            tasks = [dask.delayed(lambda url: fs.exists(url))(url) for url in chunk]
            results += dask.compute(tasks)
    missing = [not r for rs in results for r in rs]
    missing_gdf = gdf.loc[missing, :]
    if len(missing_gdf) > 0:
        flytekit.Deck(
            "Summary",
            FrameProfilingRenderer().to_html(
                df=pd.DataFrame(missing_gdf.drop(columns=["geometry"]))
            ),
        )
    return missing_gdf


@task(
    cache=True,
    cache_version="0.0.3",
    requests=Resources(cpu="4", mem="8Gi"),
    enable_deck=True,
    deck_fields=None,
)
def generate_expected_scenes_gdf(
    gdf: gpd.GeoDataFrame, dataset_enum: DatasetEnum
) -> gpd.GeoDataFrame:
    dp = get_dataset_protocol(dataset_enum=dataset_enum).scene_protocol
    bucket = get_default_bucket()
    gdf["url"] = gdf["url"].apply(lambda url: dp.src_url_to_dst_url(url, bucket=bucket))
    flytekit.Deck(
        "Summary",
        FrameProfilingRenderer().to_html(df=pd.DataFrame(gdf.drop(columns=["geometry"]))),
    )
    return gdf


@workflow
def ingest_scenes_workflow(
    bbox: list[float],
    start: datetime.datetime,
    end: datetime.datetime,
    dataset: DatasetEnum,
) -> gpd.GeoDataFrame:
    """
    Workflow to ingest scenes for a given dataset within a bounding box and time range.

    Parameters
    ----------
    bbox : list[float]
        The bounding box of which to use for the extent of the final mosaic.
    start : datetime.datetime
        The start date of the time range.
    end : datetime.datetime
        The end date of the time range.
    dataset : DatasetEnum
        The dataset to ingest which determines the scenes to scrape and upload.

    Returns
    -------
    gpd.GeoDataFrame
        A GeoDataFrame of the expected COGs required to create the final mosaic
        with the provided dataset. Will contains columns 'url', 'geometry', 'datetime'.
    """
    scenes_gdf = get_required_scenes_task(
        bbox=bbox,
        start=start,
        end=end,
        dataset_enum=dataset,
    )
    scenes_to_ingest = determine_scenes_to_ingest(
        gdf=scenes_gdf,
        dataset_enum=dataset,
    )
    batches_of_scenes_gdfs = partition_gdf_task(
        gdf=scenes_to_ingest,
        dataset_enum=dataset,
    )

    scrape_upload_partial = partial(
        scrape_and_upload_batch_task,
        dataset_enum=dataset,
    )

    map_task(
        scrape_upload_partial,
        concurrency=_SCRAPE_CONCURRENCY,
    )(gdf=batches_of_scenes_gdfs)

    return generate_expected_scenes_gdf(
        gdf=scenes_gdf,
        dataset_enum=dataset,
    )


# if __name__ == "__main__":
#     gdf = ingest_scenes_workflow(
#         bbox=[
#             -109.05919619986199,
#             36.99275055519555,
#             -102.04212644366443,
#             41.00198213121131,
#         ],
#         start=datetime.datetime(2021, 1, 1),
#         end=datetime.datetime(2021, 1, 2),
#         dataset_enum=DatasetEnum.GLAD_ARD_ANNUAL_MEAN,
#         download_scene_batch_size=10,
#     )
#     gdf
