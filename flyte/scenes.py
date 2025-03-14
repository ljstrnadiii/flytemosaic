import datetime
from functools import partial
from typing import TYPE_CHECKING, Annotated

import flytekit
import fsspec
import geopandas as gpd
import numpy as np
import pandas as pd
from distributed import Client, LocalCluster
from flytekit import HashMethod, Resources, task
from flytekit.exceptions.user import FlyteRecoverableException
from flytekitplugins.deck.renderer import FrameProfilingRenderer
from shapely import box
from union import map_task, workflow

from flyte.utils import get_default_bucket
from flytemosaic.datasets import DatasetEnum, get_dataset_protocol
from flytemosaic.datasets.protocols import TileDateUrl
from flytemosaic.datasets.utils import urls_exists

if TYPE_CHECKING:
    import s3fs

_EPHEMERAL_STORAGE = 32 * 1024**3
_SCRAPE_CONCURRENCY = 32
_CACHE_VERSION = "0.0.1"


def _cache_version(i: int = 1) -> str:
    return f"{_CACHE_VERSION}.{i}"


@task(
    cache=True,
    cache_version=_cache_version(),
    requests=Resources(cpu="2", mem="8Gi"),
    enable_deck=True,
    deck_fields=None,
)
def get_required_scenes_task(
    bbox: list[float],
    times: list[datetime.datetime],
    dataset_enum: DatasetEnum,
) -> gpd.GeoDataFrame:
    dp = get_dataset_protocol(dataset_enum=dataset_enum)
    times = list({dp.snap_to_temporal_grid(t) for t in times})
    gdf = dp.get_required_scenes_gdf(
        geo=box(*bbox),
        times=times,
    )
    # in case an implemented dataset protocol returns duplicates
    gdf = gdf.drop_duplicates()
    flytekit.Deck(
        "Summary",
        FrameProfilingRenderer().to_html(df=pd.DataFrame(gdf.drop(columns=["geometry"]))),
    )
    return gdf


@task(cache=True, cache_version=_cache_version(2), requests=Resources(cpu="2", mem="8Gi"))
def partition_gdf_task(gdf: gpd.GeoDataFrame, dataset_enum: DatasetEnum) -> list[gpd.GeoDataFrame]:
    sp = get_dataset_protocol(dataset_enum=dataset_enum).scene_protocol
    batch_size = int(_EPHEMERAL_STORAGE / (sp.max_bytes_per_file * 4))
    return [gdf.iloc[i : i + batch_size] for i in range(0, len(gdf), batch_size)]


@task(
    cache=True,
    cache_version=_cache_version(),
    secret_requests=[
        flytekit.Secret(key="glad_user"),
        flytekit.Secret(key="glad_password"),
    ],
    requests=Resources(cpu="4", mem="8Gi", ephemeral_storage=str(_EPHEMERAL_STORAGE)),
    retries=3,  # retry on recoverable errors
)
def scrape_and_upload_batch_task(
    gdf: gpd.GeoDataFrame, dataset_enum: DatasetEnum
) -> gpd.GeoDataFrame:
    ctx = flytekit.current_context()
    scene_source = get_dataset_protocol(dataset_enum=dataset_enum).scene_protocol
    bucket = get_default_bucket()
    # we check if any files have been ingested since we determined which to run
    # and also in the event this task gets retried due to a recoverable error
    # we only want to scrape and upload the missing files
    fs: s3fs.S3FileSystem = fsspec.get_filesystem_class(bucket.scheme)()
    missing = [
        not fs.exists(scene_source.src_url_to_dst_url(src_url=url, bucket=bucket))
        for url in gdf["url"]
    ]
    try:
        scene_source.scrape_tifs_and_upload_cogs_batch(
            gdf=gdf.loc[missing, :],
            workdir=ctx.working_directory,
            bucket=bucket,
        )
    except Exception as e:
        raise FlyteRecoverableException(f"Recoverable error: {e}") from e
    return gdf


@task(
    cache=True,
    cache_version=_cache_version(),
    requests=Resources(cpu="4", mem="8Gi"),
    enable_deck=True,
    deck_fields=None,
)
def determine_scenes_to_ingest(
    gdf: gpd.GeoDataFrame, dataset_enum: DatasetEnum
) -> gpd.GeoDataFrame:
    scene_source = get_dataset_protocol(dataset_enum=dataset_enum).scene_protocol
    bucket = get_default_bucket()
    dst_urls = [scene_source.src_url_to_dst_url(src_url, bucket) for src_url in gdf["url"]]
    exists = urls_exists(urls=dst_urls, bucket=bucket, n_workers=16)
    missing_gdf = gdf.loc[~np.array(exists), :]
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
    cache_version=_cache_version(),
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
    times: list[datetime.datetime],
    dataset: DatasetEnum,
) -> gpd.GeoDataFrame:
    """
    Workflow to ingest scenes for a given dataset within a bounding box and time range.

    Parameters
    ----------
    bbox : list[float]
        The bounding box of which to use for the extent of the final mosaic.
    times : list[datetime.datetime]
        The times to use for final derived feature dates. These may be snapped
        to the underlying temporal grid of the dataset.
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
        times=times,
        dataset_enum=dataset,
    )
    scenes_to_ingest_gdf = determine_scenes_to_ingest(
        gdf=scenes_gdf,
        dataset_enum=dataset,
    )
    batches_of_scenes_gdfs = partition_gdf_task(
        gdf=scenes_to_ingest_gdf,
        dataset_enum=dataset,
    )

    map_task(
        partial(scrape_and_upload_batch_task, dataset_enum=dataset),
        concurrency=_SCRAPE_CONCURRENCY,
    )(gdf=batches_of_scenes_gdfs)

    return generate_expected_scenes_gdf(
        gdf=scenes_gdf,
        dataset_enum=dataset,
    )


@task(cache=True, cache_version=_cache_version())
def get_tile_date_urls_task(
    bbox: list[float],
    dataset_enum: DatasetEnum,
    times: list[datetime.datetime],
) -> list[TileDateUrl]:
    dp = get_dataset_protocol(dataset_enum=dataset_enum)
    bucket = get_default_bucket()
    times = list({dp.snap_to_temporal_grid(t) for t in times})
    return dp.get_tile_date_urls(geo=box(*bbox), times=times, bucket=bucket)


@task(
    cache=True,
    cache_version=_cache_version(),
    requests=Resources(cpu="4", mem="8Gi"),
)
def determine_tile_dates_to_ingest(
    tile_date_urls: list[TileDateUrl],
) -> list[TileDateUrl]:
    exists = urls_exists(
        urls=[tdurl.url for tdurl in tile_date_urls],
        bucket=get_default_bucket(),
        n_workers=16,  # used for localcluster given resources above
    )
    return [td for td, r in zip(tile_date_urls, exists, strict=True) if not r]


@task(
    cache=True,
    cache_version=_cache_version(2),
    requests=Resources(cpu="3", mem="8Gi"),
)
def build_tile_date_feature_cog_task(
    tile_date_url: TileDateUrl,
    dataset_enum: DatasetEnum,
) -> TileDateUrl:
    dp = get_dataset_protocol(dataset_enum=dataset_enum)
    ctx = flytekit.current_context()
    with LocalCluster(n_workers=3) as cluster, Client(cluster):
        return dp.build_tile_date_cog(
            tile_id=tile_date_url.tile_id,
            time=tile_date_url.time,
            bucket=get_default_bucket(),
            workdir=ctx.working_directory,
        )


def _hash_dataframe(gdf: gpd.GeoDataFrame) -> str:
    return str(pd.util.hash_pandas_object(gdf))


@task(cache=True, cache_version=_cache_version())
def build_tile_date_url_gdf(
    tile_date_urls: list[TileDateUrl], dataset_enum: DatasetEnum
) -> Annotated[gpd.GeoDataFrame, HashMethod(_hash_dataframe)]:
    dp = get_dataset_protocol(dataset_enum=dataset_enum)
    geos = dp.tiles_to_geos(tile_ids=[tdu.tile_id for tdu in tile_date_urls])
    gdf = gpd.GeoDataFrame(tile_date_urls, geometry=geos, crs="EPSG:4326")
    gdf.rename(columns={"time": "datetime"}, inplace=True)
    return gdf[["url", "geometry", "datetime"]]


@workflow
def build_scene_features_workflow(
    bbox: list[float],
    times: list[datetime.datetime],
    dataset: DatasetEnum,
) -> gpd.GeoDataFrame:
    """
    Build derived features from scene-level cogs for tiles touching bbox at times.

    Parameters
    ----------
    bbox : list[float]
        The bounding box of which to use for the extent of the final mosaic.
    times : list[datetime.datetime]
        The times to build the derived features for.
    dataset : DatasetEnum
        The dataset to use to build the derived features.

    Returns
    -------
    gpd.GeoDataFrame
        A GeoDataFrame of the expected COGs required to create the final mosaic
        with the provided dataset. Will contains columns 'url', 'geometry',
        'datetime'.
    """
    tile_date_urls = get_tile_date_urls_task(
        bbox=bbox,
        dataset_enum=dataset,
        times=times,
    )

    missing_tile_date_urls = determine_tile_dates_to_ingest(tile_date_urls=tile_date_urls)

    map_task(
        partial(build_tile_date_feature_cog_task, dataset_enum=dataset),
        concurrency=32,
    )(tile_date_url=missing_tile_date_urls)

    return build_tile_date_url_gdf(
        tile_date_urls=tile_date_urls,
        dataset_enum=dataset,
    )
