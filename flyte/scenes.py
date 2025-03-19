import datetime
from typing import Annotated

import flytekit
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
    datasets: list[DatasetEnum],
) -> gpd.GeoDataFrame:
    gdfs = []
    for dataset in datasets:
        dp = get_dataset_protocol(dataset_enum=dataset)
        times = list({dp.snap_to_temporal_grid(t) for t in times})
        gdf = dp.get_required_scenes_gdf(
            geo=box(*bbox),
            times=times,
        )
        gdf["feature"] = dataset.value
        gdfs.append(gdf)
    gdf = pd.concat(gdfs, ignore_index=True)
    gdf = gdf.drop_duplicates()
    flytekit.Deck(
        "Summary",
        FrameProfilingRenderer().to_html(df=pd.DataFrame(gdf.drop(columns=["geometry"]))),
    )
    return gdf


@task(cache=True, cache_version=_cache_version(2), requests=Resources(cpu="2", mem="8Gi"))
def partition_gdf_task(gdf: gpd.GeoDataFrame) -> list[gpd.GeoDataFrame]:
    # at some point returning too many gdfs will slow things down and we could
    # parallelize with a map task.
    batches = []
    for nm, grp in gdf.groupby("feature"):
        dataset_enum = DatasetEnum(nm)
        sp = get_dataset_protocol(dataset_enum=dataset_enum).scene_protocol
        batch_size = int(_EPHEMERAL_STORAGE / (sp.max_bytes_per_file * 4))
        batches += [gdf.iloc[i : i + batch_size] for i in range(0, len(grp), batch_size)]
    return batches


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
def scrape_and_upload_batch_task(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    ctx = flytekit.current_context()
    # we should only have one single dataset enum per batch
    datasets = gdf["feature"].unique()
    if len(datasets) != 1:
        raise ValueError("Batch contains multiple datasets. This is unexpected.")
    dataset_enum = DatasetEnum(datasets[0])
    scene_source = get_dataset_protocol(dataset_enum=dataset_enum).scene_protocol
    bucket = get_default_bucket()
    # we check if any files have been ingested since we determined which to run
    # and also in the event this task gets retried due to a recoverable error
    # we only want to scrape and upload the missing files
    urls = [scene_source.src_url_to_dst_url(src_url=url, bucket=bucket) for url in gdf["url"]]
    exists = urls_exists(urls=urls, bucket=bucket)
    try:
        scene_source.scrape_tifs_and_upload_cogs_batch(
            gdf=gdf.loc[~np.array(exists), :],
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
def determine_scenes_to_ingest(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    grps = []
    for nm, grp in gdf.groupby("feature"):
        dataset_enum = DatasetEnum(nm)
        scene_source = get_dataset_protocol(dataset_enum=dataset_enum).scene_protocol
        bucket = get_default_bucket()
        grp["dst_url"] = [
            scene_source.src_url_to_dst_url(src_url, bucket) for src_url in grp["url"]
        ]
        exists = urls_exists(urls=grp["dst_url"], bucket=bucket, n_workers=16)
        missing = grp.loc[~np.array(exists), :]
        grps.append(missing)
    missing_gdf = pd.concat(grps, ignore_index=True)
    # use dst url to drop duplicate scene ingest tiles
    missing_gdf.drop_duplicates(subset=["dst_url"], inplace=True)
    missing_gdf.drop(columns=["dst_url"], inplace=True)
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
def generate_expected_scenes_gdf(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdfs = []
    bucket = get_default_bucket()
    for nm, grp in gdf.groupby("feature"):
        dataset_enum = DatasetEnum(nm)
        dp = get_dataset_protocol(dataset_enum=dataset_enum).scene_protocol
        grp["url"] = grp["url"].apply(
            lambda url, dp=dp, bucket=bucket: dp.src_url_to_dst_url(url, bucket=bucket)
        )
        gdfs.append(grp)
    gdf = pd.concat(gdfs, ignore_index=True)
    flytekit.Deck(
        "Summary",
        FrameProfilingRenderer().to_html(df=pd.DataFrame(gdf.drop(columns=["geometry"]))),
    )
    return gdf


@workflow
def ingest_scenes_workflow(
    bbox: list[float],
    times: list[datetime.datetime],
    datasets: list[DatasetEnum],
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
    datasets : list[DatasetEnum]
        The datasets to ingest which determines the scenes to scrape and upload.

    Returns
    -------
    gpd.GeoDataFrame
        A GeoDataFrame of the expected COGs required to create the final mosaic
        with the provided dataset. Will contains columns 'url', 'geometry', 'datetime'.
    """
    scenes_gdf = get_required_scenes_task(
        bbox=bbox,
        times=times,
        datasets=datasets,
    )
    scenes_to_ingest_gdf = determine_scenes_to_ingest(gdf=scenes_gdf)
    batches_of_scenes_gdfs = partition_gdf_task(gdf=scenes_to_ingest_gdf)
    map_task(scrape_and_upload_batch_task, concurrency=_SCRAPE_CONCURRENCY)(
        gdf=batches_of_scenes_gdfs
    )
    return generate_expected_scenes_gdf(gdf=scenes_gdf)


@task(cache=True, cache_version=_cache_version())
def get_tile_date_urls_task(
    bbox: list[float],
    datasets: list[DatasetEnum],
    times: list[datetime.datetime],
) -> list[TileDateUrl]:
    tdurls = []
    bucket = get_default_bucket()
    for dataset in datasets:
        dp = get_dataset_protocol(dataset_enum=dataset)
        times = list({dp.snap_to_temporal_grid(t) for t in times})
        tdurls += dp.get_tile_date_urls(geo=box(*bbox), times=times, bucket=bucket)
    return tdurls


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
def build_tile_date_feature_cog_task(tile_date_url: TileDateUrl) -> TileDateUrl:
    dp = get_dataset_protocol(dataset_enum=DatasetEnum(tile_date_url.feature))
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
    tile_date_urls: list[TileDateUrl],
) -> Annotated[gpd.GeoDataFrame, HashMethod(_hash_dataframe)]:
    gdfs = []
    df = pd.DataFrame(tile_date_urls)
    for nm, grp in df.groupby("feature"):
        dp = get_dataset_protocol(dataset_enum=DatasetEnum(nm))
        geos = dp.tiles_to_geos(tile_ids=grp["tile_id"].tolist())
        gdfs.append(gpd.GeoDataFrame(grp, geometry=geos, crs="EPSG:4326"))
    gdf = pd.concat(gdfs, ignore_index=True)
    gdf.rename(columns={"time": "datetime"}, inplace=True)
    return gdf[["url", "geometry", "datetime", "feature"]]


@workflow
def build_scene_features_workflow(
    bbox: list[float],
    times: list[datetime.datetime],
    datasets: list[DatasetEnum],
) -> gpd.GeoDataFrame:
    """
    Build derived features from scene-level cogs for tiles touching bbox at times.

    Parameters
    ----------
    bbox : list[float]
        The bounding box of which to use for the extent of the final mosaic.
    times : list[datetime.datetime]
        The times to build the derived features for.
    datasets : list[DatasetEnum]
        The datasets to use to build the derived features.

    Returns
    -------
    gpd.GeoDataFrame
        A GeoDataFrame of the expected COGs required to create the final mosaic
        with the provided datasets. Will contains columns 'url', 'geometry',
        'datetime' and "feature".
    """
    tile_date_urls = get_tile_date_urls_task(
        bbox=bbox,
        datasets=datasets,
        times=times,
    )
    missing_tile_date_urls = determine_tile_dates_to_ingest(tile_date_urls=tile_date_urls)
    map_task(build_tile_date_feature_cog_task, concurrency=32)(tile_date_url=missing_tile_date_urls)
    return build_tile_date_url_gdf(tile_date_urls=tile_date_urls)
