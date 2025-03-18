import datetime
from dataclasses import dataclass
from functools import partial
from itertools import groupby
from random import shuffle

import dask
import dask.config
import flytekit
import geopandas as gpd
import pandas as pd
import xarray as xr
from flytekit import Resources, map_task, task, workflow
from flytekit.types.file import FlyteFile
from pyproj import CRS
from union import Deck

from flyte.scenes import build_scene_features_workflow, ingest_scenes_workflow
from flyte.utils import get_default_bucket
from flytemosaic import gdal_configs
from flytemosaic.datasets import DatasetEnum, get_dataset_protocol
from flytemosaic.mosaics import (
    TemporalGTIMosaic,
    build_gti_xarray,
    build_mosaic_chunk_partitions,
    build_recommended_gti,
    build_temporal_mosaic,
)

_CACHE_VERSION = "0.0.1"


def _cache_version(i: int = 1) -> str:
    return f"{_CACHE_VERSION}.{i}"


@task(cache=True, cache_version=_cache_version(2))
def build_gti_inputs_task(
    gdf: gpd.GeoDataFrame, bounds: list[float]
) -> tuple[list[gpd.GeoDataFrame], list[list[float]]]:
    gdfs = [grp for _, grp in gdf.groupby(["datetime", "feature"])]
    return gdfs, [bounds] * len(gdfs)


@dataclass
class GTIResult:
    time: datetime.datetime
    gti: FlyteFile
    dataset: DatasetEnum


@task(cache=True, cache_version=_cache_version())
def gdf_to_gti_task(
    gdf: gpd.GeoDataFrame,
    bounds: list[float],
    crs: str,
    resolution: float,
) -> GTIResult:
    time = gdf["datetime"].unique()[0]  # should always be length 1
    unique_dsets = gdf["feature"].unique()
    if len(unique_dsets) != 1:
        raise ValueError(f"Expected 1 dataset but got {len(unique_dsets)}: {unique_dsets}")
    feature = unique_dsets[0]
    dataset_enum = DatasetEnum(feature)
    dp = get_dataset_protocol(dataset_enum=dataset_enum)
    gti_file = FlyteFile.new(filename="index.gti.fgb")
    build_recommended_gti(
        gdf=gdf,
        url_col_name="url",
        dst=gti_file.path,
        crs=CRS.from_string(crs),
        bounds=bounds,  # type: ignore # noqa: PGH003 # flyte doesn't like tuples
        dtype=dp.dtype,
        resx=resolution,
        resy=resolution,
        nodata=dp.nodata,
        band_count=len(dp.bands),
        resampling="average",
    )
    return GTIResult(time=time, gti=gti_file, dataset=dataset_enum)


@task(
    cache=True,
    cache_version=_cache_version(),
    requests=Resources(cpu="3", mem="8Gi"),
    enable_deck=True,
    deck_fields=None,
)
def build_target_mosaic_task(
    gtis: list[GTIResult],
    xy_chunksize: int,
) -> str:
    gti_mosaics = [
        TemporalGTIMosaic(
            gti=gti.gti.download(),
            chunksize=xy_chunksize,
            time=gti.time,
            dataset=gti.dataset,
        )
        for gti in gtis
    ]
    store = str(
        get_default_bucket()
        / "zarr_mosaics"
        / flytekit.current_context().execution_id.name
        / "mosaic.zarr"
    )
    target = build_temporal_mosaic(gti_mosaics=gti_mosaics)
    target.to_zarr(store, compute=False)
    Deck("Target Mosaic Store", target._repr_html_())
    return store


@dataclass
class GTIPartition:
    gti: GTIResult
    partition: dict[str, tuple[int, int]]


@task(
    cache=True,
    cache_version=_cache_version(),
    requests=Resources(cpu="3", mem="8Gi"),
)
def build_gti_partitions_task(
    store: str, chunk_partition_size: int, gtis: list[GTIResult]
) -> list[GTIPartition]:
    ds = xr.open_zarr(store)
    gti_partitions = []
    for nm, grp in groupby(sorted(gtis, key=lambda x: x.dataset.name), lambda x: x.dataset):
        dp = get_dataset_protocol(dataset_enum=nm)
        partition_indices = list(
            build_mosaic_chunk_partitions(
                ds=ds.chunk({"time": 1}),
                chunk_partition_size=int(chunk_partition_size),
                variable_name="variables",
                bands=dp.bands,
            )
        )
        gti_time = {gti.time: gti for gti in grp}
        for partition in partition_indices:
            t = ds.time.isel(time=slice(*partition["time"])).data[0]
            gti = gti_time[pd.Timestamp(t).to_pydatetime()]
            gti_partitions.append(GTIPartition(gti=gti, partition=partition))
    shuffle(gti_partitions)
    return gti_partitions


@task(
    cache=True,
    cache_version=_cache_version(),
    requests=Resources(cpu="3", mem="8Gi", ephemeral_storage="16Gi"),
    environment=gdal_configs.get_worker_config(8, debug=True),
)
def write_mosaic_partition_task(
    gti_partition: GTIPartition,
    store: str,
    xy_chunksize: int,
) -> bool:
    # single threaded dask scheduler but ALL_CPUS for GDAL_NUM_THREADS in environment
    with dask.config.set(scheduler="single-threaded"):
        dp = get_dataset_protocol(dataset_enum=gti_partition.gti.dataset)
        ds_time = build_gti_xarray(
            gti=gti_partition.gti.gti.download(),
            chunksize=xy_chunksize,
            band_names=dp.bands,
            time=gti_partition.gti.time,
        )
        region = {k: slice(*v) for k, v in gti_partition.partition.items()}
        region_wo_time_slc = {k: v for k, v in region.items() if k != "time"}
        subset = ds_time.isel(**region_wo_time_slc)  # type: ignore  # noqa: PGH003# type: ignore  # noqa: PGH003
        subset["variables"].attrs.clear()
        subset.drop("spatial_ref").to_zarr(store, region=region)
    return True


@workflow
def build_dataset_mosaic_workflow(
    bbox: list[float],
    times: list[datetime.datetime],
    datasets: list[DatasetEnum],
    resolution: float,
    crs: str,
    chunk_partition_size: int,
    xy_chunksize: int = 2048,
) -> str:
    scenes_gdf = ingest_scenes_workflow(
        bbox=bbox,
        times=times,
        datasets=datasets,
    )
    scene_features = build_scene_features_workflow(
        bbox=bbox,
        times=times,
        datasets=datasets,
    )
    scenes_gdf >> scene_features

    gdf_grouped, bounds = build_gti_inputs_task(gdf=scene_features, bounds=bbox)

    gtis = map_task(
        partial(
            gdf_to_gti_task,
            crs=crs,
            resolution=resolution,
        )
    )(gdf=gdf_grouped, bounds=bounds)

    store = build_target_mosaic_task(gtis=gtis, xy_chunksize=xy_chunksize)

    gti_partitions = build_gti_partitions_task(
        store=store,
        chunk_partition_size=chunk_partition_size,
        gtis=gtis,
    )

    map_task(
        partial(
            write_mosaic_partition_task,
            store=store,
            xy_chunksize=xy_chunksize,
        ),
        concurrency=32,
    )(gti_partition=gti_partitions)

    return store
