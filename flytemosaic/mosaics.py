import datetime
import subprocess
import tempfile
from collections.abc import Generator
from dataclasses import dataclass
from itertools import groupby, product
from math import floor
from pathlib import Path

import geopandas as gpd
import numpy as np
import rioxarray
import xarray as xr
from pyproj import CRS
from rasterio.enums import Resampling

from flytemosaic.datasets import DatasetEnum, get_dataset_protocol


def build_recommended_gti(
    gdf: gpd.GeoDataFrame,
    url_col_name: str,
    dst: str,
    crs: CRS,
    bounds: tuple[float, float, float, float],
    resx: float,
    resy: float,
    band_count: int,
    dtype: str = "Float32",
    nodata: str = "NaN",
    resampling: str = Resampling.average.name,
) -> None:
    """
    Build a recommended GDAL Raster Tile Index from a GeoDataFrame of ingested COGs.

    See https://gdal.org/en/latest/drivers/raster/gti.html for details on the
    recommended GTI format such as adding dtype to avoid unnecessary http requests
    to inspect COGs when loading the mosaic with rioxarray.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame of ingested COGs with columns [location, time, geometry]
        where the location is NOT yet converted to VSI path as we convert it here
        e.g. gs://bucket/path/to/file.tif -> /vsigs/bucket/path/to/file.tif

        The provided gdf should only contain COGs with the same bands, dtype, nodata,
        and date if relevant as the last pixel will be used for overlapping pixels.
        Consider building one GTI per datasource and time period. The COGs ARE
        allowed to be in different CRS!
    dst : str
        The path to the output GTI file. This should be a VSI path if you want
        or a local path with existing parent dirs. The caller will likely be
        best managers of temporary files. suffix with `*.gti.fgb`.
    crs : CRS
        The CRS of the GTI. This is used to set the SRS metadata in the GTI.
    bounds : tuple[float, float, float, float]
        The bounds of the mosaic in EPSG:4326 regardless of the provided crs?
    resx : float
        The x resolution akong x in srs units.
    resy : float
        The y resolution along y in srs units.
    band_count : int
        The number of bands in the COGs.
    dtype : str
        See GDAL GTI docs on DATA_TYPE. Default is "Float32" and NOT "float32".
    nodata : str
        See GDAL GTI docs on NODATA. Default is "NaN" while dtype is floating-point.
    resampling : str
        The resampling method to use for the GTI. Default is "average" and
        see GDAL GTI docs on RESAMPLING.
    """
    gdf = gdf.to_crs(crs)
    gdf["location"] = (
        gdf[url_col_name].str.replace("gs://", "/vsigs/").str.replace("s3://", "/vsis3/")
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        tile_index_path = str(Path(tmpdir) / "prelim.gti.fgb")
        gdf.to_file(tile_index_path, driver="FlatGeobuf", mode="w")
        subprocess.run(
            [
                "ogr2ogr",
                "-f",
                "FlatGeobuf",
                dst,
                tile_index_path,
                "-mo",
                f"DATA_TYPE={dtype}",
                "-mo",
                f"RESAMPLING={resampling}",
                "-mo",
                f"NODATA={nodata}",
                # these are in epsg:4326 or the srs?
                "-mo",
                f"MINX={bounds[0]}",
                "-mo",
                f"MINY={bounds[1]}",
                "-mo",
                f"MAXX={bounds[2]}",
                "-mo",
                f"MAXY={bounds[3]}",
                "-mo",
                f"BAND_COUNT={band_count}",
                "-mo",
                f"RESX={resx}",
                "-mo",
                f"RESY={resy}",
                # being verbose here though crs would be inferred from gdf
                "-mo",
                f"SRS={crs.to_proj4()}",
            ],
            check=True,
        )


def build_gti_xarray(
    gti: str,
    chunksize: int,
    band_names: list[str],
    resx: float | None = None,
    resy: float | None = None,
    time: datetime.datetime | None = None,
) -> xr.Dataset:
    """
    Build an xarray Dataset from a GTI file for a specific date and source.

    The final dims are [band, y, x] with the band names as provided or if time
    is provided, [time, band, y, x]. The usual spatial_ref coordinate will also
    be present since we use rioxarray to open the GTI.

    Given gti has the recommended metadata as set in :func:`build_recommended_gti`,
    this should be very snappy since there is no need to make any http requests
    to inspect the COGs. This can be a big improvement for compared to VRTs as
    GDAL needs to inspect the COGs, which can be a big problem for many many files.

    Parameters
    ----------
    gti : str
        The path to the GTI file likely built from :func:`build_recommended_gti`.
    chunksize : int
        The chunksize to use for the DataArray in x and y.
    band_names : list[str]
        The list of band names to use for the DataArray.
    resx : float
        The optional x resolution if different than the one in the GTI. Since this
        is supported as an open kwarg, we can use the same gti, but use a
        different resolution.
    resy : float
        The optional y resolution if different than the one in the GTI. Since this
        is supported as an open kwarg, we can use the same gti, but use a
        different resolution.
    time : datetime.datetime | None
        The optional time to add as a dimension to the DataArray.

    Returns
    -------
    xr.DataArray
        A lazy mosaic data array
    """
    # see https://gdal.org/en/latest/drivers/raster/gti.html#open-options
    open_kwargs = {}
    if resx is not None and resy is not None:
        open_kwargs = {"resx": resx, "resy": resy}

    da: xr.DataArray = rioxarray.open_rasterio(  # type: ignore  # noqa: PGH003
        gti,
        chunks=(1, chunksize, chunksize),
        lock=False,
        open_kwargs=open_kwargs,
    )
    da["band"] = band_names
    if time is not None:
        da = da.expand_dims("time")
        da["time"] = [time]
    return da.to_dataset(name="variables")


@dataclass
class TemporalGTIMosaic:
    gti: str
    chunksize: int
    time: datetime.datetime
    dataset: DatasetEnum


def build_temporal_mosaic(gti_mosaics: list[TemporalGTIMosaic]) -> xr.Dataset:
    """
    Given temporal mosaics, build a target mosaic for to_zarr(..., compute=False).

    Do NOT call .compute() on this DataArray as rioxarray appears to have a sticky
    datareader and will use the most recent gti index for all arrays. We bypass
    this issue by using this function only for .to_zarr(..., compute=False) and
    then in separate processes call .to_zarr(..., region=...) to write out to a
    specific region that corresponds to a single TemporalGTIMosaic. We can easily
    afford this since gti is very fast to construct and open with rioxarray.

    Parameters
    ----------
    gti_mosaics : list[GTIMosaic]
        The list of GTIMosaics to build the mosaic from.

    Returns
    -------
    xr.DataArray
        The mosaic DataArray with dims [time, band, y, x].
    """
    feature_dsets = []
    for nm, grp in groupby(
        sorted(gti_mosaics, key=lambda x: x.dataset.name),
        key=lambda x: x.dataset,
    ):
        dp = get_dataset_protocol(dataset_enum=nm)
        feature_dsets.append(
            xr.concat(
                [
                    build_gti_xarray(
                        gti=gti.gti,
                        chunksize=gti.chunksize,
                        band_names=dp.bands,
                        time=gti.time,
                    )
                    for gti in grp
                ],
                dim="time",
            )
        )
    return (
        xr.concat(feature_dsets, dim="band").transpose("time", "band", "y", "x").chunk({"time": 1})
    )


def build_mosaic_chunk_partitions(
    ds: xr.Dataset, chunk_partition_size: float, variable_name: str, bands: list[str]
) -> Generator[dict[str, tuple[int, int]], None, None]:
    """
    Compute start/stop indices for integer multiples of chunks in x and y.

    This is used to slice integer multiples of chunks in x and y equally to give
    a mechanism to partition the dataset into smaller units of work. First, x
    and y equally means that incremental increases in chunk partitions are n^2.
    That means the next size up might be larger than chunk_partition_size. For
    example, if chunk_partition_size is 128mb, the next size up is 4x=512mb,
    9x=1152mb, 16x=2048mb, etc.

    Parameters
    ----------
    ds : xr.Dataset
        The xarray array from which to use chunk sizes to build partitions.
    chunk_partition_size : float
        The maximum size of the partition in bytes. The final size of the partition
        will be the largest integer multiple of the chunks in x and y and include
        all the bands if provided.
    variable_name : str
        The example variable to use to inspect underlying chunks for size.
    bands : list[str]
        The list of bands to use for the partitioned chunks. If provided, the
        final chunks will include start and stops for the bands. This means that
        the bands must be contiguous and will raise an error if not.

    Returns
    -------
    Generator[dict[str, list[tuple[int, int]]], None, None]
        A generator of dictionaries with keys as dimension names and values as
        lists of tuples of start and end indices for each chunk partition. We
        avoid slice objects explicitly in favor of primitive types for easier
        serialization for distributed computing.

    Raises
    ------
    ValueError
        If the optional bands are not contiguous coordinates.
    """
    # chunks include all bands
    subset = ds.sel(band=bands) if bands else ds
    subset = subset.chunk({"band": -1})

    # estimate the bytes per chunk
    arr = subset[variable_name].data
    bytes_per_chunk = int(np.prod(arr.chunksize) * arr.dtype.itemsize)

    xy_multiplier = max(1, floor((chunk_partition_size // bytes_per_chunk) ** 0.5))
    if xy_multiplier != 1:
        x_chunksize = xy_multiplier * subset.chunksizes["x"][0]
        y_chunksize = xy_multiplier * subset.chunksizes["y"][0]
        subset = subset.chunk(x=x_chunksize, y=y_chunksize)

    chunk_idx = {
        dim: np.cumsum([0] + list(subset.chunksizes[dim])).tolist() for dim in subset.chunksizes
    }
    # Use the original dataset to create offset for provided bands
    if bands:
        idx = [ds.get_index("band").get_loc(v) for v in bands]
        # check that idx is contiguous
        if idx != list(range(idx[0], idx[-1] + 1)):
            raise ValueError("Band indices are not contiguous")
        chunk_idx.update({"band": (idx[0], idx[-1] + 1)})

    chunks = {
        dim: [(chunk_idx[dim][i], chunk_idx[dim][i + 1]) for i in range(len(chunk_idx[dim]) - 1)]
        for dim in chunk_idx
    }
    for slc in product(*chunks.values()):
        yield dict(zip(list(subset.chunksizes.keys()), slc, strict=True))  # type: ignore  # noqa: PGH003
