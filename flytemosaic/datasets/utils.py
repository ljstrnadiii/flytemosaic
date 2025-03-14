import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import dask
import fsspec
import fsspec.implementations
import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray
import rioxarray.raster_array
import xarray as xr
from distributed import Client, LocalCluster
from more_itertools import chunked
from yarl import URL

if TYPE_CHECKING:
    import s3fs


def download_files_with_aria(
    df: pd.DataFrame | gpd.GeoDataFrame,
    url_column: str,
    workdir: str,
    user: str | None = None,
    password: str | None = None,
) -> None:
    """
    Helper to download files in a dataframe with a url column using aria2c.

    Parameters
    ----------
    df : pd.DataFrame | gpd.GeoDataFrame
        The DataFrame with the urls to download. Must have 'url' column.
    url_column : str
        The name of the column with the urls.
    workdir : str
        The directory to download the files.
    user : str, optional
        The username for the website, by default None. If not provided, user should
        have .netrc file with credentials.
    password : str, optional
        The password for the website, by default None. If not provided, user should
        have .netrc file with credentials.
    """
    urls = df[url_column].tolist()
    url_list_path = os.path.join(workdir, "urls.txt")
    with open(url_list_path, "w") as f:
        for url in urls:
            f.write(url + "\n")
            # match the url dir structure by adding the out parameter option
            # https://aria2.github.io/manual/en/html/aria2c.html#id2
            f.write("\t" + "out=" + URL(url).path.lstrip("/") + "\n")

    subprocess.run(
        [
            "aria2c",
            "--input-file=" + url_list_path,
            "--dir=" + workdir,
            "--max-concurrent-downloads=1",
        ]
        + (["--http-user=" + user, "--http-passwd=" + password] if user and password else []),
        check=True,
    )


def download_files_with_fsspec(
    df: pd.DataFrame | gpd.GeoDataFrame,
    url_column: str,
    workdir: str,
) -> None:
    """
    Helper to download files in a dataframe with a url column using aria2c.

    Parameters
    ----------
    df : pd.DataFrame | gpd.GeoDataFrame
        The DataFrame with the urls to download. Must have 'url' column.
    url_column : str
        The name of the column with the urls.
    workdir : str
        The directory to download the files.
    """
    urls = df[url_column].tolist()
    if len(df) > 0:
        fs: s3fs.S3FileSystem = fsspec.get_filesystem_class(URL(df[url_column].iloc[0]).scheme)(
            anon=True
        )
        # will download concurrently using default max_concurrency=10
        fs.download(
            urls,
            [os.path.join(workdir, URL(url).path.lstrip("/")) for url in urls],
        )


def scene_urls_to_cog(
    urls: list[str],
    workdir: str,
    xfunc: Callable[[xr.DataArray], xr.DataArray],
) -> str:
    """
    Given scene urls, rioxarray open, stack, apply function, write to COG.

    Currently assumes tifs in urls are stackable. Could in the future take a
    corresponding list of dates to stack in time for time-aware interpolation
    techniques for example.

    Parameters
    ----------
    urls : list[str]
        List of urls to download and process.
    workdir : str
        A unique dir to store the temporary COG.
    xfunc : Callable[[xr.DataArray], xr.DataArray]
        The function to apply to the xarray dataset with dims (time,band, y, x).
        We assume to stack over time dimension so make sure to use "time" in
        this callable to reduce over time dimension.
    """
    # read and locally persist the required cogs
    arrays: list[xr.DataArray] = [
        rioxarray.open_rasterio(url, chunks="auto")  # type: ignore # noqa: PGH003
        for url in urls  # type: ignore # noqa: PGH003
    ]
    nodata = arrays[0].rio.nodata
    array = xr.concat(arrays, dim="time").chunk({"band": -1, "x": 512, "y": 512})
    tmp_zarr = Path(workdir) / "locally_persisted.zarr"
    array.to_dataset(name="variables").to_zarr(tmp_zarr, mode="w")
    array = xr.open_zarr(tmp_zarr)["variables"]

    # compute future by reducing over "time" dims e.g. over scenes
    # is likely in float32.
    array = xfunc(array).compute()

    # fill nodata with nan now that were are in float32
    if nodata is not None:
        array = array.where(array != nodata)
    array.rio.write_nodata(np.nan, inplace=True)

    # write out to COG
    array.rio.to_raster(
        str(Path(workdir) / "subset.tif"),
        driver="COG",
        # https://gdal.org/en/stable/drivers/raster/cog.html#creation-options
        BLOCKSIZE=512,
        BIGTIFF="IF_SAFER",
        NUM_THREADS="4",
    )
    return str(Path(workdir) / "subset.tif")


def urls_exists(urls: list[str], bucket: URL, n_workers: int = 16) -> list[bool]:
    """
    Check if many urls exist in the bucket using dask local cluster to parallelize.

    Parameters
    ----------
    urls : list[str]
        List of urls to check.
    bucket : URL
        The bucket to check the urls in.
    n_workers : int, optional
        The number of workers to use, by default 16.

    Returns
    -------
    list[bool]
        List of booleans indicating if the urls exist.
    """
    fs = fsspec.get_filesystem_class(bucket.scheme)()
    exists = []
    with LocalCluster(n_workers=n_workers) as cluster, Client(cluster):
        for chunk in chunked(urls, int(2**12)):
            tasks = [dask.delayed(lambda url: fs.exists(url))(url) for url in chunk]
            exists += list(dask.compute(*tasks))
    return exists
