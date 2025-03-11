import datetime
import datetime as dt
import os
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import fsspec
import geopandas as gpd
import pandas as pd
import rioxarray
from fsspec.implementations.local import LocalFileSystem
from shapely.geometry.base import BaseGeometry
from yarl import URL

from flytemosaic.datasets.protocols import SceneSourceProtocol, TemporalDatasetProtocol
from flytemosaic.datasets.utils import download_files_with_aria

if TYPE_CHECKING:
    import xarray as xr

HOST = "https://glad.umd.edu/"
URL_PATTERN = "https://glad.umd.edu/dataset/glad_ard2/{lat}/{tile}/{period}.tif"


def _period_to_datetime(period: int) -> dt.datetime:
    year_offset = (period - 392) // 23
    interval_within_year = period - (392 + year_offset * 23)
    return dt.datetime(1997 + year_offset, 1, 1) + dt.timedelta(days=interval_within_year * 16)


def _date_to_period(time: dt.datetime) -> int:
    delta = time.replace(tzinfo=None) - dt.datetime(year=time.year, month=1, day=1)
    return (392 + 23 * (time.year - 1997)) + delta.days // 16


@lru_cache(maxsize=1)
def _glad_tile_gdf() -> gpd.GeoDataFrame:
    return gpd.read_parquet(
        os.path.join(
            os.path.dirname(Path(__file__).parent),
            "data",
            "glad_tiling.parquet",
        )
    )


def _add_scene_url(period: int, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    lats = gdf["TILE"].str.split("_").str[-1]
    gdf["datetime"] = _period_to_datetime(period)
    gdf["url"] = [
        URL_PATTERN.format(lat=lat, tile=tile, period=period)
        for lat, tile in zip(lats, gdf["TILE"], strict=True)
    ]
    return gdf


class GladARDSceneSource(SceneSourceProtocol):
    """
    Implements SceneSourceProtocol for GLAD ARD datasets.

    This class implements the scene-specific methods and attributes to support
    scraping scenes for any datasets that depend on this scene source.

    Attributes
    ----------
    nodata : int
        The no data value for the dataset, which is 0 for glad ards uin16 tifs.
    max_bytes_per_file : int
        The maximum number of bytes per file for the dataset. Since they are
        all the same size, dtype, and number of bands, this is a constant.
    """

    @property
    def nodata(self) -> int:
        return 0

    @property
    def max_bytes_per_file(self) -> int:
        # 8 bands, 4004x4004 pixels, 2 bytes per pixel since uint16
        return 8 * 4004**2 * 2

    def src_url_to_dst_url(self, src_url: str, bucket: URL) -> str:
        """
        Convert a source URL to a destination URL.

        Parameters
        ----------
        src_url : str
            The source URL with a https://glad.umd.edu/ prefix.
        bucket : URL
            The destination bucket URL.

        Returns
        -------
        str
            The expected destination url given the source url and bucket.
        """
        return str(self.scene_bucket(bucket) / str(Path(src_url).relative_to(HOST)))

    def scrape_tifs_uploads_cogs_batch(
        self,
        gdf: gpd.GeoDataFrame,
        workdir: str,
        bucket: URL,
        user: str | None = None,
        password: str | None = None,
    ) -> gpd.GeoDataFrame:
        """
        Download urls in gdf to workdir, create cogs, upload to scene bucket.

        Parameters
        ----------
        gdf : gpd.GeoDataFrame
            The GeoDataFrame with a "url" column with source https://glad.umd.edu/
            elements with a small batching keeping in mind this is done in serial
            and likely being run multiple times in different tasks.
        workdir : str
            A temporary directory to download and process the files.
        bucket : URL
            The destination bucket URL.
        user : str, optional
            The username for the glad.umd.edu website, by default None. If not
            provided, user should have .netrc file with credentials.
        password : str, optional
            The password for the glad.umd.edu website, by default None. If not
            provided, user should have .netrc file with credentials.

        Returns
        -------
        gpd.GeoDataFrame
            The GeoDataFrame with the "url" column updated to the destination of
            the cogs in the scene bucket.
        """
        # download batch of files in gdf['url'] to workdir
        download_files_with_aria(
            df=gdf,
            url_column="url",  # use location from the beginning?
            workdir=workdir,
            user=user,
            password=password,
        )
        # upload batch of files in workdir to scene bucket
        dst_bucket = self.scene_bucket(bucket)
        fs = fsspec.get_filesystem_class(dst_bucket.scheme)()
        downloaded_tifs = list(Path(workdir).glob("**/*.tif"))
        for tif_path in downloaded_tifs:
            data: xr.DataArray = rioxarray.open_rasterio(tif_path)  # type: ignore  # noqa: PGH003
            if not isinstance(data.rio, rioxarray.raster_array.RasterArray):
                raise TypeError("type narrowing trick for dynamic type checking")
            data.rio.write_nodata(self.nodata, inplace=True)
            data.rio.to_raster(
                tif_path,
                driver="COG",
                # https://gdal.org/en/stable/drivers/raster/cog.html#creation-options
                BLOCKSIZE=512,
                BIGTIFF="IF_SAFER",
                NUM_THREADS="4",
            )
            dst_url = str(dst_bucket / str(tif_path.relative_to(workdir)))
            if isinstance(fs, LocalFileSystem):
                Path(dst_url).parent.mkdir(parents=True, exist_ok=True)
            fs.put(tif_path, dst_url)

        # use relative url temporarily to match with downloaded_tifs
        relative_urls = [str(tif.relative_to(workdir)) for tif in downloaded_tifs]
        subset = gdf.loc[
            (gdf["url"].apply(lambda x: str(Path(x).relative_to(HOST))).isin(relative_urls)),
            :,
        ]
        # Now point url to assets on the scene bucket
        subset["url"] = subset["url"].apply(lambda x: self.src_url_to_dst_url(x, bucket))
        return subset


class GladARDAnnualMean(TemporalDatasetProtocol):
    """
    Implementation of the TemporalDatasetProtocol for the GLAD annual median dataset.

    This dataset requires we download all the necessary scenes for a given
    geo and date, then calculate all the bands' annual median values and store
    to a deterministic location.

    Attributes
    ----------
    scene_protocol : SceneSourceProtocol
        The scene source protocol for the dataset.
    """

    @property
    def name(self) -> str:
        return "glad_annual_mean"

    @property
    def scene_protocol(self) -> SceneSourceProtocol:
        return GladARDSceneSource()

    @property
    def earliest(self) -> datetime.datetime:
        return datetime.datetime(1997, 1, 1)

    @property
    def latest(self) -> datetime.datetime:
        return datetime.datetime.now() - datetime.timedelta(days=14)  # ?

    @property
    def window(self) -> datetime.timedelta:
        return datetime.timedelta(days=365)

    @property
    def bands(self) -> list[str]:
        return [f"{self.name}:{b}" for b in ["B1", "B2", "B3", "B4", "B5", "B6", "B7"]]

    def get_required_scenes_gdf(
        self, geo: BaseGeometry, start: datetime.datetime, end: datetime.datetime
    ) -> gpd.GeoDataFrame:
        """
        Get the required scene gdf for a geo and time range.

        Parameters
        ----------
        geo : BaseGeometry
            The geometry to use to determine which tiles to scrape.
        start : datetime.datetime
            The start of the time range to be used with window.
        end : datetime.datetime
            The end of the time range to be used with window.

        Returns
        -------
        gpd.GeoDataFrame
            The GeoDataFrame with the required scenes for the geo and time range
            with columns "datetime", "url", and "geometry".
        """
        scene_gdf = _glad_tile_gdf()
        relevant_tiles = scene_gdf.loc[scene_gdf.intersects(geo), :]
        start_period = _date_to_period(max(start - self.window, self.earliest))
        end_period = _date_to_period(min(end, self.latest))
        gdfs: gpd.GeoDataFrame = []
        for period in list(range(start_period, end_period + 1)):
            gdfs.append(_add_scene_url(period, relevant_tiles))
        return pd.concat(gdfs)[["datetime", "url", "geometry"]].reset_index(drop=True)
