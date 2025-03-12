import datetime
import datetime as dt
import os
import tempfile
from functools import lru_cache
from itertools import product
from pathlib import Path

import fsspec
import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray
import xarray as xr
from fsspec.implementations.local import LocalFileSystem
from shapely.geometry.base import BaseGeometry
from yarl import URL

from flytemosaic.datasets.protocols import (
    SceneSourceProtocol,
    TemporalDatasetProtocol,
    TileDate,
    TileDateUrl,
)
from flytemosaic.datasets.utils import (
    download_files_with_fsspec,
)

# Server recently went down
# HOST = "https://glad.umd.edu/"
# URL_PATTERN = "https://glad.umd.edu/dataset/glad_ard2/{lat}/{tile}/{period}.tif"

# 2020-2024 available on s3
HOST = "s3://glad.landsat.ard/"
URL_PATTERN = "s3://glad.landsat.ard/data/tiles/{lat}/{tile}/{period}.tif"


def _period_to_datetime(period: int) -> dt.datetime:
    year_offset = (period - 392) // 23
    interval_within_year = period - (392 + year_offset * 23)
    return dt.datetime(1997 + year_offset, 1, 1) + dt.timedelta(days=interval_within_year * 16)


def _datetime_to_period(time: dt.datetime) -> int:
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

    def src_url_to_dst_url(self, src_url: str, bucket: URL, relative_dir: str = HOST) -> str:
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

        return str(self.scene_bucket(bucket) / str(Path(src_url).relative_to(relative_dir)))

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
        download_files_with_fsspec(
            df=gdf,
            url_column="url",  # use location from the beginning?
            workdir=workdir,
            # user=user,
            # password=password,
        )
        # upload batch of files in workdir to scene bucket
        fs = fsspec.get_filesystem_class(bucket.scheme)()
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
            dst_url = self.src_url_to_dst_url(str(tif_path), bucket, workdir)
            if isinstance(fs, LocalFileSystem):
                Path(dst_url).parent.mkdir(parents=True, exist_ok=True)
            fs.put_file(tif_path, dst_url)

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

    @property
    def dtype(self) -> str:
        return "float32"

    @property
    def nodata(self) -> str:
        return "nan"

    def get_required_scenes_gdf(
        self, geo: BaseGeometry, times: list[datetime.datetime]
    ) -> gpd.GeoDataFrame:
        """
        Get the required scene gdf for a geo and time range.

        Parameters
        ----------
        geo : BaseGeometry
            The geometry to use to determine which tiles to scrape.
        times : list[datetime.datetime]
            The times to use to determine which tiles to scrape.

        Returns
        -------
        gpd.GeoDataFrame
            The GeoDataFrame with the required scenes for the geo and time range
            with columns "datetime", "url", and "geometry".
        """
        scene_gdf = _glad_tile_gdf()
        relevant_tiles = scene_gdf.loc[scene_gdf.intersects(geo), :]
        periods = set()
        for t in times:
            start_period = _datetime_to_period(max(t - self.window, self.earliest))
            end_period = _datetime_to_period(min(t, self.latest))
            periods.update(set(range(start_period, end_period + 1)))
        gdfs: gpd.GeoDataFrame = []
        for period in periods:
            gdfs.append(_add_scene_url(period, relevant_tiles))
        return pd.concat(gdfs)[["datetime", "url", "geometry"]].reset_index(drop=True)

    def snap_to_temporal_grid(self, time: datetime.datetime) -> datetime.datetime:
        """
        Snap a time to the temporal grid of the dataset.

        Parameters
        ----------
        time : datetime.datetime
            The time to snap to the temporal grid.

        Returns
        -------
        datetime.datetime
            The time quantized or snapped to the underlying datasets temporal grid.
        """
        return datetime.datetime(time.year, 1, 1)

    def _get_required_scene_cogs(
        self, tile_id: str, time: datetime.datetime, bucket: URL
    ) -> list[str]:
        start_period = _datetime_to_period(max(time - self.window, self.earliest))
        end_period = _datetime_to_period(time=time)
        return [
            self.scene_protocol.src_url_to_dst_url(
                str(
                    Path(
                        URL_PATTERN.format(lat=tile_id.split("_")[-1], tile=tile_id, period=period)
                    )
                ),
                bucket,
            )
            for period in range(start_period, end_period + 1)
        ]

    def _get_tiles(self, geo: BaseGeometry) -> list[str]:
        """
        Get the tiles that intersect with the geo using glad ard native tiling.F

        Parameters
        ----------
        geo : BaseGeometry
            The geometry to use to determine which tiles to scrape.

        Returns
        -------
        list[str]

        """
        # we could have used any tiling scheme here .e.g. morecantile
        tiling_gdf = _glad_tile_gdf()
        return tiling_gdf.loc[tiling_gdf.intersects(geo), "TILE"].drop_duplicates().tolist()

    def get_tile_dates(self, geo: BaseGeometry, times: list[datetime.datetime]) -> list[TileDate]:
        return [
            TileDate(tile_id=tile_id, time=t)
            for tile_id, t in product(
                self._get_tiles(geo),
                {self.snap_to_temporal_grid(time=t) for t in times},
            )
        ]

    def get_tile_date_url(self, tile_id: str, time: datetime.datetime, bucket: URL) -> str:
        # creates a deterministic URL for the derived feature COG
        return str(
            self.feature_bucket(bucket=bucket)
            / f"{tile_id.split('_')[-1]}"
            / f"{tile_id}"
            / f"{time:%Y%m%d}.tif"
        )

    def build_tile_date_cog(
        self,
        tile_id: str,
        time: datetime.datetime,
        bucket: URL,
        workdir: str,
    ) -> TileDateUrl:
        url = self.get_tile_date_url(tile_id=tile_id, time=time, bucket=bucket)
        fs = fsspec.get_filesystem_class(bucket.scheme)()
        if not fs.exists(url):
            required_cogs = self._get_required_scene_cogs(tile_id=tile_id, time=time, bucket=bucket)

            # read and locally persist the required cogs
            arrays: list[xr.DataArray] = [
                rioxarray.open_rasterio(url, chunks="auto")
                for url in required_cogs  # type: ignore # noqa: PGH003
            ]
            array = xr.concat(arrays, dim="time").chunk({"band": -1, "x": 512, "y": 512})
            tmp_zarr = Path(workdir) / "locally_persisted.zarr"
            array.to_dataset(name="variables").to_zarr(tmp_zarr, mode="w")
            array = xr.open_zarr(tmp_zarr)["variables"]

            # simplest of quality flags
            array = array.where(array.sel(band=8) == 1)
            # temporal composite using mean (nan tolerant!)
            array = array.mean(dim="time").astype("float32")
            # fill nodata with nan now that were are in float32
            array = array.where(array != self.scene_protocol.nodata)
            array.rio.write_nodata(np.nan, inplace=True)
            # mean over quality flag is a bit meaningless
            subset = array.isel(band=slice(0, 7)).compute()
            with tempfile.NamedTemporaryFile(suffix=".tif") as f:
                subset.rio.to_raster(
                    f.name,
                    driver="COG",
                    # https://gdal.org/en/stable/drivers/raster/cog.html#creation-options
                    BLOCKSIZE=512,
                    BIGTIFF="IF_SAFER",
                    NUM_THREADS="4",
                )
                fs.put(f.name, url)
        return TileDateUrl(tile_id=tile_id, time=time, url=url)

    def get_geo_from_tile_ids(self, tile_ids: list[str]) -> list[BaseGeometry]:
        glad_tile_index = _glad_tile_gdf().set_index("TILE")
        return glad_tile_index.loc[tile_ids, "geometry"].tolist()
