import datetime
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Protocol

import fsspec
import geopandas as gpd
import xarray as xr
from shapely.geometry.base import BaseGeometry
from yarl import URL

from flytemosaic.datasets.utils import scene_urls_to_cog


@dataclass
class TileDateUrl:
    tile_id: str
    time: datetime.datetime
    url: str
    feature: str


class SceneSourceProtocol(Protocol):
    """
    Protocol to define how to interact with dataset sources.

    Attributes
    ----------
    nodata : float
        The value that represents no data in the dataset.
    max_bytes_per_file : int
        The maximum number of bytes per file.
    """

    def scene_bucket(self, bucket: URL) -> URL:
        return bucket / "scenes"

    @property
    def nodata(self) -> float: ...

    @property
    def max_bytes_per_file(self) -> int: ...

    @property
    def host(self) -> str: ...

    def tile_date_to_scenes(
        self,
        tile_id: str,
        bucket: URL,
        time: datetime.datetime,
        window: datetime.timedelta,
        earliest: datetime.datetime,
        latest: datetime.datetime,
    ) -> list[str]:
        """
        Given a tile_id, time, and window, return the relevant scenes.

        This can be used to determine all the scenes that are relevant to a
        specific tile and time range.
        """
        ...

    def src_url_to_dst_url(self, src_url: str, bucket: URL, relative_dir: str | None = None) -> str:
        """
        Convert a source URL to a destination URL.

        Parameters
        ----------
        src_url : str
            The source URL with a https://glad.umd.edu/ prefix.
        bucket : URL
            The destination bucket URL.
        relative_dir : str, optional
            The relative directory to remove from the source URL, by default
            with use the host.

        Returns
        -------
        str
            The expected destination url given the source url and bucket.
        """
        d = relative_dir or self.host
        return str(self.scene_bucket(bucket) / str(Path(src_url).relative_to(d)))

    def scrape_tifs_and_upload_cogs_batch(
        self,
        gdf: gpd.GeoDataFrame,
        workdir: str,
        bucket: URL,
        user: str | None = None,
        password: str | None = None,
    ) -> gpd.GeoDataFrame:
        """
        Scrape tifs, build cogs, and upload them to the scene bucket.

        Parameters
        ----------
        gdf : gpd.GeoDataFrame
            The GeoDataFrame with the scenes to scrape and must include a "url" column.
        workdir : str
            The working directory to store the temporary files.
        bucket : URL
            The destination bucket URL.
        user : str, optional
            The user to authenticate with the source.
        password : str, optional
            The password to authenticate with the source.

        Returns
        -------
        gpd.GeoDataFrame
            The GeoDataFrame with the scenes that were successfully converted to
            COG and uploaded to the scene bucket. This gdf should match the input
            gdf with url column replaced with urls to the COGs in the scene bucket.
        """
        ...


class TemporalDatasetProtocol(Protocol):
    """
    Protocol to define how to interact with temporal datasets.

    Attributes
    ----------
    name : str
        A unique name of the dataset.
    scene_protocol : SceneSourceProtocol
        The scene source protocol to interact with the scenes.
    earliest : datetime.datetime
        The earliest available file of the dataset.
    latest : datetime.datetime
        The latest available file of the dataset.
    window : datetime.timedelta
        The temporal window of the dataset earlier than a date e.g. a date of
        2021-01-01 with a window of 1 year would have a range of 2020-01-01 to
        2021-01-01. Keeping this simple for now.
    bands : list[str]
        The bands available in each COG.
    dtype : str
        The data type of the final derived feature COGs.
    nodata : str
        The no data value of the final derived feature COGs. Is a string since
        it gets passed to ogr2ogr for gti driver.
    """

    @property
    def name(self) -> str: ...

    @property
    def scene_protocol(self) -> SceneSourceProtocol: ...

    @property
    def earliest(self) -> datetime.datetime: ...

    @property
    def latest(self) -> datetime.datetime: ...

    @property
    def window(self) -> datetime.timedelta: ...

    @property
    def bands(self) -> list[str]: ...

    @property
    def dtype(self) -> str: ...

    @property
    def nodata(self) -> str: ...

    def get_required_scenes_gdf(
        self, geo: BaseGeometry, times: list[datetime.datetime]
    ) -> gpd.GeoDataFrame:
        """
        A helper to get the required scenes for a geo and time range.

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
        ...

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

    def tiles_to_geos(self, tile_ids: list[str]) -> list[BaseGeometry]:
        """
        Compute the geometries (in EPSG:4326) given a tile_id.

        Parameters
        ----------
        tile_ids : list[str]
            The unique identifier of the tile.

        Returns
        -------
        list[BaseGeometry]
            The geometries of each tile in EPSG:4326.
        """
        ...

    def geo_to_tiles(self, geo: BaseGeometry) -> gpd.GeoDataFrame:
        """ """
        ...

    def scenes_to_feature_cog(self, array: xr.DataArray) -> xr.DataArray:
        """
        Given an array with dim "time", apply this to reduce feature over time.

        Parameters
        ----------
        array : xr.DataArray
            The array with dims (time, band, y, x).

        Returns
        -------
        xr.DataArray
            The array with dims (band, y, x).
        """
        ...

    def feature_bucket(self, bucket: URL) -> URL:
        return bucket / "features"

    def get_tile_date_url(self, tile_id: str, time: datetime.datetime, bucket: URL) -> str:
        """
        Given a tile and date, return the expected URL of the COG.

        Parameters
        ----------
        tile_id : str
            A unique identifier for the tile.
        time : datetime.datetime
            The datetime of the derived feature.

        Returns
        -------
        str
            A unique deterministic URL for the derived feature COG.
        """
        return str(
            self.feature_bucket(bucket=bucket) / self.name / f"{tile_id}" / f"{time:%Y%m%d}.tif"
        )

    def build_tile_date_cog(
        self,
        tile_id: str,
        time: datetime.datetime,
        bucket: URL,
        workdir: str,
    ) -> TileDateUrl:
        """
        Given a tile and date, assume scenes have been ingested and build a COG.

        For example, lookup all the scenes it would take to gap fill, interpolate,
        apply summary stats, or any other function to apply over many scenes to
        create a temporal composite.

        Parameters
        ----------
        tile_id : str
            A unique identifier for the tile.
        time : datetime.datetime
            The datetime of the derived feature.
        bucket : URL
            The destination bucket URL.
        workdir : str
            The working directory to store the temporary files.

        Returns
        -------
        str
            The URL to the COG stored somewhere based no
        """
        ...
        url = self.get_tile_date_url(tile_id=tile_id, time=time, bucket=bucket)
        fs = fsspec.get_filesystem_class(bucket.scheme)()
        if not fs.exists(url):
            required_cogs = self.scene_protocol.tile_date_to_scenes(
                tile_id=tile_id,
                bucket=bucket,
                time=time,
                window=self.window,
                earliest=self.earliest,
                latest=self.latest,
            )
            # TODO: optionally require all scenes exist?
            derived_cog = scene_urls_to_cog(
                urls=required_cogs,
                workdir=workdir,
                xfunc=self.scenes_to_feature_cog,
            )
            fs.put_file(derived_cog, url)
        return TileDateUrl(tile_id=tile_id, time=time, url=url, feature=self.name)

    def get_tile_date_urls(
        self, geo: BaseGeometry, times: list[datetime.datetime], bucket: URL
    ) -> list[TileDateUrl]:
        relevant_tiles = self.geo_to_tiles(geo=geo)["tile_id"].unique().tolist()
        return [
            TileDateUrl(
                feature=self.name,
                tile_id=tile_id,
                time=t,
                url=self.get_tile_date_url(tile_id=tile_id, time=t, bucket=bucket),
            )
            for tile_id, t in product(
                relevant_tiles,
                {self.snap_to_temporal_grid(time=t) for t in times},
            )
        ]
