import datetime
from dataclasses import dataclass
from typing import Protocol

import geopandas as gpd
from shapely.geometry.base import BaseGeometry
from yarl import URL


@dataclass
class TileDate:
    tile_id: str
    time: datetime.datetime


@dataclass
class TileDateUrl(TileDate):
    url: str


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

    def src_url_to_dst_url(self, src_url: str, bucket: URL) -> str: ...

    def scrape_tifs_uploads_cogs_batch(
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

    def feature_bucket(self, bucket: URL) -> URL:
        return bucket / "features"

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
        ...

    def get_tile_dates(self, geo: BaseGeometry, times: list[datetime.datetime]) -> list[TileDate]:
        """
        ...
        """
        ...

    def build_tile_date_cog(
        self, tile_id: str, time: datetime.datetime, bucket: URL, workdir: str
    ) -> TileDateUrl:
        """
        Given a tile and date, assume scenes have been ingested and build a COG.

        For example, lookup all the scenes it would take to gap fill, interpolate,
        apply summary stats, or any other function to apply over many scenes to
        create a temporal composite.

        Returns
        -------
        str
            The URL to the COG stored somewhere based no
        """
        ...

    def get_geo_from_tile_ids(self, tile_ids: list[str]) -> list[BaseGeometry]:
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
