import datetime
from typing import Protocol

import geopandas as gpd
from shapely.geometry.base import BaseGeometry
from yarl import URL


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
        Scrapte tifs, build cogs, and upload them to the scene bucket.

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

    def get_required_scenes_gdf(
        self, geo: BaseGeometry, start: datetime.datetime, end: datetime.datetime
    ) -> gpd.GeoDataFrame:
        """
        A helper to get the required scenes for a geo and time range.

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
        ...
