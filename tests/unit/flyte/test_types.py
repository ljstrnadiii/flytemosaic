import geopandas as gpd
from flytekit import task

import flyte.types  # noqa: F401 to register the encoding/decoding handlers


def test_geopandas_encodes_decodes():
    @task
    def _gdf_task(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        return gdf

    gdf = gpd.GeoDataFrame(
        {"geometry": gpd.points_from_xy([0, 1], [0, 1]), "other_column": [1, 2]},
        crs="EPSG:4326",
    )
    rt_gdf = _gdf_task(gdf)
    assert rt_gdf.equals(gdf)
