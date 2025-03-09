import geopandas as gpd
from flytekit.types.structured.structured_dataset import (
    PARQUET,
    StructuredDatasetTransformerEngine,
)

from flyte.types import GeoPandasDecodingHandler, GeoPandasEncodingHandler

StructuredDatasetTransformerEngine.register(
    GeoPandasEncodingHandler(gpd.GeoDataFrame, None, PARQUET)
)
StructuredDatasetTransformerEngine.register(GeoPandasDecodingHandler(gpd.GeoDataFrame))
