import os
import typing
from pathlib import Path

import geopandas as gpd
import pyarrow.lib
from flytekit import FlyteContext
from flytekit.models import literals
from flytekit.types.structured.structured_dataset import (
    StructuredDataset,
    StructuredDatasetDecoder,
    StructuredDatasetEncoder,
    StructuredDatasetMetadata,
    StructuredDatasetType,
)


class GeoPandasEncodingHandler(StructuredDatasetEncoder):
    """Allows us to use GeoPandas dataframes as structured datasets in Flyte."""

    def encode(
        self,
        ctx: FlyteContext,
        structured_dataset: StructuredDataset,
        structured_dataset_type: StructuredDatasetType,
    ) -> literals.StructuredDataset:
        uri = typing.cast(str, structured_dataset.uri) or ctx.file_access.join(
            ctx.file_access.raw_output_prefix, ctx.file_access.get_random_string()
        )
        if not ctx.file_access.is_remote(uri):
            Path(uri).mkdir(parents=True, exist_ok=True)
        path = os.path.join(uri, f"{0:05}.parquet")
        df = typing.cast(gpd.GeoDataFrame, structured_dataset.dataframe)
        df.to_parquet(path)
        structured_dataset_type.format = "parquet"
        return literals.StructuredDataset(
            uri=uri, metadata=StructuredDatasetMetadata(structured_dataset_type)
        )


class GeoPandasDecodingHandler(StructuredDatasetDecoder):
    """Supports decoding of multiple file formats for GeoPandas dataframes."""

    def decode(
        self,
        ctx: FlyteContext,
        flyte_value: literals.StructuredDataset,
        current_task_metadata: StructuredDatasetMetadata,
    ) -> gpd.GeoDataFrame:
        try:
            return gpd.read_parquet(flyte_value.uri)
        except pyarrow.lib.ArrowInvalid:
            return gpd.read_file(flyte_value.uri)
