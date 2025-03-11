from enum import Enum

from flytemosaic.datasets.glad import GladARDAnnualMean
from flytemosaic.datasets.protocols import TemporalDatasetProtocol


class DatasetEnum(Enum):
    GLAD_ARD_ANNUAL_MEAN = "glad_ard_annual_mean"


_DATASET_ENUM_TO_PROTOCOL: dict[DatasetEnum, TemporalDatasetProtocol] = {
    DatasetEnum.GLAD_ARD_ANNUAL_MEAN: GladARDAnnualMean(),
}


def get_dataset_protocol(dataset_enum: DatasetEnum) -> TemporalDatasetProtocol:
    if dataset_enum not in _DATASET_ENUM_TO_PROTOCOL:
        raise ValueError(f"Dataset {dataset_enum} not yet supported.")
    return _DATASET_ENUM_TO_PROTOCOL[dataset_enum]
