import dask.array as da
import numpy as np
import pytest
import xarray as xr

from flytemosaic.mosaics import build_mosaic_chunk_partitions


@pytest.mark.parametrize(
    ("target_chunks", "actual_chunks", "bands"),
    [
        (800, 800, ["a"]),
        (800, 800, ["a", "b"]),
        (800, 800, ["b", "c"]),
        (800 * 4 * 2 - 1, 800, ["a", "b"]),
        (800 * 4 * 2, 3200, ["a"]),
        (800 * 4 * 2, 3200, ["a", "b"]),
        (800 * 4 * 3, 3200, ["a", "b", "c"]),
    ],
)
def test_build_mosaic_chunk_partitions_expected_sizes(
    target_chunks: int,
    actual_chunks: int,
    bands: list[str],
):
    dset = xr.Dataset(
        {
            "variables": xr.DataArray(
                # chunk nbytes from (1,1,10,10) are tightly coupled with the
                # target chunks and actual chunk parameters
                da.random.random((3, 5, 100, 100), chunks=(1, 1, 10, 10)),
                dims=["band", "time", "y", "x"],
            )
        },
        coords={"band": ["a", "b", "c"]},
    )
    for slcs in build_mosaic_chunk_partitions(
        dset, target_chunks, variable_name="variables", bands=bands
    ):
        subset = dset["variables"].isel(**{k: slice(*v) for k, v in slcs.items()})
        assert set(subset["band"].data) == set(bands)
        assert subset.nbytes == actual_chunks * len(bands)


@pytest.mark.parametrize("nbytes", [200, 800, 1600, 3200])
def test_build_mosaic_chunk_partitions_fully_partitions(nbytes: int):
    dset = xr.Dataset(
        {
            "variables": xr.DataArray(
                da.random.random((3, 5, 102, 103), chunks=(1, 5, 50, 50)),
                dims=["band", "time", "y", "x"],
            )
        },
        coords={"band": ["a", "b", "c"]},
    )
    target = np.zeros_like(dset["variables"])
    for band_set in [["a"], ["b", "c"]]:
        for slcs in build_mosaic_chunk_partitions(
            dset, nbytes, variable_name="variables", bands=band_set
        ):
            prev = target[
                slcs["band"][0] : slcs["band"][1],
                slcs["time"][0] : slcs["time"][1],
                slcs["y"][0] : slcs["y"][1],
                slcs["x"][0] : slcs["x"][1],
            ]
            prev += 1
    assert target.sum() == np.prod(dset["variables"].size)


def test_build_mosaic_chunk_partitions_raises_not_contiguous():
    dset = xr.Dataset(
        {
            "variables": xr.DataArray(
                da.random.random((3, 5, 100, 100), chunks=(1, 1, 10, 10)),
                dims=["band", "time", "y", "x"],
            )
        },
        coords={"band": ["a", "b", "c"]},
    )
    with pytest.raises(ValueError, match=r"Band indices are not contiguous"):
        for _ in build_mosaic_chunk_partitions(
            dset, 800, variable_name="variables", bands=["a", "c"]
        ):
            pass
