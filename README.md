# Building Large Xarray Mosaics with Flyte + GDAL
This is an example repo to demonstrate:
- UnionAI/Flyte to orchestrate e2e workflows to ingest and build xarrays
- GDAL's GTI with rioxarray to efficiently build large Xarray datasets
- Flyte-Dask plugin for dask clusters at your fingertips

## Up Next
- [ ] Add ingest workflows for downloading GLAD ARD tifs with aria2.
- [ ] Add workflows to build large xarray to zarr
- [ ] Demo multiple approaches of parallelization of ^.

## Nice-to-Haves:
1. add basic github actions for testing, linting, pre-commit, building and pushing container to gchr?

## Requirements:
1. gh is installed + gh auth login has read:packages permissions
2. mamba, conda-lock, docker
3. docker desktop or colima for `make build-push`
4. ensure visibility of ghcr image is public.
5. union auth create --host ... if using a unionai deployment
