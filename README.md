# Building Large Xarray Mosaics with Flyte + GDAL
Accompanying slides can be found [here](https://docs.google.com/presentation/d/1kGUqyaFuJjAu9upxVZpFDRlxjcI27u2EJpHvfjQiOYQ/edit?usp=sharing).

This project demonstrates:
- UnionAI/Flyte to orchestrate e2e workflows to ingest and build xarrays
- GDAL's GTI with rioxarray to efficiently build large Xarray datasets
- Flyte-Dask plugin for dask clusters at your fingertips


## Entrypoint
Not sure where to start? Look to the `flyte/build.py:build_dataset_mosaic_workflow`
workflow for the main entrypoint defining a workflow to build mosaics.

## Nice-to-Haves:
1. add basic github actions for testing, linting, pre-commit, building and pushing container to gchr?

## Requirements:
1. gh is installed + gh auth login has read:packages permissions
2. mamba, conda-lock, docker
3. docker desktop or colima for `make build-push`
4. ensure visibility of ghcr image is public.
5. union auth create --host ... if using a unionai deployment
