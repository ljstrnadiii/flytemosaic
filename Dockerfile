FROM --platform=linux/amd64 mambaorg/micromamba:1.5.3

ENV ENV_NAME=flytemosaic
WORKDIR /flytemosaic

# Copy dependency files first (for better caching)
COPY conda-lock.yml pyproject.toml ./

# Install dependencies with caching for Conda packages
RUN --mount=type=cache,target=/opt/conda/pkgs \
    micromamba create -y -n $ENV_NAME --file conda-lock.yml && \
    micromamba clean --index-cache --yes

# Run code in the conda environment
ARG MAMBA_DOCKERFILE_ACTIVATE=1

# Copy project source code
COPY flytemosaic ./flytemosaic/
COPY flyte ./flyte/

# Install Python package in editable mode
RUN micromamba run -n $ENV_NAME pip install -e . --no-deps


# remove once plugins get published on next flytekit release
USER root
RUN apt-get update && apt-get install -y git
USER mambauser
RUN micromamba run -n $ENV_NAME pip install git+https://github.com/ljstrnadiii/flytekit.git@add_xarray_support#subdirectory=plugins/flytekit-xarray-zarr
RUN micromamba run -n $ENV_NAME pip install git+https://github.com/ljstrnadiii/flytekit.git@fix_geopandas_plugin#subdirectory=plugins/flytekit-geopandas

# Temp fix: Change user to root since fast registering copies into /root
# (Consider specifying `--destination-dir` in `register` instead)
USER root
