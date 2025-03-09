CONTAINER_HOST ?= ghcr.io/ljstrnadiii
PACKAGE_NAME ?= flytemosaic
FLYTE_PROJECT ?= flyteexamples

# simple versioning scheme
DATE_TAG := $(shell date +%Y%m%d-%H%M%S)

lock:
	conda-lock --mamba  -f environment.yml -p linux-64 -p osx-64 -p osx-arm64

install:
	@if conda env list | grep -q $(PACKAGE_NAME); then \
		echo "🔄 Updating Conda environment: $(PACKAGE_NAME)"; \
		mamba env update --name $(PACKAGE_NAME) --file conda-lock.yml --prune; \
		mamba run --no-capture-output --name $(PACKAGE_NAME) pip install -e . --no-deps; \
	else \
		echo "🚀 Creating new Conda environment: $(PACKAGE_NAME)"; \
		conda-lock install --mamba --name $(PACKAGE_NAME) $(LOCK_FILE); \
		mamba run --no-capture-output --name $(PACKAGE_NAME) pip install -e . --no-deps; \
	fi
	@echo "✅ Conda environment '$(PACKAGE_NAME)' is ready!"

build-push:
	@echo "🚀 Building Docker image..."
	DOCKER_BUILDKIT=1 docker build -f Dockerfile -t $(CONTAINER_HOST)/$(PACKAGE_NAME):$(DATE_TAG) .
	@echo "✅ Docker image built with tag $(CONTAINER_HOST)/$(PACKAGE_NAME):$(DATE_TAG)"
	@echo "🚀 Pushing Docker image to GHCR..."
	docker push $(CONTAINER_HOST)/$(PACKAGE_NAME):$(DATE_TAG)
	@echo "✅ Docker image pushed to $(CONTAINER_HOST)/$(PACKAGE_NAME):$(DATE_TAG)"

latest-image:
	@echo "🚀 Fetching latest Docker image tag..."
	@LATEST_TAG=$$(gh api "/user/packages/container/$(PACKAGE_NAME)/versions" --jq '.[0].metadata.container.tags[0]') && \
	echo "✅ Latest Docker image tag: $(CONTAINER_HOST)/$(PACKAGE_NAME):$${LATEST_TAG}"

fast-register:
	@echo "🚀 Fast registering tasks and workflows with most recent image..."
	@LATEST_TAG=$$(gh api "/user/packages/container/$(PACKAGE_NAME)/versions" --jq '.[0].metadata.container.tags[0]') && \
	LATEST_IMAGE=$(CONTAINER_HOST)/$(PACKAGE_NAME):$$LATEST_TAG && \
	mamba run --no-capture-output --name $(PACKAGE_NAME) \
	    union register \
	    --project $(FLYTE_PROJECT) \
	    --domain development \
	    --image $$LATEST_IMAGE \
	    flyte

register: build-push
	@echo "🚀 Registering tasks and workflows with --copy none..."
	@LATEST_TAG=$$(gh api "/user/packages/container/$(PACKAGE_NAME)/versions" --jq '.[0].metadata.container.tags[0]') && \
	LATEST_IMAGE=$(CONTAINER_HOST)/$(PACKAGE_NAME):$$LATEST_TAG && \
	mamba run --no-capture-output --name $(PACKAGE_NAME) \
	    union register \
	    --project $(FLYTE_PROJECT) \
	    --domain development \
	    --image $$LATEST_IMAGE \
	    --copy none  \
		--version $(DATE_TAG) \
	    flyte