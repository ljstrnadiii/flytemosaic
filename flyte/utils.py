from urllib.parse import urlparse

from flytekit.core.context_manager import FlyteContextManager
from yarl import URL

PROJECT_DIR = "flytemosaic"


def get_default_bucket() -> URL:
    """
    Get a deterministic default bucket for the project.

    Returns
    -------
    URL
        The default bucket for the project. It will be deterministic remotely,
        but random locally.
    """
    ctx = FlyteContextManager.current_context()
    parsed_url = urlparse(ctx.file_access.raw_output_prefix)
    if parsed_url.scheme == "":
        # local case will be random
        return URL(parsed_url.path)
    else:
        return URL(f"{parsed_url.scheme}://{parsed_url.netloc}") / PROJECT_DIR
