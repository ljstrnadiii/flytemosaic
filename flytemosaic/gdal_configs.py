def get_worker_config(memory_gb: int, debug: bool = False) -> dict[str, str]:
    """
    Apply some heuristics to determine the optimal GDAL configuration for a worker.

    Parameters
    ----------
    memory_gb : int
        The amount of memory available to the worker in gigabytes.
    debug : bool, optional
        Whether to enable debug mode, by default False.

    Returns
    -------
    dict[str, str]
        A dictionary of GDAL configuration based on the input memory.
    """
    return {
        "GDAL_HTTP_MAX_RETRY": "20",
        "GDAL_HTTP_RETRY_DELAY": "30",
        "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
        "GDAL_HTTP_MULTIPLEX": "YES",
        "GDAL_HTTP_VERSION": "2",
        "GDAL_DISABLE_READDIR_ON_OPEN": "TRUE",
        "CPL_VSIL_CURL_CACHE_SIZE": str(1024**3 * memory_gb * 1 / 3),
        "CPL_VSIL_CURL_CHUNK_SIZE": str(1024**2 * 12),
        "VSI_CACHE": "TRUE",
        "VSI_CACHE_SIZE": str(1024**3 * memory_gb * 1 / 3),
        "GDAL_CACHEMAX": str(1024**3 * memory_gb * 1 / 2),
        "GDAL_NUM_THREADS": "ALL_CPUS",
        "CPL_DEBUG": "ON" if debug else "OFF",
        "CPL_CURL_VERBOSE": "YES" if debug else "NO",
    }
