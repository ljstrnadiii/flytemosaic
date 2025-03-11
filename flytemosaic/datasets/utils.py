import os
import subprocess

import geopandas as gpd
import pandas as pd
from yarl import URL


def download_files_with_aria(
    df: pd.DataFrame | gpd.GeoDataFrame,
    url_column: str,
    workdir: str,
    user: str | None = None,
    password: str | None = None,
) -> None:
    """
    Helper to download files in a dataframe with a url column using aria2c.

    Parameters
    ----------
    df : pd.DataFrame | gpd.GeoDataFrame
        The DataFrame with the urls to download. Must have 'url' column.
    url_column : str
        The name of the column with the urls.
    workdir : str
        The directory to download the files.
    user : str, optional
        The username for the website, by default None. If not provided, user should
        have .netrc file with credentials.
    password : str, optional
        The password for the website, by default None. If not provided, user should
        have .netrc file with credentials.
    """
    urls = df[url_column].tolist()
    url_list_path = os.path.join(workdir, "urls.txt")
    with open(url_list_path, "w") as f:
        for url in urls:
            f.write(url + "\n")
            # match the url dir structure by adding the out parameter option
            # https://aria2.github.io/manual/en/html/aria2c.html#id2
            f.write("\t" + "out=" + URL(url).path.lstrip("/") + "\n")

    subprocess.run(
        [
            "aria2c",
            "--input-file=" + url_list_path,
            "--dir=" + workdir,
            "--max-concurrent-downloads=1",
        ]
        + (["--http-user=" + user, "--http-passwd=" + password] if user and password else []),
        check=True,
    )
