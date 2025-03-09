import datetime as dt
import os

URL_PATTERN = "https://glad.umd.edu/dataset/glad_ard2/{lat}/{tile}/{period}.tif"


def period_to_date(period: int) -> dt.datetime:
    year_offset = (period - 392) // 23
    interval_within_year = period - (392 + year_offset * 23)
    return dt.datetime(1997 + year_offset, 1, 1) + dt.timedelta(days=interval_within_year * 16)


def date_to_period(time: dt.datetime) -> int:
    delta = time.replace(tzinfo=None) - dt.datetime(year=time.year, month=1, day=1)
    return (392 + 23 * (time.year - 1997)) + delta.days // 16


def path_to_datetime(path: str) -> dt.datetime:
    return period_to_date(int(os.path.basename(path).strip(".tif")))
