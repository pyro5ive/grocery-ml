import pandas as pd
import numpy as np

class SchoolFeatures:

    @staticmethod
    def compute_days_until_school_start(dates: pd.Series) -> pd.Series:
        dates = pd.to_datetime(dates)
        start = pd.to_datetime({"year": dates.dt.year, "month": 8, "day": 15})
        start = start.where(dates <= start, start + pd.DateOffset(years=1))
        return (start - dates).dt.days
    ####################################################################

    @staticmethod
    def compute_days_until_school_end(dates: pd.Series) -> pd.Series:
        dates = pd.to_datetime(dates)
        end = pd.to_datetime({"year": dates.dt.year, "month": 5, "day": 31})
        end = end.where(dates <= end, end + pd.DateOffset(years=1))
        return (end - dates).dt.days
    ####################################################################

    @staticmethod
    def compute_school_season_index(dates: pd.Series) -> pd.Series:
        dates = pd.to_datetime(dates)

        start = pd.to_datetime({"year": dates.dt.year, "month": 8, "day": 15})
        end = pd.to_datetime({"year": dates.dt.year, "month": 5, "day": 31})
        season_len = (end - start).dt.days
        idx = pd.Series(0.0, index=dates.index)
        before = dates < start
        during = (dates >= start) & (dates <= end)
        after = dates > end
        idx[before] = -((start[before] - dates[before]).dt.days) / 365.0
        idx[during] = ((dates[during] - start[during]).dt.days) / season_len[during]
        idx[after] = ((dates[after] - end[after]).dt.days) / 365.0

        return idx
    ####################################################################
