import pandas as pd
import numpy as np

class SchoolFeatures:

    SCHOOL_START_MONTH = 8
    SCHOOL_START_DAY = 15
    SCHOOL_END_MONTH = 5
    SCHOOL_END_DAY = 31

    @staticmethod
    def _get_school_start(dates: pd.Series) -> pd.Series:
        return pd.to_datetime({"year": dates.dt.year, "month": 8, "day": 15})

    ####################################################################

    @staticmethod
    def _get_school_end(dates: pd.Series) -> pd.Series:
        return pd.to_datetime({"year": dates.dt.year, "month": 5, "day": 31})

    ####################################################################

    @staticmethod
    def compute_days_until_school_start(dates: pd.Series) -> pd.Series:
        dates = pd.to_datetime(dates)
        start = SchoolFeatures._get_school_start(dates)
        start = start.where(dates <= start, start + pd.DateOffset(years=1))
        return (start - dates).dt.days

    ####################################################################

    @staticmethod
    def compute_days_until_school_end(dates: pd.Series) -> pd.Series:
        dates = pd.to_datetime(dates)
        end = SchoolFeatures._get_school_end(dates)
        end = end.where(dates <= end, end + pd.DateOffset(years=1))
        return (end - dates).dt.days

    ####################################################################

    @staticmethod
    def compute_is_school_in_session(dates: pd.Series) -> pd.Series:
        dates = pd.to_datetime(dates)
        start = SchoolFeatures._get_school_start(dates)
        end = SchoolFeatures._get_school_end(dates)

        in_session = (dates >= start) | (dates <= end)
        return in_session.astype(int)

    ####################################################################

    @staticmethod
    def compute_school_cycle_position(dates: pd.Series) -> pd.Series:
        dates = pd.to_datetime(dates)

        start = SchoolFeatures._get_school_start(dates)

        # Align school year start
        start = start.where(dates >= start, start - pd.DateOffset(years=1))

        days_since_start = (dates - start).dt.days
        cycle_length = 365.0

        return days_since_start / cycle_length

    ####################################################################
