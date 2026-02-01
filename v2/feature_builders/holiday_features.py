import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
import numpy as np
import logging

logger = logging.getLogger(__name__)

class HolidayFeatures:

    extraHolidays = {
        "MardiGras": [
            "2024-02-13",
            "2025-03-04",
            "2026-02-17",
            "2027-02-09",
            "2028-02-29",
            "2029-02-13",
            "2030-03-05"
        ]
    }

    @staticmethod
    def compute_days_until_next_holiday(dates: pd.Series) -> pd.Series:
        colName = "daysUntilNextHoliday_feat"

        dates = pd.to_datetime(dates).dt.normalize()
        start = dates.min()
        end = dates.max() + pd.Timedelta(days=366)

        holidays = USFederalHolidayCalendar().holidays(start=start, end=end)

        next_idx = holidays.searchsorted(dates, side="left")
        next_holidays = pd.Series(holidays[next_idx], index=dates.index)

        series = (next_holidays - dates).dt.days
        series.name = colName
        return series
    #####################################################################

    @staticmethod
    def compute_days_since_last_holiday(dates: pd.Series) -> pd.Series:
        colName = "daysSinceLastHoliday_feat"

        dates = pd.to_datetime(dates).dt.normalize()
        start = dates.min() - pd.Timedelta(days=366)
        end = dates.max()

        holidays = USFederalHolidayCalendar().holidays(start=start, end=end)

        prev_idx = holidays.searchsorted(dates, side="right") - 1
        prev_idx = np.clip(prev_idx, 0, None)
        prev_holidays = pd.Series(holidays[prev_idx], index=dates.index)

        series = (dates - prev_holidays).dt.days
        series.name = colName
        return series
    #####################################################################

    @staticmethod
    def compute_holiday_proximity_index(dates: pd.Series, scale: int = 30) -> pd.Series:
        colName = "holidayProximityIndex_feat"

        dates = pd.to_datetime(dates).dt.normalize()
        before = HolidayFeatures.compute_days_until_next_holiday(dates)
        after = HolidayFeatures.compute_days_since_last_holiday(dates)

        proximity = pd.Series(0.0, index=dates.index)

        before_mask = before <= after
        after_mask = after < before

        proximity.loc[before_mask] = ((scale - before.loc[before_mask]) / scale).clip(lower=0)
        proximity.loc[after_mask] = -((scale - after.loc[after_mask]) / scale).clip(lower=0)

        proximity.name = colName
        return proximity
    #####################################################################

    @staticmethod
    def build_federal_holiday_flag_and_proximity_features(dates: pd.Series, scale: int = 15) -> pd.DataFrame:
        dates = pd.to_datetime(dates).dt.normalize()
        calendar = USFederalHolidayCalendar()
        rules = calendar.rules

        result_df = pd.DataFrame(index=dates.index)

        start = dates.min() - pd.Timedelta(days=366)
        end = dates.max() + pd.Timedelta(days=366)

        for rule in rules:
            holiday_dates = rule.dates(start, end).normalize()

            clean_name = rule.name.replace(" ", "").replace("'", "")
            is_col = f"is{clean_name}_holiday_feat"
            prox_col = f"proximity_{clean_name}_holiday_feat"

            result_df[is_col] = dates.isin(holiday_dates).astype("int8")
            result_df[prox_col] = HolidayFeatures._compute_proximity_to_dates(dates, holiday_dates, scale)

        return result_df
    #####################################################################

    @staticmethod
    def build_extra_holiday_flag_and_proximity_features(dates: pd.Series, holidayName: str, scale: int = 30) -> pd.DataFrame:
        dates = pd.to_datetime(dates).dt.normalize()
        holiday_dates = pd.to_datetime(
            HolidayFeatures.extraHolidays.get(holidayName, [])
        ).normalize()

        is_col = f"is{holidayName}_holiday_feat"
        prox_col = f"proximity_{holidayName}_holiday_feat"

        result_df = pd.DataFrame(index=dates.index)
        result_df[is_col] = dates.isin(holiday_dates).astype("int8")
        result_df[prox_col] = HolidayFeatures._compute_proximity_to_dates(dates, holiday_dates, scale)

        return result_df
    #####################################################################

    @staticmethod
    def _compute_proximity_to_dates(dates: pd.Series, holiday_dates: pd.DatetimeIndex, scale: int) -> pd.Series:
        next_idx = holiday_dates.searchsorted(dates, side="left")
        prev_idx = np.clip(next_idx - 1, 0, None)

        next_holidays = pd.Series(holiday_dates[next_idx], index=dates.index)
        prev_holidays = pd.Series(holiday_dates[prev_idx], index=dates.index)

        before = (next_holidays - dates).dt.days
        after = (dates - prev_holidays).dt.days

        proximity = pd.Series(0.0, index=dates.index)

        before_mask = before <= after
        after_mask = after < before

        proximity.loc[before_mask] = ((scale - before.loc[before_mask]) / scale).clip(lower=0)
        proximity.loc[after_mask] = -((scale - after.loc[after_mask]) / scale).clip(lower=0)

        return proximity
    #####################################################################
