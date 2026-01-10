import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
import numpy as np

class HolidayFeatures:

    @staticmethod
    def compute_days_until_next_holiday(dates: pd.Series) -> pd.Series:
        dates = pd.to_datetime(dates)
        start = dates.min().normalize()
        end = dates.max().normalize() + pd.Timedelta(days=366)
        holidays = USFederalHolidayCalendar().holidays(start=start, end=end)
        next_idx = holidays.searchsorted(dates, side="left")
        next_holidays = pd.Series(holidays[next_idx], index=dates.index)
        return (next_holidays - dates).dt.days
    #####################################################################

    @staticmethod
    def compute_days_since_last_holiday(dates: pd.Series) -> pd.Series:
        dates = pd.to_datetime(dates)
        start = dates.min().normalize() - pd.Timedelta(days=366)
        end = dates.max().normalize()
        holidays = USFederalHolidayCalendar().holidays(start=start, end=end)
        prev_idx = holidays.searchsorted(dates, side="right") - 1
        prev_idx = np.clip(prev_idx, 0, None)
        prev_holidays = pd.Series(holidays[prev_idx], index=dates.index)
        return (dates - prev_holidays).dt.days
    #####################################################################

    @staticmethod
    def compute_holiday_proximity_index(dates: pd.Series, scale: int = 30) -> pd.Series:
        dates = pd.to_datetime(dates)
        before = HolidayFeatures.compute_days_until_next_holiday(dates)
        after = HolidayFeatures.compute_days_since_last_holiday(dates)
        proximity = pd.Series(0.0, index=dates.index)
        before_mask = before <= after
        after_mask = after < before
        proximity.loc[before_mask] = ((scale - before.loc[before_mask]) / scale).clip(lower=0)
        proximity.loc[after_mask] = -((scale - after.loc[after_mask]) / scale).clip(lower=0)

        return proximity
    ###################################################################
