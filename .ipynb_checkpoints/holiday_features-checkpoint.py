import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
import numpy as np

class HolidayFeatures:

    @staticmethod
    def compute_days_until_next_holiday(d):
        d = pd.to_datetime(d)
        holidays = USFederalHolidayCalendar().holidays()
        diffs = (holidays - d).days
        diffs = diffs[diffs >= 0]
        return diffs.min() if len(diffs) > 0 else np.nan
    ####################################################################

    @staticmethod
    def compute_days_since_last_holiday(d):
        d = pd.to_datetime(d)
        holidays = USFederalHolidayCalendar().holidays()
        diffs = (d - holidays).days
        diffs = diffs[diffs >= 0]
        return diffs.min() if len(diffs) > 0 else np.nan
    ####################################################################

    @staticmethod
    def compute_holiday_proximity_index(d, scale=30):
        """
        Returns a smooth value between -1 and +1 depending on
        distance to holidays. Neural networks LOVE this.
        Negative = after holiday
        Positive = before holiday
        """
        before = HolidayFeatures.compute_days_until_next_holiday(d)
        after = HolidayFeatures.compute_days_since_last_holiday(d)
    
        if pd.isna(before) and pd.isna(after):
            return 0
    
        # choose the nearest side (before or after)
        if before <= after:
            return +max(0, (scale - before) / scale)
        else:
            return -max(0, (scale - after) / scale)
    ####################################################################
    
    # @staticmethod
    # def ComputeDaysUntilBirthday(d, bday):
    #     d = pd.to_datetime(d)
    #     bday = pd.to_datetime(bday)
    
    #     this_year = pd.Timestamp(d.year, bday.month, bday.day)
    #     if d <= this_year:
    #         return (this_year - d).days
    #     else:
    #         next_year = pd.Timestamp(d.year + 1, bday.month, bday.day)
    #         return (next_year - d).days
    # ####################################################################
    
    # @staticmethod
    # def ComputeDaysSinceBirthday(d, bday):
    #     d = pd.to_datetime(d)
    #     bday = pd.to_datetime(bday)
    
    #     this_year = pd.Timestamp(d.year, bday.month, bday.day)
    #     if d >= this_year:
    #         return (d - this_year).days
    #     else:
    #         last_year = pd.Timestamp(d.year - 1, bday.month, bday.day)
    #         return (d - last_year).days
    ####################################################################

 