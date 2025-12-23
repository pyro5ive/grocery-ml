import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

class HolidayFeatures:

    @staticmethod
    def ComputeDaysUntilNextHoliday(d):
        d = pd.to_datetime(d)
        holidays = USFederalHolidayCalendar().holidays()
        diffs = (holidays - d).days
        diffs = diffs[diffs >= 0]
        return diffs.min() if len(diffs) > 0 else np.nan
    ####################################################################

    @staticmethod
    def ComputeDaysSinceLastHoliday(d):
        d = pd.to_datetime(d)
        holidays = USFederalHolidayCalendar().holidays()
        diffs = (d - holidays).days
        diffs = diffs[diffs >= 0]
        return diffs.min() if len(diffs) > 0 else np.nan
    ####################################################################

    @staticmethod
    def ComputeHolidayProximityIndex(d, scale=30):
        """
        Returns a smooth value between -1 and +1 depending on
        distance to holidays. Neural networks LOVE this.
        Negative = after holiday
        Positive = before holiday
        """
        before = HolidayFeatures.ComputeDaysUntilNextHoliday(d)
        after = HolidayFeatures.ComputeDaysSinceLastHoliday(d)
    
        if pd.isna(before) and pd.isna(after):
            return 0
    
        # choose the nearest side (before or after)
        if before <= after:
            return +max(0, (scale - before) / scale)
        else:
            return -max(0, (scale - after) / scale)
    ####################################################################
    
    @staticmethod
    def ComputeDaysUntilBirthday(d, bday):
        d = pd.to_datetime(d)
        bday = pd.to_datetime(bday)
    
        this_year = pd.Timestamp(d.year, bday.month, bday.day)
        if d <= this_year:
            return (this_year - d).days
        else:
            next_year = pd.Timestamp(d.year + 1, bday.month, bday.day)
            return (next_year - d).days
    ####################################################################
    
    @staticmethod
    def ComputeDaysSinceBirthday(d, bday):
        d = pd.to_datetime(d)
        bday = pd.to_datetime(bday)
    
        this_year = pd.Timestamp(d.year, bday.month, bday.day)
        if d >= this_year:
            return (d - this_year).days
        else:
            last_year = pd.Timestamp(d.year - 1, bday.month, bday.day)
            return (d - last_year).days
    ####################################################################

 