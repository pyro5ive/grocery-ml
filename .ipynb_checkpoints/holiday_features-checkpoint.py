import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

class HolidayFeatures:

    @staticmethod
    def daysUntilNextHoliday(d):
        d = pd.to_datetime(d)
        holidays = USFederalHolidayCalendar().holidays()
        diffs = (holidays - d).days
        diffs = diffs[diffs >= 0]
        return diffs.min() if len(diffs) > 0 else np.nan
    ####################################################################

    @staticmethod
    def daysSinceLastHoliday(d):
        d = pd.to_datetime(d)
        holidays = USFederalHolidayCalendar().holidays()
        diffs = (d - holidays).days
        diffs = diffs[diffs >= 0]
        return diffs.min() if len(diffs) > 0 else np.nan
    ####################################################################

    @staticmethod
    def holidayProximityIndex(d, scale=30):
        """
        Returns a smooth value between -1 and +1 depending on
        distance to holidays. Neural networks LOVE this.
        Negative = after holiday
        Positive = before holiday
        """
        before = HolidayFeatures.daysUntilNextHoliday(d)
        after = HolidayFeatures.daysSinceLastHoliday(d)
    
        if pd.isna(before) and pd.isna(after):
            return 0
    
        # choose the nearest side (before or after)
        if before <= after:
            return +max(0, (scale - before) / scale)
        else:
            return -max(0, (scale - after) / scale)
    ####################################################################
    
    @staticmethod
    def daysUntilBirthday(d, bday):
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
    def daysSinceBirthday(d, bday):
        d = pd.to_datetime(d)
        bday = pd.to_datetime(bday)
    
        this_year = pd.Timestamp(d.year, bday.month, bday.day)
        if d >= this_year:
            return (d - this_year).days
        else:
            last_year = pd.Timestamp(d.year - 1, bday.month, bday.day)
            return (d - last_year).days


    def daysUntilSchoolStart(d):
        d = pd.to_datetime(d)
        start = pd.Timestamp(d.year, 8, 15)
        if d <= start:
            return (start - d).days
        else:
            next_start = pd.Timestamp(d.year + 1, 8, 15)
            return (next_start - d).days
    ####################################################################

    @staticmethod
    def daysUntilSchoolEnd(d):
        d = pd.to_datetime(d)
        end = pd.Timestamp(d.year, 5, 31)
        if d <= end:
            return (end - d).days
        else:
            next_end = pd.Timestamp(d.year + 1, 5, 31)
            return (next_end - d).days
    ####################################################################

    @staticmethod
    def schoolSeasonIndex(d):
        """
        Smooth 0→1 curve inside school season.
        <0 before season, >1 after.
        Good for neural nets.
        """
        d = pd.to_datetime(d)
        start = pd.Timestamp(d.year, 8, 15)
        end   = pd.Timestamp(d.year, 5, 31)
    
        # If date is after Dec, school season continues in Jan–May.
        if d < start:
            return -((start - d).days) / 365.0
        elif start <= d <= end:
            return (d - start).days / (end - start).days
        else:
            return (d - end).days / 365.0
    
    ####################################################################