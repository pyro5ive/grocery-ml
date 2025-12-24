import pandas as pd
class SchoolFeatures:
    
    @staticmethod
    def compute_days_until_school_start(d):
        d = pd.to_datetime(d)
        start = pd.Timestamp(d.year, 8, 15)
        if d <= start:
            return (start - d).days
        else:
            next_start = pd.Timestamp(d.year + 1, 8, 15)
            return (next_start - d).days
    ####################################################################

    @staticmethod
    def compute_days_until_school_end(d):
        d = pd.to_datetime(d)
        end = pd.Timestamp(d.year, 5, 31)
        if d <= end:
            return (end - d).days
        else:
            next_end = pd.Timestamp(d.year + 1, 5, 31)
            return (next_end - d).days
    ####################################################################

    @staticmethod
    def compute_school_season_index(d):
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