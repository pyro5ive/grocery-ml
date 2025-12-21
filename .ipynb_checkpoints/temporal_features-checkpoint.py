import numpy as np

class TemporalFeatures:

    @staticmethod
    # ================================================
    # FREQUENCY WINDOWS (7, 15, 30, 90, 365)
    # True rolling-window implementation
    # ================================================
   
    #######################################################
    
    @staticmethod
    def DaysSinceLastTrip(grouped):
        return grouped["date"].diff().dt.days.fillna(0)
    #######################################################

    @staticmethod
    def AvgDaysBetweenTrips(grouped):
        return grouped["daysSinceLastTrip"].replace(0, np.nan).expanding().mean().fillna(0)    
    #######################################################
    @staticmethod
    def CreateDateFeatures(grouped):
        dt = grouped["date"]
        grouped["year"]    = dt.dt.year
        grouped["month"]   = dt.dt.month
        grouped["day"]     = dt.dt.day
        grouped["dow"]     = dt.dt.dayofweek
        grouped["doy"]     = dt.dt.dayofyear
        grouped["quarter"] = dt.dt.quarter
        return grouped;
    #######################################################
    @staticmethod
    def encode_sin_cos(value, period):
        angle = 2 * np.pi * value / period
        return np.sin(angle), np.cos(angle)
####################################################################################################
    