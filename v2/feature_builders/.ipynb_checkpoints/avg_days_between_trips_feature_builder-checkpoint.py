
import logging
import pandas as pd


class AvgDaysBetweenTrips_FeatureBuilder:

    requiredFeatures = [];

    producedFeatures = ["daysSinceLastTrip_raw"];
    
    def __init__(this,  reference_date_col):
        self.logger = logging.getLogger(this.__class__.__name__);
        self.dateCol = reference_date_col;
    #-----------------------------------------------------------------#
    
    def build_feature(this, df):
        return targetDf["daysSinceLastTrip_raw"].replace(0, np.nan).expanding().mean().fillna(0)    

    #-----------------------------------------------------------------#

    
    #-----------------------------------------------------------------#
        
    