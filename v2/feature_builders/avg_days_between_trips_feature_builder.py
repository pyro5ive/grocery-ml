import logging
import pandas as pd
import numpy as np

class AvgDaysBetweenTrips_FeatureBuilder:

    daysSinceLastTripRawColName = "daysBetweenTrips_raw";
    avgDaysBetweenTripsRawColName = "avgDaysBetweenTrips_raw";
    avgDaysBetweenTripsTransformedColName = "avgDaysBetweenTrips_log1p_feat";

    requiredFeatures = [ daysSinceLastTripRawColName ];
    producedFeatures = [ avgDaysBetweenTripsTransformedColName, avgDaysBetweenTripsRawColName ];

    def __init__(this):
        this.logger = logging.getLogger(this.__class__.__name__);
    #======================================================================#
    def build_feature(this, df):
        this._validate_required_columns(df);
        df = df.sort_values("date");
        df = this._compute_avg_days_between_trips(df);
        df[this.avgDaysBetweenTripsTransformedColName] = this._apply_log1p(df[this.avgDaysBetweenTripsRawColName]);
        return df;
    #======================================================================#

    def _validate_required_columns(this, df):
        missing = [f for f in this.requiredFeatures if f not in df.columns];
        if missing:
            raise Exception(f"{this.__class__.__name__} missing required columns: {missing}");
    #======================================================================#

    def _compute_avg_days_between_trips(this, df):
        expandingMean = df[this.daysSinceLastTripRawColName].expanding().mean().shift(1);
        df[this.avgDaysBetweenTripsRawColName] = expandingMean.fillna(0);
        return df;
    #======================================================================#

    def _apply_log1p(this, series):
        return np.log1p(series);
    #======================================================================#
