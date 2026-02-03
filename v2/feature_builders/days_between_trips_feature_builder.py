
import logging
import pandas as pd
import numpy as np

class DaysBetweenTrips_FeatureBuilder:

    daysSinceLastTripRawColName = "daysBetweenTrips_raw";
    daysSinceLastTripTransformedColName = "daysBetweenTrips_log1p_feat";

    requiredFeatures = ["date"];
    producedFeatures = [ daysSinceLastTripTransformedColName, daysSinceLastTripRawColName ];

    def __init__(this,  reference_date_col):
        this.logger = logging.getLogger(this.__class__.__name__);
        this.dateCol = reference_date_col;
        this.requiredFeatures.append(reference_date_col);
        this.requiredFeatureTypes[reference_date_col] = pd.api.types.is_datetime64_any_dtype
    #-----------------------------------------------------------------#
    
    def build_feature(this, df):
        this._validate_required_columns(df);
        this._validate_required_column_types(df);
        tripDf = df[[this.dateCol]].drop_duplicates().sort_values(this.dateCol);
        tripDf = this._compute(tripDf);
        tripDf[this.daysSinceLastTripTransformedColName] = this._apply_log1p(tripDf[this.daysSinceLastTripRawColName])
        mergedDf = df.merge(tripDf, on=this.dateCol, how="left")
        return mergedDf;
    #-----------------------------------------------------------------#
    def _validate_required_columns(this, df):
        missing = [f for f in this.requiredFeatures if f not in df.columns]
        if missing:
            raise Exception(
                f"{this.__class__.__name__} missing required columns: {missing}"
            )
    # -----------------------------------------------------------------#
    def _validate_required_column_types(this, df):
        for col, validator in this.requiredFeatureTypes.items():
            if not validator(df[col]):
                actualType = str(df[col].dtype)
                raise Exception(
                    f"{this.__class__.__name__} column '{col}' failed type validation. actualType={actualType}"
                )
    #-----------------------------------------------------------------#
    def _compute(this, df):
        df[this.daysSinceLastTripRawColName] = df[this.dateCol].diff().dt.days.fillna(0);
        return df;
    # -----------------------------------------------------------------#
    def _apply_log1p(this, series):
        return np.log1p(series)
    # --------------------------#