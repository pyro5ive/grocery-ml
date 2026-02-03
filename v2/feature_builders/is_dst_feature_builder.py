import logging
import pandas as pd
import pytz

class IsDst_FeatureBuilder:
    isDstColName = "isDst_feat";
    requiredFeatures = [ "date" ];
    producedFeatures = [ isDstColName ];
    def __init__(this, date_col: str = "date", tz: str = "America/Chicago"):
        this.logger = logging.getLogger(this.__class__.__name__);
        this.dateCol = date_col;
        this.timezoneName = tz;
        this.requiredFeatureTypes = {};
        this.requiredFeatureTypes[this.dateCol] = pd.api.types.is_datetime64_any_dtype;
    #======================================================================#
    def build_feature(this, df):
        this._validate_required_columns(df);
        this._validate_required_column_types(df);
        df = this._compute_is_dst(df);
        return df;
    #======================================================================#
    def _compute_is_dst(this, df):
        tzObj = pytz.timezone(this.timezoneName);
        df[this.isDstColName] = 0;
        rowCount = int(len(df));
        i = 0;
        while i < rowCount:
            currentDate = df.at[i, this.dateCol];
            localizedDate = tzObj.localize(currentDate);
            if localizedDate.dst() != pd.Timedelta(0):
                df.at[i, this.isDstColName] = 1;
            else:
                df.at[i, this.isDstColName] = 0;
            i = i + 1;
        return df;
    #======================================================================#
    def _validate_required_columns(this, df):
        missing = [f for f in this.requiredFeatures if f not in df.columns];
        if missing:
            raise Exception(f"{this.__class__.__name__} missing required columns: {missing}");
    #======================================================================#
    def _validate_required_column_types(this, df):
        for col, validator in this.requiredFeatureTypes.items():
            if not validator(df[col]):
                actualType = str(df[col].dtype);
                raise Exception(f"{this.__class__.__name__} column '{col}' failed type validation. actualType={actualType}");
    #======================================================================#
