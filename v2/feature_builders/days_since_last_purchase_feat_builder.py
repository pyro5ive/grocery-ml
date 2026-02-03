
import logging
import pandas as pd
import numpy as np

class DaysSinceLastPurchase_FeatBuilder:

    dateCol = "date"
    featColNameTransformed = "daysSinceLastPurchase_log1p_feat";
    featColNameRaw = "daysSinceLastPurchase_raw";

    requiredFeatures = ["itemId", "didBuy_target"];
    producedFeatures = [featColNameTransformed, featColNameRaw];

    def __init__(this,):
        this.logger = logging.getLogger(this.__class__.__name__);
    ############################################################################
    
    def build_feature(this, df):
        this._validate_required_columns(df)
        df = this._compute_days_since_last_purchase_for_item(df);
        df = this._apply_transform(df);
        return df; 
    ######################################################################
    def _validate_required_columns(this, df):
        missing = [f for f in this.requiredFeatures if f not in df.columns]
        if missing:
            raise Exception(f"{this.__class__.__name__} missing required columns: {missing}")
    ############################################################################

    def _apply_transform(this,df ):
        df[this.featColNameTransformed] = np.log1p(df[this.featColNameRaw]);
        return df;
    ############################################################################

    def _compute_days_since_last_purchase_for_item(this, df):
        df = df.sort_values(["itemId", this.dateCol]).reset_index(drop=True);
        df[this.featColNameRaw] = np.nan;
        last_purchase_date = {}
    
        for i in range(len(df)):
            itemId = df.at[i, "itemId"]
            current_date = df.at[i, this.dateCol]
    
            if itemId in last_purchase_date:
                df.at[i, this.featColNameRaw] = (current_date - last_purchase_date[itemId]).days
            else:
                df.at[i, this.featColNameRaw] = np.nan
    
            if "didBuy_target" in df.columns and df.at[i, "didBuy_target"] == 1:
                last_purchase_date[itemId] = current_date
    
        df[this.featColNameRaw] = df[this.featColNameRaw].fillna(0)
        return df
    #-----------------------------------------------------------------#
        
    