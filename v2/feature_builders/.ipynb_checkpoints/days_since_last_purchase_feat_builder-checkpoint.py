
import logging
import pandas as pd


class DaysSinceLastPurchase_FeatBuilder:

    requiredFeatures = ["itemId", "didBuy_target"];

    producedFeatures = ["daysSinceLastPurchase_feat"];
    
    def __init__(this,  dateCol: str = "date"):
        this.logger = logging.getLogger(this.__class__.__name__);
        this.dateCol = dateCol;
    #-----------------------------------------------------------------#
    
    def build_feature(this, df):
        this._validate_required_columns(df)
        df = this._compute_days_since_last_purchase_for_item(df);
        return df; 
    #-----------------------------------------------------------------#
    def _validate_required_columns(this, df):
        missing = [f for f in this.requiredFeatures if f not in df.columns]
        if missing:
            raise Exception(f"{this.__class__.__name__} missing required columns: {missing}")
    #-----------------------------------------------------------------#
    
    def _compute_days_since_last_purchase_for_item(this, df):
        df = df.sort_values(["itemId", reference_date_col]).reset_index(drop=True)
        df[colName] = np.nan
        last_purchase_date = {}
    
        for i in range(len(df)):
            itemId = df.at[i, "itemId"]
            current_date = df.at[i, reference_date_col]
    
            if itemId in last_purchase_date:
                df.at[i, colName] = (current_date - last_purchase_date[itemId]).days
            else:
                df.at[i, colName] = np.nan
    
            if "didBuy_target" in df.columns and df.at[i, "didBuy_target"] == 1:
                last_purchase_date[itemId] = current_date
    
        df[colName] = df[colName].fillna(0)
        return df
    #-----------------------------------------------------------------#
        
    