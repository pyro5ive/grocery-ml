import logging
import pandas as pd
import numpy as np

class ItemTotalPurchaseCount_FeatureBuilder:
    totalPurchaseCountRawColName = "itemTotalPurchaseCount_raw";
    totalPurchaseCountTransformedColName = "itemTotalPurchaseCount_log1p_feat";
    requiredFeatures = [ "itemId", "date", "didBuy_target" ];
    producedFeatures = [ totalPurchaseCountRawColName, totalPurchaseCountTransformedColName ];
    def __init__(this, item_id_col: str = "itemId", date_col: str = "date", target_col: str = "didBuy_target"):
        this.logger = logging.getLogger(this.__class__.__name__);
        this.itemIdCol = item_id_col;
        this.dateCol = date_col;
        this.targetCol = target_col;
        this.requiredFeatureTypes = {};
        this.requiredFeatureTypes[this.itemIdCol] = pd.api.types.is_integer_dtype;
        this.requiredFeatureTypes[this.dateCol] = pd.api.types.is_datetime64_any_dtype;
        this.requiredFeatureTypes[this.targetCol] = pd.api.types.is_bool_dtype;
    #======================================================================#
    def build_feature(this, df):
        this._validate_required_columns(df);
        this._validate_required_column_types(df);
        df = df.sort_values([ this.itemIdCol, this.dateCol ]).copy();
        df = this._compute_total_purchase_count(df);
        df[this.totalPurchaseCountTransformedColName] = this._apply_log1p(df[this.totalPurchaseCountRawColName]);
        return df;
    #======================================================================#
    def _compute_total_purchase_count(this, df):
        df[this.totalPurchaseCountRawColName] = (
            df.groupby(this.itemIdCol)[this.targetCol]
              .cumsum()
              .shift(1)
              .fillna(0)
              .astype(int)
        );
        return df;
    #======================================================================#
    def _apply_log1p(this, series):
        return np.log1p(series);
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
