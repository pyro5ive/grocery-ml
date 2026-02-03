import logging
import pandas as pd
import numpy as np

class AvgDaysBetweenItemPurchases_FeatureBuilder:

    dateCol = "date";
    targetCol = "didBuy_target";
    itemIdCol = "itemId";
    daysSinceCol = "daysSinceLastPurchase_raw";

    avgDaysBetweenItemPurchasesRawColName = "avgDaysBetweenItemPurchases_raw";
    avgDaysBetweenItemPurchasesFeatColName = "avgDaysBetweenItemPurchases_feat";

    requiredFeatureTypes = {};
    requiredFeatureTypes[itemIdCol] = pd.api.types.is_integer_dtype;
    requiredFeatureTypes[dateCol] = pd.api.types.is_datetime64_any_dtype;
    requiredFeatureTypes[targetCol] = pd.api.types.is_bool_dtype;
    requiredFeatureTypes[daysSinceCol] = pd.api.types.is_numeric_dtype;

    requiredFeatures = [ itemIdCol, dateCol, targetCol, daysSinceCol ];
    producedFeatures = [ avgDaysBetweenItemPurchasesRawColName, avgDaysBetweenItemPurchasesFeatColName ];
    def __init__(this):
        this.logger = logging.getLogger(this.__class__.__name__);
    #======================================================================#
    def build_feature(this, df):
        this._validate_required_columns(df);
        this._validate_required_column_types(df);
        df = df.sort_values([ this.itemIdCol, this.dateCol ]).reset_index(drop=True);
        df = this._compute_avg_gap(df);
        return df;
    #======================================================================#
    def _compute_avg_gap(this, df):
        df[this.avgDaysBetweenItemPurchasesRawColName] = 0.0;
        df[this.avgDaysBetweenItemPurchasesFeatColName] = 0.0;
        grouped = df.groupby(this.itemIdCol);
        for itemId, group in grouped:
            idx = group.index;
            gaps = group[this.daysSinceCol];
            purchaseMask = group[this.targetCol] == True;
            purchaseGaps = gaps.where(purchaseMask);
            expandingMean = purchaseGaps.expanding().mean().shift(1);
            df.loc[idx, this.avgDaysBetweenItemPurchasesRawColName] = expandingMean.fillna(0.0);
        df[this.avgDaysBetweenItemPurchasesFeatColName] = np.log1p(df[this.avgDaysBetweenItemPurchasesRawColName]);
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
