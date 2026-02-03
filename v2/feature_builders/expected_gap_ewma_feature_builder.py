import logging
import pandas as pd

class ExpectedGapEwma_FeatureBuilder:

    alpha = 0.3;
    dateCol = "date";
    targetCol = "didBuy_target";
    itemIdCol = "itemId";
    expectedGapEwmaColName = "expectedDaysBetweenPurchases_ewma_feat";

    requiredFeatureTypes = {};
    requiredFeatureTypes[itemIdCol] = pd.api.types.is_integer_dtype;
    requiredFeatureTypes[dateCol] = pd.api.types.is_datetime64_any_dtype;
    requiredFeatureTypes[targetCol] = pd.api.types.is_bool_dtype;

    requiredFeatures = [ itemIdCol, dateCol, targetCol ];
    producedFeatures = [ expectedGapEwmaColName ];

    def __init__(this,  ):
        this.logger = logging.getLogger(this.__class__.__name__);

    #======================================================================#
    def build_feature(this, df):
        this._validate_required_columns(df);
        this._validate_required_column_types(df);
        df = df.sort_values([ this.itemIdCol, this.dateCol ]).reset_index(drop=True);
        df = this._compute_expected_gap_ewma(df);
        return df;
    #======================================================================#
    def _compute_expected_gap_ewma(this, df):
        df[this.expectedGapEwmaColName] = 0.0;
        lastPurchaseDateByItem = {};
        ewmaGapByItem = {};
        rowCount = int(len(df));
        i = 0;
        while i < rowCount:
            itemId = df.at[i, this.itemIdCol];
            currentDate = df.at[i, this.dateCol];
            didBuy = df.at[i, this.targetCol];
            if int(didBuy) == 1:
                if itemId in lastPurchaseDateByItem:
                    gapDays = int((currentDate - lastPurchaseDateByItem[itemId]).days);
                    if itemId in ewmaGapByItem:
                        prevEwma = float(ewmaGapByItem[itemId]);
                        newEwma = (this.alpha * float(gapDays)) + ((1.0 - this.alpha) * prevEwma);
                    else:
                        newEwma = float(gapDays);
                    ewmaGapByItem[itemId] = float(newEwma);
                    df.at[i, this.expectedGapEwmaColName] = float(newEwma);
                else:
                    df.at[i, this.expectedGapEwmaColName] = 0.0;
                lastPurchaseDateByItem[itemId] = currentDate;
            else:
                if itemId in ewmaGapByItem:
                    df.at[i, this.expectedGapEwmaColName] = float(ewmaGapByItem[itemId]);
                else:
                    df.at[i, this.expectedGapEwmaColName] = 0.0;
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