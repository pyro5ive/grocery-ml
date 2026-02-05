import logging
import pandas as pd
import numpy as np

class ItemSupplyLevel_FeatureBuilder:

    itemSupplyLevelRawColName = "itemSupplyLevel_raw";
    itemSupplyLevelClippedFeatColName = "itemSupplyLevel_clipped_feat";

    daysSinceCol: str = "daysSinceLast_Purchase_raw";
    avgGapCol: str = "avgDaysBetween_ItemPurchases_raw";

    producedFeatures = [itemSupplyLevelRawColName, itemSupplyLevelClippedFeatColName];
    requiredFeatures = [ daysSinceCol, avgGapCol ];
    requiredFeatureTypes = {};
    requiredFeatureTypes[daysSinceCol] = pd.api.types.is_numeric_dtype;
    requiredFeatureTypes[avgGapCol] = pd.api.types.is_numeric_dtype;

    def __init__(this, ):
        this.logger = logging.getLogger(this.__class__.__name__);
    #======================================================================#

    def build_feature(this, df):
        this.logger.info("build_feature() start rows=%s", len(df));
        this._validate_required_columns(df);
        this._validate_required_column_types(df);
        df = this._compute_supply_level(df);
        this.logger.info("build_feature() done rows=%s", len(df));
        return df;
    #======================================================================#

    def _compute_supply_level(this, df):
        this.logger.info("_compute_supply_level() start");
        ratio = np.where(
            df[this.avgGapCol] > 0,
            df[this.daysSinceCol] / df[this.avgGapCol],
            0.0
        );
        df[this.itemSupplyLevelRawColName] = 1.0 - ratio;
        df[this.itemSupplyLevelClippedFeatColName] = np.clip(
            df[this.itemSupplyLevelRawColName],
            0.0,
            1.0
        );
        this.logger.info("_compute_supply_level() complete");
        return df;
    #======================================================================#

    def _validate_required_columns(this, df):
        missing = [f for f in this.requiredFeatures if f not in df.columns];
        if missing:
            this.logger.error("_validate_required_columns() missing=%s", missing);
            raise Exception(f"{this.__class__.__name__} missing required columns: {missing}");
        this.logger.info("_validate_required_columns() ok");
    #======================================================================#

    def _validate_required_column_types(this, df):
        for col, validator in this.requiredFeatureTypes.items():
            if not validator(df[col]):
                actualType = str(df[col].dtype);
                this.logger.error("_validate_required_column_types() failed col=%s actualType=%s", col, actualType);
                raise Exception(f"{this.__class__.__name__} column '{col}' failed type validation. actualType={actualType}");
        this.logger.info("_validate_required_column_types() ok");
    #======================================================================#
