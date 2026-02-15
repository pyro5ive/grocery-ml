import logging
import pandas as pd
import numpy as np
from abstractions.feature_builder_base import FeatureBuilderBase


#======================================================#
class AvgDaysBetweenItemPurchasesFeatureBuilder(FeatureBuilderBase):
    """
    Feature builder that computes the average days between item purchases per item ID.
    Produces a raw average column and a log1p-transformed continuous feature column.
    """

    dateCol: str = "date"
    targetCol: str = "didBuy_target"
    itemIdCol: str = "itemId"
    daysSinceCol: str = "daysSinceLast_Purchase_raw"
    avgDaysBetweenItemPurchasesRawColName: str = "avgDaysBetween_ItemPurchases_raw"
    avgDaysBetweenItemPurchasesFeatColName: str = "avgDaysBetween_ItemPurchases_log1p_cont"

    requiredFeatures: list[str]
    requiredFeatureTypes: dict
    producedFeatures: list[str]
    logger: logging.Logger

    #======================================================#
    def __init__(self):
        """
        Initializes the feature builder.
        """
        self.logger = logging.getLogger(self.__class__.__name__)

        self.requiredFeatures = [self.itemIdCol, self.dateCol, self.targetCol, self.daysSinceCol]
        self.producedFeatures = [self.avgDaysBetweenItemPurchasesRawColName, self.avgDaysBetweenItemPurchasesFeatColName]

        self.requiredFeatureTypes = {
            self.itemIdCol:    pd.api.types.is_integer_dtype,
            self.dateCol:      pd.api.types.is_datetime64_any_dtype,
            self.targetCol:    pd.api.types.is_bool_dtype,
            self.daysSinceCol: pd.api.types.is_numeric_dtype
        }

        self.logger.info("AvgDaysBetweenItemPurchasesFeatureBuilder initialized")

    #======================================================#
    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build the average days between item purchases feature columns.

        :param df: Input DataFrame containing required columns.
        :type df: pd.DataFrame
        :returns: DataFrame with raw and log1p feature columns added.
        :rtype: pd.DataFrame
        :raises ValueError: If required columns are missing or fail type validation.
        """
        self.logger.info("build(): start rows=%s", len(df))

        self._validate_required_columns(df)
        self._validate_required_column_types(df)

        df = df.sort_values([self.itemIdCol, self.dateCol]).reset_index(drop=True)
        df = self._compute_avg_gap(df)
        df = self._apply_log1p_transformation(df)

        self.logger.info("build(): done rows=%s", len(df))
        return df

    #======================================================#
    def get_feature_names_in(self) -> list[str]:
        """
        Return the input column names this builder requires.

        :returns: List of required input column names.
        :rtype: list[str]
        """
        return list(self.requiredFeatures)

    #======================================================#
    def get_feature_names_out(self) -> list[str]:
        """
        Return the output column names this builder produces.

        :returns: List of produced feature column names.
        :rtype: list[str]
        """
        return list(self.producedFeatures)

    #======================================================#
    def _compute_avg_gap(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute expanding mean of purchase gaps per item ID and write to raw column.

        :param df: Input DataFrame sorted by itemId and date.
        :type df: pd.DataFrame
        :returns: DataFrame with raw average gap column populated.
        :rtype: pd.DataFrame
        """
        df[self.avgDaysBetweenItemPurchasesRawColName] = 0.0

        for itemId, group in df.groupby(self.itemIdCol):
            idx = group.index
            gaps: pd.Series = group[self.daysSinceCol]
            purchaseMask: pd.Series = group[self.targetCol] == True
            purchaseGaps: pd.Series = gaps.where(purchaseMask)
            expandingMean: pd.Series = purchaseGaps.expanding().mean().shift(1)
            df.loc[idx, self.avgDaysBetweenItemPurchasesRawColName] = expandingMean.fillna(0.0)

        return df

    #======================================================#
    def _apply_log1p_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply log1p transformation to the raw average gap column.

        :param df: Input DataFrame containing the raw average gap column.
        :type df: pd.DataFrame
        :returns: DataFrame with log1p feature column populated.
        :rtype: pd.DataFrame
        """
        df[self.avgDaysBetweenItemPurchasesFeatColName] = np.log1p(
            df[self.avgDaysBetweenItemPurchasesRawColName]
        )
        return df

    #======================================================#
    def _validate_required_columns(self, df: pd.DataFrame) -> None:
        """
        Validate that all required columns are present in the DataFrame.

        :param df: Input DataFrame to validate.
        :type df: pd.DataFrame
        :raises ValueError: If any required columns are missing.
        """
        missing: list[str] = [f for f in self.requiredFeatures if f not in df.columns]
        if missing:
            raise ValueError(f"{self.__class__.__name__} missing required columns: {missing}")

    #======================================================#
    def _validate_required_column_types(self, df: pd.DataFrame) -> None:
        """
        Validate that all required columns pass their type validators.

        :param df: Input DataFrame to validate.
        :type df: pd.DataFrame
        :raises ValueError: If any required column fails type validation.
        """
        for col, validator in self.requiredFeatureTypes.items():
            if not validator(df[col]):
                actualType: str = str(df[col].dtype)
                raise ValueError(
                    f"{self.__class__.__name__} column '{col}' failed type validation. actualType={actualType}"
                )