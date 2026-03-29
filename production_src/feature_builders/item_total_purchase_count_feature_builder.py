import logging
import pandas as pd
import numpy as np
from abstractions.feature_builder_base import FeatureBuilderBase


#======================================================#
class ItemTotalPurchaseCountFeatureBuilder(FeatureBuilderBase):
    """
    Feature builder that computes the cumulative purchase count per item up to each row.
    Produces a raw cumulative count column and a log1p-transformed continuous feature column.
    """

    totalPurchaseCountRawColName: str = "itemTotalPurchCountToDate_raw"
    totalPurchaseCountTransformedColName: str = "itemTotalPurchCountToDate_log1p_cont"
    itemIdCol: str = "itemId"
    dateCol: str = "date"
    targetCol: str = "didBuy_target"

    requiredFeatures: list[str]
    producedFeatures: list[str]
    requiredFeatureTypes: dict
    logger: logging.Logger

    #======================================================#
    def __init__(self):
        """
        Initializes the feature builder.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.requiredFeatures = [self.itemIdCol, self.dateCol, self.targetCol]
        self.producedFeatures = [self.totalPurchaseCountRawColName, self.totalPurchaseCountTransformedColName]
        self.requiredFeatureTypes = {
            self.itemIdCol:  pd.api.types.is_integer_dtype,
            self.dateCol:    pd.api.types.is_datetime64_any_dtype,
            self.targetCol:  pd.api.types.is_bool_dtype
        }
        self.logger.info("ItemTotalPurchaseCountFeatureBuilder initialized")

    #======================================================#
    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build the cumulative item purchase count feature columns.

        :param df: Input DataFrame containing itemId, date and didBuy_target columns.
        :type df: pd.DataFrame
        :returns: DataFrame with raw and log1p feature columns added.
        :rtype: pd.DataFrame
        :raises ValueError: If required columns are missing or fail type validation.
        """
        self.logger.info("build(): start rows=%s", len(df))

        self._validate_required_columns(df)
        self._validate_required_column_types(df)

        df = df.sort_values([self.itemIdCol, self.dateCol]).copy()
        df = self._compute_total_purchase_count(df)
        df[self.totalPurchaseCountTransformedColName] = np.log1p(df[self.totalPurchaseCountRawColName])

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
    def _compute_total_purchase_count(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the cumulative purchase count per item up to but not including each row.

        :param df: Input DataFrame sorted by itemId and date.
        :type df: pd.DataFrame
        :returns: DataFrame with raw cumulative purchase count column populated.
        :rtype: pd.DataFrame
        """
        df[self.totalPurchaseCountRawColName] = (
            df.groupby(self.itemIdCol)[self.targetCol]
              .cumsum()
              .shift(1)
              .fillna(0)
              .astype(int)
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