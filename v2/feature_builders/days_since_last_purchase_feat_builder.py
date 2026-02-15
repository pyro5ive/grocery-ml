import logging
import pandas as pd
import numpy as np
from abstractions.feature_builder_base import FeatureBuilderBase


#======================================================#
class DaysSinceLastPurchaseFeatureBuilder(FeatureBuilderBase):
    """
    Feature builder that computes the number of days since the last purchase per item.
    Derives values from item purchase history rows in the DataFrame.
    Produces a raw days column and a log1p-transformed continuous feature column.
    """

    dateCol: str = "date"
    itemIdCol: str = "itemId"
    targetCol: str = "didBuy_target"
    featColNameRaw: str = "daysSinceLast_Purchase_raw"
    featColNameTransformed: str = "daysSinceLast_Purchase_log1p_cont"

    requiredFeatures: list[str]
    producedFeatures: list[str]
    logger: logging.Logger

    #======================================================#
    def __init__(self):
        """
        Initializes the feature builder.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.requiredFeatures = [self.itemIdCol, self.targetCol, self.dateCol]
        self.producedFeatures = [self.featColNameRaw, self.featColNameTransformed]
        self.logger.info("DaysSinceLastPurchaseFeatureBuilder initialized")

    #======================================================#
    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build the days since last purchase feature columns.

        :param df: Input DataFrame containing itemId, date and didBuy_target columns.
        :type df: pd.DataFrame
        :returns: DataFrame with raw and log1p feature columns added.
        :rtype: pd.DataFrame
        :raises ValueError: If required columns are missing.
        """
        self.logger.info("build(): start rows=%s", len(df))

        self._validate_required_columns(df)

        df = self._compute_days_since_last_purchase_for_item(df)
        df = self._apply_transform(df)

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
    def _compute_days_since_last_purchase_for_item(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute days since last purchase per item by iterating rows chronologically.

        :param df: Input DataFrame containing itemId, date and didBuy_target columns.
        :type df: pd.DataFrame
        :returns: DataFrame with raw days since last purchase column populated.
        :rtype: pd.DataFrame
        """
        df = df.sort_values([self.itemIdCol, self.dateCol]).reset_index(drop=True)
        df[self.featColNameRaw] = np.nan
        lastPurchaseDateByItem: dict = {}

        for i in range(len(df)):
            itemId = df.at[i, self.itemIdCol]
            currentDate = df.at[i, self.dateCol]

            if itemId in lastPurchaseDateByItem:
                df.at[i, self.featColNameRaw] = (currentDate - lastPurchaseDateByItem[itemId]).days
            else:
                df.at[i, self.featColNameRaw] = np.nan

            if df.at[i, self.targetCol] == 1:
                lastPurchaseDateByItem[itemId] = currentDate

        df[self.featColNameRaw] = df[self.featColNameRaw].fillna(0)
        return df

    #======================================================#
    def _apply_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply log1p transformation to the raw days since last purchase column.

        :param df: Input DataFrame containing the raw days since last purchase column.
        :type df: pd.DataFrame
        :returns: DataFrame with log1p feature column populated.
        :rtype: pd.DataFrame
        """
        df[self.featColNameTransformed] = np.log1p(df[self.featColNameRaw])
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