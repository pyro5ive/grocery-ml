import logging
import pandas as pd
import numpy as np
from abstractions.feature_builder_base import FeatureBuilderBase


#======================================================#
class ItemSupplyLevelFeatureBuilder(FeatureBuilderBase):
    """
    Feature builder that computes the supply level of an item based on days since
    last purchase relative to the average purchase gap.
    Produces a raw supply level column and a clipped continuous feature column.
    """

    itemSupplyLevelRawColName: str = "itemSupplyLevel_raw"
    itemSupplyLevelClippedFeatColName: str = "itemSupplyLevel_clipped_cont"
    daysSinceCol: str = "daysSinceLast_Purchase_raw"
    avgGapCol: str = "avgDaysBetween_ItemPurchases_raw"

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
        self.requiredFeatures = [self.daysSinceCol, self.avgGapCol]
        self.producedFeatures = [self.itemSupplyLevelRawColName, self.itemSupplyLevelClippedFeatColName]
        self.requiredFeatureTypes = {
            self.daysSinceCol: pd.api.types.is_numeric_dtype,
            self.avgGapCol:    pd.api.types.is_numeric_dtype
        }
        self.logger.info("ItemSupplyLevelFeatureBuilder initialized")

    #======================================================#
    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build the item supply level feature columns.

        :param df: Input DataFrame containing daysSinceLast_Purchase_raw and avgDaysBetween_ItemPurchases_raw columns.
        :type df: pd.DataFrame
        :returns: DataFrame with raw and clipped supply level feature columns added.
        :rtype: pd.DataFrame
        :raises ValueError: If required columns are missing or fail type validation.
        """
        self.logger.info("build(): start rows=%s", len(df))

        self._validate_required_columns(df)
        self._validate_required_column_types(df)

        df = self._compute_supply_level(df)

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
    def _compute_supply_level(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute supply level as 1 minus the ratio of days since last purchase to average gap.
        Clips the result between 0.0 and 1.0.

        :param df: Input DataFrame containing required columns.
        :type df: pd.DataFrame
        :returns: DataFrame with raw and clipped supply level columns populated.
        :rtype: pd.DataFrame
        """
        self.logger.info("_compute_supply_level(): start")

        ratio: np.ndarray = np.where(
            df[self.avgGapCol] > 0,
            df[self.daysSinceCol] / df[self.avgGapCol],
            0.0
        )
        df[self.itemSupplyLevelRawColName] = 1.0 - ratio
        df[self.itemSupplyLevelClippedFeatColName] = np.clip(
            df[self.itemSupplyLevelRawColName],
            0.0,
            1.0
        )

        self.logger.info("_compute_supply_level(): done")
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
            self.logger.error("_validate_required_columns(): missing=%s", missing)
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
                self.logger.error("_validate_required_column_types(): failed col=%s actualType=%s", col, actualType)
                raise ValueError(
                    f"{self.__class__.__name__} column '{col}' failed type validation. actualType={actualType}"
                )