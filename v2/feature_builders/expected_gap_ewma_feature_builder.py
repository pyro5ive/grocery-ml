import logging
import pandas as pd
from abstractions.feature_builder_base import FeatureBuilderBase


#======================================================#
class ExpectedGapEwmaFeatureBuilder(FeatureBuilderBase):
    """
    Feature builder that computes the expected days between purchases per item
    using an exponentially weighted moving average (EWMA).
    Produces a single continuous feature column.
    """

    alpha: float = 0.3
    dateCol: str = "date"
    targetCol: str = "didBuy_target"
    itemIdCol: str = "itemId"
    expectedGapEwmaColName: str = "expectedDaysBetween_Purchases_ewma_cont"

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

        self.requiredFeatures = [self.itemIdCol, self.dateCol, self.targetCol]
        self.producedFeatures = [self.expectedGapEwmaColName]
        self.requiredFeatureTypes = {
            self.itemIdCol:  pd.api.types.is_integer_dtype,
            self.dateCol:    pd.api.types.is_datetime64_any_dtype,
            self.targetCol:  pd.api.types.is_bool_dtype
        }
        self.logger.info("ExpectedGapEwmaFeatureBuilder initialized")

    #======================================================#
    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build the expected gap EWMA feature column.

        :param df: Input DataFrame containing itemId, date and didBuy_target columns.
        :type df: pd.DataFrame
        :returns: DataFrame with EWMA feature column added.
        :rtype: pd.DataFrame
        :raises ValueError: If required columns are missing or fail type validation.
        """
        self.logger.info("build(): start rows=%s", len(df))

        self._validate_required_columns(df)
        self._validate_required_column_types(df)

        df = df.sort_values([self.itemIdCol, self.dateCol]).reset_index(drop=True)
        df = self._compute_expected_gap_ewma(df)

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
    def _compute_expected_gap_ewma(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the EWMA of purchase gaps per item by iterating rows chronologically.

        :param df: Input DataFrame sorted by itemId and date.
        :type df: pd.DataFrame
        :returns: DataFrame with EWMA feature column populated.
        :rtype: pd.DataFrame
        """
        df[self.expectedGapEwmaColName] = 0.0
        lastPurchaseDateByItem: dict = {}
        ewmaGapByItem: dict = {}
        rowCount: int = int(len(df))
        i: int = 0

        while i < rowCount:
            itemId = df.at[i, self.itemIdCol]
            currentDate = df.at[i, self.dateCol]
            didBuy = df.at[i, self.targetCol]

            if int(didBuy) == 1:
                if itemId in lastPurchaseDateByItem:
                    gapDays: int = int((currentDate - lastPurchaseDateByItem[itemId]).days)
                    if itemId in ewmaGapByItem:
                        prevEwma: float = float(ewmaGapByItem[itemId])
                        newEwma: float = (self.alpha * float(gapDays)) + ((1.0 - self.alpha) * prevEwma)
                    else:
                        newEwma: float = float(gapDays)
                    ewmaGapByItem[itemId] = float(newEwma)
                    df.at[i, self.expectedGapEwmaColName] = float(newEwma)
                else:
                    df.at[i, self.expectedGapEwmaColName] = 0.0
                lastPurchaseDateByItem[itemId] = currentDate
            else:
                if itemId in ewmaGapByItem:
                    df.at[i, self.expectedGapEwmaColName] = float(ewmaGapByItem[itemId])
                else:
                    df.at[i, self.expectedGapEwmaColName] = 0.0
            i = i + 1

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