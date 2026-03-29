import logging
import pandas as pd
import numpy as np
from abstractions.feature_builder_base import FeatureBuilderBase


#======================================================#
class DaysSinceLastTripFeatureBuilder(FeatureBuilderBase):
    """
    Feature builder that computes the number of days since the last shopping trip.
    Operates on unique trip dates and merges results back onto the full DataFrame.
    Produces a raw days column and a log1p-transformed continuous feature column.
    """

    dateCol: str = "date"
    daysSinceLastTripRawColName: str = "daysSinceLast_Trip_raw"
    daysSinceLastTripTransformedColName: str = "daysSinceLast_Trip_log1p_cont"

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
        self.requiredFeatures = [self.dateCol]
        self.requiredFeatureTypes = {self.dateCol: pd.api.types.is_datetime64_any_dtype}
        self.producedFeatures = [self.daysSinceLastTripRawColName, self.daysSinceLastTripTransformedColName]
        self.logger.info("DaysSinceLastTripFeatureBuilder initialized")

    #======================================================#
    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build the days since last trip feature columns.
        Deduplicates by date, computes on unique trips, then merges back onto the full DataFrame.

        :param df: Input DataFrame containing the date column.
        :type df: pd.DataFrame
        :returns: DataFrame with raw and log1p feature columns added.
        :rtype: pd.DataFrame
        :raises ValueError: If required columns are missing or fail type validation.
        """
        self.logger.info("build(): start rows=%s", len(df))

        self._validate_required_columns(df)
        self._validate_required_column_types(df)

        tripDf: pd.DataFrame = df[[self.dateCol]].drop_duplicates().sort_values(self.dateCol)
        tripDf = self._compute(tripDf)
        tripDf[self.daysSinceLastTripTransformedColName] = np.log1p(tripDf[self.daysSinceLastTripRawColName])

        mergedDf: pd.DataFrame = df.merge(tripDf, on=self.dateCol, how="left")

        self.logger.info("build(): done rows=%s", len(mergedDf))
        return mergedDf

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
    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute days since last trip using date diff on unique trip dates.

        :param df: DataFrame of unique trip dates sorted chronologically.
        :type df: pd.DataFrame
        :returns: DataFrame with raw days since last trip column populated.
        :rtype: pd.DataFrame
        """
        df[self.daysSinceLastTripRawColName] = df[self.dateCol].diff().dt.days.fillna(0)
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