import logging
import pandas as pd
import numpy as np
from abstractions.feature_builder_base import FeatureBuilderBase


#======================================================#
class AvgDaysBetweenTripsFeatureBuilder(FeatureBuilderBase):
    """
    Feature builder that computes the average days between shopping trips.
    Derives an expanding mean from the daysSinceLast_Trip_raw column.
    Produces a raw average column and a log1p-transformed continuous feature column.
    """

    daysSinceLastTripRawColName: str = "daysSinceLast_Trip_raw"
    avgDaysBetweenTripsRawColName: str = "avgDaysBetween_Trips_raw"
    avgDaysBetweenTripsTransformedColName: str = "avgDaysBetween_Trips_log1p_cont"

    requiredFeatures: list[str]
    producedFeatures: list[str]
    logger: logging.Logger

    #======================================================#
    def __init__(self):
        """
        Initializes the feature builder.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.requiredFeatures = [self.daysSinceLastTripRawColName]
        self.producedFeatures = [self.avgDaysBetweenTripsRawColName, self.avgDaysBetweenTripsTransformedColName]
        self.logger.info("AvgDaysBetweenTripsFeatureBuilder initialized")

    #======================================================#
    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build the average days between trips feature columns.

        :param df: Input DataFrame containing the daysSinceLast_Trip_raw column.
        :type df: pd.DataFrame
        :returns: DataFrame with raw and log1p feature columns added.
        :rtype: pd.DataFrame
        :raises ValueError: If required columns are missing.
        """
        self.logger.info("build(): start rows=%s", len(df))

        self._validate_required_columns(df)

        df = df.sort_values("date").reset_index(drop=True)
        df = self._compute_avg_days_between_trips(df)
        df[self.avgDaysBetweenTripsTransformedColName] = np.log1p(df[self.avgDaysBetweenTripsRawColName])

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
    def _compute_avg_days_between_trips(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the expanding mean of the daysSinceLast_Trip_raw column.

        :param df: Input DataFrame sorted by date.
        :type df: pd.DataFrame
        :returns: DataFrame with raw average gap column populated.
        :rtype: pd.DataFrame
        """
        expandingMean: pd.Series = df[self.daysSinceLastTripRawColName].expanding().mean().shift(1)
        df[self.avgDaysBetweenTripsRawColName] = expandingMean.fillna(0)
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