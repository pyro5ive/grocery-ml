import logging
import pandas as pd
import numpy as np
from datetime import timedelta
from abstractions.feature_builder_base import FeatureBuilderBase


#======================================================#
class PaydayProximityFeatureBuilder(FeatureBuilderBase):
    """
    Feature builder that computes payday proximity features for a named person.
    Derives cyclical sin/cos features, scaled proximity, and a binary isPayday flag
    based on a biweekly pay cycle anchored to a known payday date.
    """

    personName: str
    anchorPayday: pd.Timestamp
    dateCol: str
    cycleLength: int
    rawCol: str
    proximityCol: str
    scaledCol: str
    sinCol: str
    cosCol: str
    isPaydayCol: str
    requiredFeatures: list[str]
    producedFeatures: list[str]
    logger: logging.Logger

    #======================================================#
    def __init__(self, personName: str, anchorPayday: pd.Timestamp, dateCol: str = "date"):
        """
        :param personName: Name of the person whose pay cycle is being modeled.
        :type personName: str
        :param anchorPayday: A known payday date to anchor the biweekly cycle.
        :type anchorPayday: pd.Timestamp
        :param dateCol: DataFrame column name containing trip dates.
        :type dateCol: str
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.personName = personName
        self.anchorPayday = pd.Timestamp(anchorPayday).tz_localize(None)
        self.dateCol = dateCol
        self.cycleLength = 14
        self.rawCol = f"payday_{self.personName}_raw"
        self.proximityCol = f"payday_proximity_{self.personName}"
        self.scaledCol = f"payday_proximity_{self.personName}_scaled_cont"
        self.sinCol = f"payday_proximity_{self.personName}_sin_feat"
        self.cosCol = f"payday_proximity_{self.personName}_cos_feat"
        self.isPaydayCol = f"isPayday_{self.personName}_bin_feat"
        self.requiredFeatures = [self.dateCol]
        self.producedFeatures = [
            self.rawCol,
            self.proximityCol,
            self.scaledCol,
            self.sinCol,
            self.cosCol,
            self.isPaydayCol
        ]
        self.logger.info(
            "PaydayProximityFeatureBuilder initialized personName=%s anchorPayday=%s",
            self.personName,
            self.anchorPayday
        )

    #======================================================#
    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build all payday proximity feature columns.

        :param df: Input DataFrame containing the date column.
        :type df: pd.DataFrame
        :returns: DataFrame with all payday proximity feature columns added.
        :rtype: pd.DataFrame
        :raises ValueError: If the date column is missing from the DataFrame.
        """
        self.logger.info("build(): start rows=%s personName=%s", len(df), self.personName)

        self._validate_required_columns(df)

        df = df.copy()
        df[self.dateCol] = pd.to_datetime(df[self.dateCol]).dt.tz_localize(None)
        df = self._build_proximity(df)

        df[self.scaledCol] = df[self.proximityCol] / float(self.cycleLength)

        angle: pd.Series = 2.0 * np.pi * (df[self.proximityCol] / float(self.cycleLength))
        df[self.sinCol] = np.sin(angle)
        df[self.cosCol] = np.cos(angle)
        df[self.isPaydayCol] = (df[self.proximityCol] == 0).astype(bool)

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
    def _build_proximity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the nearest payday date and absolute proximity in days for each row.

        :param df: Input DataFrame containing the date column.
        :type df: pd.DataFrame
        :returns: DataFrame with raw nearest payday and proximity columns populated.
        :rtype: pd.DataFrame
        """
        df[self.rawCol] = df[self.dateCol].apply(self._nearest_payday)
        df[self.proximityCol] = (df[self.rawCol] - df[self.dateCol]).abs().dt.days
        return df

    #======================================================#
    def _nearest_payday(self, currentDate: pd.Timestamp) -> pd.Timestamp:
        """
        Compute the nearest payday date to the given date based on the biweekly cycle.

        :param currentDate: The date to find the nearest payday for.
        :type currentDate: pd.Timestamp
        :returns: The nearest payday date.
        :rtype: pd.Timestamp
        """
        currentDate = pd.Timestamp(currentDate).tz_localize(None)
        daysDiff: int = (currentDate - self.anchorPayday).days
        cycleOffset: int = int(round(daysDiff / float(self.cycleLength)))
        return self.anchorPayday + timedelta(days=cycleOffset * self.cycleLength)

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