import logging
import pandas as pd
import numpy as np
from abstractions.feature_builder_base import FeatureBuilderBase
from feature_builders.school_features import SchoolFeatures


#======================================================#
class SchoolScheduleFeatureBuilder(FeatureBuilderBase):
    """
    Feature builder that computes school schedule proximity and cycle features.
    Derives days until school start/end, an in-session binary flag,
    and cyclical sin/cos features based on the school year cycle position.
    """

    dateCol: str
    requiredFeatures: list[str]
    producedFeatures: list[str]
    logger: logging.Logger

    #======================================================#
    def __init__(self, dateCol: str = "date"):
        """
        :param dateCol: DataFrame column name containing trip dates.
        :type dateCol: str
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.dateCol = dateCol
        self.requiredFeatures = [self.dateCol]
        self.producedFeatures = [
            "daysUntilSchool_Start_raw",
            "daysUntilSchool_End_raw",
            "isSchoolInSession_bin_feat",
            "schoolCycle_sin_feat",
            "schoolCycle_cos_feat"
        ]
        self.logger.info("SchoolScheduleFeatureBuilder initialized dateCol=%s", self.dateCol)

    #======================================================#
    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build all school schedule feature columns.

        :param df: Input DataFrame containing the date column.
        :type df: pd.DataFrame
        :returns: DataFrame with school schedule feature columns added.
        :rtype: pd.DataFrame
        :raises ValueError: If the date column is missing from the DataFrame.
        """
        self.logger.info("build(): start rows=%s", len(df))

        self._validate_required_columns(df)

        dates: pd.Series = pd.to_datetime(df[self.dateCol])

        df["daysUntilSchool_Start_raw"] = SchoolFeatures.compute_days_until_school_start(dates)
        df["daysUntilSchool_End_raw"] = SchoolFeatures.compute_days_until_school_end(dates)
        df["isSchoolInSession_bin_feat"] = SchoolFeatures.compute_is_school_in_session(dates)

        cyclePos: pd.Series = SchoolFeatures.compute_school_cycle_position(dates)
        df["schoolCycle_sin_feat"] = np.sin(2 * np.pi * cyclePos)
        df["schoolCycle_cos_feat"] = np.cos(2 * np.pi * cyclePos)

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