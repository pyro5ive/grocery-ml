import logging
import pandas as pd
import pytz
from abstractions.feature_builder_base import FeatureBuilderBase


#======================================================#
class IsDstFeatureBuilder(FeatureBuilderBase):
    """
    Feature builder that computes whether a date falls within daylight saving time.
    Produces a single binary boolean feature column.
    """

    dateCol: str = "date"
    isDstColName: str = "isDst_bin_feat"
    timeZoneName: str = "America/Chicago"

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
        self.producedFeatures = [self.isDstColName]
        self.logger.info("IsDstFeatureBuilder initialized")

    #======================================================#
    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build the isDst binary feature column.

        :param df: Input DataFrame containing the date column.
        :type df: pd.DataFrame
        :returns: DataFrame with isDst binary feature column added.
        :rtype: pd.DataFrame
        :raises ValueError: If required columns are missing or fail type validation.
        """
        self.logger.info("build(): start rows=%s", len(df))

        self._validate_required_columns(df)
        self._validate_required_column_types(df)

        df = self._compute_is_dst(df)

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
    def _compute_is_dst(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute whether each date falls within daylight saving time for the configured timezone.

        :param df: Input DataFrame containing the date column.
        :type df: pd.DataFrame
        :returns: DataFrame with isDst binary feature column populated.
        :rtype: pd.DataFrame
        """
        tzObj = pytz.timezone(self.timeZoneName)
        df[self.isDstColName] = False
        rowCount: int = int(len(df))
        i: int = 0

        while i < rowCount:
            currentDate = df.at[i, self.dateCol]
            localizedDate = tzObj.localize(currentDate)
            df.at[i, self.isDstColName] = localizedDate.dst() != pd.Timedelta(0)
            i = i + 1

        df[self.isDstColName] = df[self.isDstColName].astype(bool)
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