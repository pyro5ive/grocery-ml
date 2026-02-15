import logging
import pandas as pd
from abstractions.feature_builder_base import FeatureBuilderBase


#======================================================#
class WeatherHistoryFeatureBuilder(FeatureBuilderBase):
    """
    Feature builder that merges historical weather data onto the DataFrame by date.
    Loads weather data from a CSV source and joins feelsLike, humidity and precip columns.
    """

    sourcePath: str = "../data/weather/VisualCrossing-70062 2000-01-01 to 2026-23-1.csv"

    requiredFeatures: list[str]
    producedFeatures: list[str]
    dateCol: str
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
        self.producedFeatures = ["feelsLike_cont", "humidity_cont", "precip_cont"]
        self.logger.info("WeatherHistoryFeatureBuilder initialized dateCol=%s", self.dateCol)

    #======================================================#
    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge historical weather data onto the DataFrame by date.

        :param df: Input DataFrame containing the date column.
        :type df: pd.DataFrame
        :returns: DataFrame with weather feature columns merged in.
        :rtype: pd.DataFrame
        :raises ValueError: If the date column is missing from the DataFrame.
        """
        self.logger.info("build(): start rows=%s", len(df))

        self._validate_required_columns(df)

        weatherDf: pd.DataFrame = self._load_weather_df()
        outDf: pd.DataFrame = df.copy()
        outDf[self.dateCol] = pd.to_datetime(outDf[self.dateCol]).dt.normalize()
        outDf = outDf.merge(weatherDf, left_on=self.dateCol, right_index=True, how="left")

        self.logger.info("build(): done rows=%s", len(outDf))
        return outDf

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
    def _load_weather_df(self) -> pd.DataFrame:
        """
        Load and preprocess the weather CSV into a date-indexed DataFrame.

        :returns: DataFrame indexed by date with renamed weather feature columns.
        :rtype: pd.DataFrame
        """
        self.logger.info("_load_weather_df(): loading from path=%s", self.sourcePath)

        df: pd.DataFrame = pd.read_csv(
            self.sourcePath,
            usecols=["datetime", "feelslike", "humidity", "precip"]
        )
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["date"] = df["datetime"].dt.normalize()
        df = df.rename(columns={
            "feelslike": "feelsLike_cont",
            "humidity":  "humidity_cont",
            "precip":    "precip_cont"
        })
        df = df.drop(columns=["datetime"]).groupby("date", as_index=True).mean()

        self.logger.info("_load_weather_df(): done rows=%s", len(df))
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