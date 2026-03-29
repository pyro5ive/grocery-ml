import logging
import pandas as pd
from abstractions.feature_builder_base import FeatureBuilderBase


#======================================================#
class WeatherHistoryFeatureBuilder(FeatureBuilderBase):
    """
    Feature builder that merges historical weather data onto the DataFrame by date.
    Loads weather data from a CSV source and joins feelsLike, humidity and precip columns.
    """

    requiredFeatures: list[str]
    producedFeatures: list[str]
    dateCol: str
    sourcePath: str
    logger: logging.Logger

    #======================================================#
    def __init__(self, sourcePath: str, dateCol: str = "date"):
        """
        :param sourcePath: Filesystem path to the historical weather CSV.
        :type sourcePath: str
        :param dateCol: DataFrame column name containing trip dates.
        :type dateCol: str
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.sourcePath = sourcePath
        self.dateCol = dateCol
        self.requiredFeatures = [self.dateCol]
        self.producedFeatures = ["feelsLike_cont", "humidity_cont", "precip_cont"]

        self.logger.info(
            "WeatherHistoryFeatureBuilder initialized sourcePath=%s dateCol=%s",
            self.sourcePath,
            self.dateCol
        )

    #======================================================#
    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge historical weather data onto the DataFrame by date.
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
        return list(self.requiredFeatures)

    #======================================================#
    def get_feature_names_out(self) -> list[str]:
        return list(self.producedFeatures)

    #======================================================#
    def _load_weather_df(self) -> pd.DataFrame:
        """
        Load and preprocess the weather CSV into a date-indexed DataFrame.
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
        missing: list[str] = [f for f in self.requiredFeatures if f not in df.columns]
        if missing:
            raise ValueError(f"{self.__class__.__name__} missing required columns: {missing}")
