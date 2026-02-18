import logging
import pandas as pd
import sys

from datetime import datetime
from typing import List, Optional

from abstractions.event_df_builder_base import EventDfBuilderBase
from abstractions.feature_builder_base import FeatureBuilderBase
from abstractions.prediction_feature_builder_base import PredictionFeatureBuilderBase
from models.datasource_paths_config import DataSourcePathsConfig


logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

#======================================================#
class PredictionInputDfBuilder:
    """
    Builds the prediction-time input DataFrame.

    This builder mirrors the training pipeline closely, but:
    - combines historical + live events
    - injects a synthetic prediction-date event
    - applies prediction-specific feature builders
    """

    # ---- injected collaborators ----
    eventsDfBuilders: List[EventDfBuilderBase]
    predictionDfFeatBuilders: List[PredictionFeatureBuilderBase]
    featureBuilders: List[FeatureBuilderBase]

    itemIdFeatureBuilder: FeatureBuilderBase
    weatherForcastFeatureBuilder: PredictionFeatureBuilderBase
    predictionDateEventsDfBuilder: EventDfBuilderBase

    # ---- configuration ----
    trainingPaths: dict[str, str]
    livePaths: dict[str, str]

    # ---- runtime state ----
    predInputDf: Optional[pd.DataFrame]
    historicalEventsDfCache: Optional[pd.DataFrame]
    newPurchaseEventsDfCache: Optional[pd.DataFrame]

    logger: logging.Logger

    #======================================================#
    def __init__(
            self,
            eventsDfBuilders: list[EventDfBuilderBase],
            predictionDfFeatBuilders: list[PredictionFeatureBuilderBase],
            featureBuilders: list[FeatureBuilderBase],
            itemIdFeatureBuilder: FeatureBuilderBase,
            weatherForcastFeatureBuilder: PredictionFeatureBuilderBase,
            predictionDateEventsDfBuilder: EventDfBuilderBase,
            dataSourcePathConfig: DataSourcePathsConfig
    ):
        """
        All dependencies are injected via DI.
        No builders decide whether they are training or live —
        that decision is made here by which path set is passed.
        """

        self.eventsDfBuilders = eventsDfBuilders
        self.predictionDfFeatBuilders = predictionDfFeatBuilders
        self.featureBuilders = featureBuilders

        self.itemIdFeatureBuilder = itemIdFeatureBuilder
        self.weatherForcastFeatureBuilder = weatherForcastFeatureBuilder
        self.predictionDateEventsDfBuilder = predictionDateEventsDfBuilder

        # Split source paths once, centrally
        self.trainingPaths = dataSourcePathConfig.trainingPaths
        self.livePaths = dataSourcePathConfig.livePaths

        self.predInputDf = None
        self.historicalEventsDfCache = None
        self.newPurchaseEventsDfCache = None

        self.logger = logging.getLogger(self.__class__.__name__)
    #======================================================================#
    def build_df(self, predDate: datetime) -> pd.DataFrame:
        """
        Build the full prediction input DataFrame for a given prediction date.
        """
        self.logger.info(
            "Building the prediction input df. Prediction date is %s",
            predDate
        )

        # Build base event rows (no features yet)
        self._build_events_df(predDate)

        # Prediction-time target column (always true)
        self._build_target_col()

        # Item-id mapping must happen before other features
        self.predInputDf = self.itemIdFeatureBuilder.build_feature(
            self.predInputDf
        )

        # Apply shared feature pipeline
        self.predInputDf = self._apply_feature_pipeline(
            self.predInputDf
        )

        # Apply forecast-only features for the latest date
        latestDate: datetime = self.predInputDf["date"].max()
        latestRowsDf: pd.DataFrame = self.predInputDf[
            self.predInputDf["date"] == latestDate
        ]

        self.weatherForcastFeatureBuilder.build_df(
            latestRowsDf,
            latestDate
        )

        return self.predInputDf
    #======================================================================#
    def _build_events_df(self, predDate: datetime) -> pd.DataFrame:
        """
        Build and cache:
        - historical events (training paths)
        - live/new events (live paths)
        - synthetic prediction-date events
        """
        eventDfs: list[pd.DataFrame] = []

        # ---- historical events (cached) ----
        if self.historicalEventsDfCache is None:
            dfs: list[pd.DataFrame] = []

            for builder in self.eventsDfBuilders:
                df: pd.DataFrame = builder.build_df(
                    self.trainingPaths
                )
                if df is not None and not df.empty:
                    dfs.append(df)

            if len(dfs) == 0:
                raise RuntimeError(
                    "No historical events produced"
                )

            self.historicalEventsDfCache = pd.concat(
                dfs,
                ignore_index=True
            )

        # ---- live events (cached) ----
        if self.newPurchaseEventsDfCache is None:
            dfs: list[pd.DataFrame] = []

            for builder in self.eventsDfBuilders:
                df: pd.DataFrame = builder.build_df(
                    self.livePaths
                )
                if df is not None and not df.empty:
                    dfs.append(df)

            self.newPurchaseEventsDfCache = (
                pd.concat(dfs, ignore_index=True)
                if len(dfs) > 0
                else pd.DataFrame()
            )

        # ---- prediction-date events ----
        itemList: list[str] = (
            self.historicalEventsDfCache["item"]
            .dropna()
            .unique()
            .tolist()
        )

        predictionDatesDf: pd.DataFrame = (
            self.predictionDateEventsDfBuilder.build_df(
                predDate,
                itemList
            )
        )

        # ---- combine all events ----
        eventDfs.append(self.historicalEventsDfCache)
        eventDfs.append(self.newPurchaseEventsDfCache)
        eventDfs.append(predictionDatesDf)

        self.predInputDf = pd.concat(
            eventDfs,
            ignore_index=True
        )

        self.predInputDf = (
            self.predInputDf
            .sort_values(["item", "date"])
            .reset_index(drop=True)
        )

        return self.predInputDf
    #======================================================================#
    def _build_target_col(self) -> None:
        """
        Prediction-time target column.
        Always true; used only to keep schema aligned with training.
        """
        self.logger.info("_build_target_col()")

        self.predInputDf["didBuy_target"] = True
        self.predInputDf["didBuy_target"] = (
            self.predInputDf["didBuy_target"]
            .astype(bool)
        )
    #======================================================================#
    def _apply_feature_pipeline(self,df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the shared (training-compatible) feature pipeline.
        """
        self.logger.info("_apply_feature_pipeline() start")

        for builder in self.featureBuilders:
            builderName: str = builder.__class__.__name__
            self.logger.info("Applying feature builder: %s", builderName)
            df = builder.build_feature(df)

        self.logger.info("_apply_feature_pipeline() done")
        return df
    #======================================================================#
