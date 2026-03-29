import logging
import pandas as pd
import sys

from datetime import datetime
from typing import List, Optional

from abstractions.event_df_builder_base import EventDfBuilderBase
from abstractions.feature_builder_base import FeatureBuilderBase
from abstractions.item_id_builder_base import ItemIdBuilderBase
from abstractions.prediction_feature_builder_base import PredictionFeatureBuilderBase
from models.datasource_paths_config import DataSourcePathsConfig
from purchase_event_builders.prediction_date_events_df_builder import PredictionDateEventsDfBuilder

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

    itemIdFeatureBuilder: ItemIdBuilderBase
    weatherForcastFeatureBuilder: PredictionFeatureBuilderBase
    predictionDateEventsDfBuilder: PredictionDateEventsDfBuilder

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
            itemIdFeatureBuilder: ItemIdBuilderBase,
            weatherForcastFeatureBuilder: PredictionFeatureBuilderBase,
            predictionDateEventsDfBuilder: PredictionDateEventsDfBuilder,
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
        self.logger.info("Building the prediction input df. Prediction date is %s",predDate);

        # Build base event rows (no features yet)
        self._build_events_df(predDate);

        # Prediction-time target column (always true)
        self._build_target_col()

        # Item-id mapping must happen before other features
        self.predInputDf = self.itemIdFeatureBuilder.build(self.predInputDf);

        # Apply shared feature pipeline
        self.predInputDf = self._apply_feature_pipeline(self.predInputDf)

        # Apply forecast-only features for the latest date
        latestDate: datetime = self.predInputDf["date"].max()
        latestRowsDf: pd.DataFrame = self.predInputDf[
            self.predInputDf["date"] == latestDate
        ]

        self.weatherForcastFeatureBuilder.build_df(latestRowsDf,latestDate)

        return self.predInputDf
    #======================================================================#
    def _build_events_df(self, predDate: datetime):
        """
        Build and combine all event sources.
        """

        historicalDf = self._build_historical_events();
        liveDf = self._build_live_events();

        itemList: list[str] = historicalDf["item"].dropna().unique().tolist()

        predictionDatesDf = self.predictionDateEventsDfBuilder.build_df(predDate, itemList)

        eventDfs = [historicalDf, liveDf, predictionDatesDf]

        self.predInputDf = (
            pd.concat(eventDfs, ignore_index=True)
            .sort_values(["item", "date"])
            .reset_index(drop=True)
        )

        # return self.predInputDf
    #======================================================================#
    def _build_historical_events(self) -> pd.DataFrame:

        dfs  = []

        if self.historicalEventsDfCache is not None:
            return self.historicalEventsDfCache

        for builder in self.eventsDfBuilders:
            df = builder.build_df(self.trainingPaths)

            if df is not None and not df.empty:
                dfs.append(df)

        if not dfs:
            raise RuntimeError("No historical events produced")

        self.historicalEventsDfCache = pd.concat(dfs, ignore_index=True)

        return self.historicalEventsDfCache
    # ======================================================================#
    def _build_live_events(self) -> pd.DataFrame:

        if self.newPurchaseEventsDfCache is not None:
            return self.newPurchaseEventsDfCache

        dfs: list[pd.DataFrame] = []

        for builder in self.eventsDfBuilders:
            df = builder.build_df(self.livePaths)

            if df is not None and not df.empty:
                dfs.append(df)

        if dfs:
            self.newPurchaseEventsDfCache = pd.concat(dfs, ignore_index=True)
        else:
            self.newPurchaseEventsDfCache = pd.DataFrame()

        return self.newPurchaseEventsDfCache
    # ======================================================================#
    def _build_target_col(self) -> None:
        """
        Prediction-time target column.
        Always true; used only to keep schema aligned with training.
        """
        self.logger.info("_build_target_col()")

        self.predInputDf["didBuy_target"] = True;
        self.predInputDf["didBuy_target"] = self.predInputDf["didBuy_target"].astype(bool);

    #======================================================================#
    def _apply_feature_pipeline(self,df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the shared (training-compatible) feature pipeline.
        """
        self.logger.info("_apply_feature_pipeline() start")

        for builder in self.featureBuilders:
            builderName = builder.__class__.__name__
            self.logger.info("Applying feature builder: %s", builderName)
            df = builder.build(df)

        self.logger.info("_apply_feature_pipeline() done")
        return df
    #======================================================================#
