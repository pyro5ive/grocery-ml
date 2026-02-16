import logging
import pandas as pd
import sys

from datetime import datetime

from abstractions.feature_builder_base import FeatureBuilderBase
from abstractions.prediction_feature_builder_base import PredictionFeatureBuilderBase


logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

class PredictionInputDfBuilder:

    def __init__(
            self,
            predictionDfFeatBuilders: list[PredictionFeatureBuilderBase],
            featureBuilders: list[FeatureBuilderBase],
    ):

        self.predictionDfFeatBuilders = predictionDfFeatBuilders;
        self.featureBuilders = featureBuilders;


        self.predInputDf = None
        self.logger = logging.getLogger(self.__class__.__name__);
        self.newPurchaseEventsDfBuilder = PurchaseEventAggregateBuilder(liveSources);
        self.historicalPurchaseEventsDfBuilder =  PurchaseEventAggregateBuilder(trainingSources);



        self.historicalEventsDfCache = None
        self.newPurchaseEventsDfCache = None


    #======================================================================#
    def build_df(self, predDate: datetime) -> pd.DataFrame:
        self.logger.info(f"Building the prediction input df. Prediction date is {predDate}");
        # build df with just purchase events (no feats)
        self._build_events_df(predDate);
        self._build_target_col();
        self.predInputDf = self.itemIdFeatureBuilder.build_feature(self.predInputDf);
        self.predInputDf = self._apply_feature_pipeline(self.predInputDf);

        latestDate = self.predInputDf["date"].max();
        latestRowsDf = self.predInputDf[self.predInputDf["date"] == latestDate];
        latestRowsDf = self.weatherForcastFeatureBuilder.build_df(latestRowsDf, latestDate);
        # self.predInputDf = self.sameTripQtyCombiner.filter_df(self.predInputDf);
        
        return self.predInputDf;
    #======================================================================#
    def _build_events_df(self, predDate: datetime):
        eventDfs = []
        if self.historicalEventsDfCache is None:
            self.historicalEventsDfCache = self.historicalPurchaseEventsDfBuilder.build_df();
        #
        if self.newPurchaseEventsDfCache is None:
            self.newPurchaseEventsDfCache = self.newPurchaseEventsDfBuilder.build_df()
        #
        itemList = self.historicalEventsDfCache["item"].unique().tolist()
        predictionDatesDf = self.predictionDateEventsDfBuilder.build_df(predDate, itemList)
        #
        eventDfs.append(self.historicalEventsDfCache)
        eventDfs.append(self.newPurchaseEventsDfCache)
        eventDfs.append(predictionDatesDf)
        #
        self.predInputDf = pd.concat(eventDfs, ignore_index=True)
        self.predInputDf = self.predInputDf.sort_values(["item", "date"]).reset_index(drop=True)
        return self.predInputDf;
    #======================================================================#
    def _build_target_col(self):
        self.logger.info("_build_target_col()");
        self.predInputDf["didBuy_target"] = True;
        self.predInputDf["didBuy_target"] = self.predInputDf["didBuy_target"].astype(bool);
    #======================================================================#
    def _apply_feature_pipeline(self, df):
        self.logger.info("_apply_feature_pipeline() start");
        for builder in self.featureBuilders:
            builderName = builder.__class__.__name__;
            self.logger.info("Applying feature builder: %s", builderName);
            df = builder.build_feature(df);
        self.logger.info("_apply_feature_pipeline() done");
        return df;
    #======================================================================#
