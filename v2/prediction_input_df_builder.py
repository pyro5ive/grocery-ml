import logging
import pandas as pd
import sys
from feature_builders.payday_prox_feature_builder import PaydayProximity_FeatureBuilder
from feature_builders.weather_history_feat_builder import WeatherHistory_FeatureBuilder
from feature_builders.item_supply_level_feature_builder import ItemSupplyLevel_FeatureBuilder
from feature_builders.school_schedule_feat_builder import SchoolSchedule_FeatureBuilder
from feature_builders.days_since_last_purchase_feat_builder import DaysSinceLastPurchase_FeatBuilder
from feature_builders.item_id_feature_builder import ItemIdFeatureBuilder
from feature_builders.days_since_last_trip_feature_builder import DaysSinceLastTrip_FeatureBuilder
from feature_builders.avg_days_between_item_purchases_feature_builder import AvgDaysBetweenItemPurchases_FeatureBuilder
from feature_builders.avg_days_between_trips_feature_builder import AvgDaysBetweenTrips_FeatureBuilder
from feature_builders.expected_gap_ewma_feature_builder import ExpectedGapEwma_FeatureBuilder
from feature_builders.item_total_purchase_count_feature_builder import ItemTotalPurchaseCount_FeatureBuilder
from feature_builders.is_dst_feature_builder import IsDst_FeatureBuilder
from sample_filters.combine_same_trip_qty import SameTripQtyCombiner
from datetime import datetime
from purchase_event_builders.purchase_event_aggregate_builder import PurchaseEventAggregateBuilder
from purchase_event_builders.prediction_date_events_df_builder import PredictionDateEventsDfBuilder
from feature_builders.weather_forecast_feature_builder import WeatherForecastFeatureBuilder
from services.weather.weather_service import NwsWeatherService

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

class PredictionInputDfBuilder:

    def __init__(self, liveSources, trainingSources):

        self.featureBuilders = [];
        self.predInputDf = None
        self.logger = logging.getLogger(self.__class__.__name__);
        self.newPurchaseEventsDfBuilder = PurchaseEventAggregateBuilder(liveSources);
        self.historicalPurchaseEventsDfBuilder =  PurchaseEventAggregateBuilder(trainingSources);
        self._register_feature_builders();
        self.itemIdFeatureBuilder = ItemIdFeatureBuilder();
        self.sameTripQtyCombiner = SameTripQtyCombiner();
        self.predictionDateEventsDfBuilder = PredictionDateEventsDfBuilder();
        self.historicalEventsDfCache = None
        self.newPurchaseEventsDfCache = None
        self.weatherForcastService = None
        self.weatherService = NwsWeatherService();
        self.weatherForcastFeatureBuilder = WeatherForecastFeatureBuilder(self.weatherService, 29.9934, -90.2580);

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
    # ======================================================================#

    #======================================================================#
    def _register_feature_builders(self):
        self.logger.info("_register_feature_builders()");
        self.featureBuilders.append(WeatherHistory_FeatureBuilder());
        self.featureBuilders.append(SchoolSchedule_FeatureBuilder());
        self.featureBuilders.append(IsDst_FeatureBuilder());
        self.featureBuilders.append(PaydayProximity_FeatureBuilder("ang", pd.Timestamp("2026-01-23", tz="US/Central")));
        self.featureBuilders.append(PaydayProximity_FeatureBuilder("sjm", pd.Timestamp("2026-01-30", tz="US/Central")));
        self.featureBuilders.append(DaysSinceLastPurchase_FeatBuilder());
        self.featureBuilders.append(AvgDaysBetweenItemPurchases_FeatureBuilder());
        self.featureBuilders.append(DaysSinceLastTrip_FeatureBuilder());
        self.featureBuilders.append(AvgDaysBetweenTrips_FeatureBuilder());
        self.featureBuilders.append(ExpectedGapEwma_FeatureBuilder());
        self.featureBuilders.append(ItemTotalPurchaseCount_FeatureBuilder());
        self.featureBuilders.append(ItemSupplyLevel_FeatureBuilder());
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
