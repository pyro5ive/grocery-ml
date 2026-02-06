import logging
import sys

import pandas as pd

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
from purchase_event_builders.winn_dixie_events_df_builder import WinnDixieEventsDfBuilder
from negative_sample_builders.same_trip_negative_sample_builder import  SameTripNegativeSampleBuilder
from negative_sample_builders.non_trip_negative_sample_builder import NonTripNegativeSampleBuilder
from sample_filters.combine_same_trip_qty import SameTripQtyCombiner


logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

class TrainingDataBuilder:

    def __init__(this, sources):
        this.logger = logging.getLogger(this.__class__.__name__);
        this.sources = sources;
        this.winndixieDfBuilder = WinnDixieEventsDfBuilder(this.sources);
        this.featureBuilders = [];
        this._register_feature_builders();
        this.sameTripNegativeSampleBuilder = SameTripNegativeSampleBuilder();
        this.nonTripNegativeSampleBuilder = NonTripNegativeSampleBuilder();
        this.itemIdFeatueBuilder = ItemIdFeatureBuilder();
        this.sameTripQtyCombiner = SameTripQtyCombiner();
    #======================================================================#
    def build_df(this):
        this.logger.info("build_df() start");
        df = this._build_event_dfs();
        df = this._build_target_col(df);
        df = this.itemIdFeatueBuilder.build_feature(df);
        df = this._build_negative_samples(df);
        df = this.sameTripQtyCombiner.filter_df(df);
        df = this._apply_feature_pipeline(df);
        this.logger.info("build_training_df() done rows=%s cols=%s", len(df), len(df.columns));

        return df;
    #======================================================================#
    def _build_event_dfs(this):
        this.logger.info("_build_event_dfs()");
        return this.winndixieDfBuilder.build_df();
    #======================================================================#
    def _build_target_col(this, df):
        this.logger.info("_build_target_col()");
        df["didBuy_target"] = True;
        df["didBuy_target"] = df["didBuy_target"].astype(bool);
        return df;
    # ======================================================================#
    def _build_negative_samples(this, df):
        this.logger.info("_build_negative_samples()");
        df = this.sameTripNegativeSampleBuilder.build_samples(df);
        df  = this.nonTripNegativeSampleBuilder.build_samples(df);
        return df;
    # ======================================================================#
    # def _apply_core_fetures(this, df):
    #     this.logger.info("_apply_core_fetures()");
    #     df = this.itemIdFeatueBuilder.build_feature(df);
    #     df = this._build_target_col(df);
    #     return df;

    #======================================================================#
    def _register_feature_builders(this):
        this.logger.info("_register_feature_builders()");
        this.featureBuilders.append(WeatherHistory_FeatureBuilder());
        this.featureBuilders.append(SchoolSchedule_FeatureBuilder());
        this.featureBuilders.append(IsDst_FeatureBuilder());
        this.featureBuilders.append(PaydayProximity_FeatureBuilder("ang", pd.Timestamp("2026-01-23", tz="US/Central")));
        this.featureBuilders.append(PaydayProximity_FeatureBuilder("sjm", pd.Timestamp("2026-01-30", tz="US/Central")));
        this.featureBuilders.append(DaysSinceLastPurchase_FeatBuilder());
        this.featureBuilders.append(AvgDaysBetweenItemPurchases_FeatureBuilder());
        this.featureBuilders.append(DaysSinceLastTrip_FeatureBuilder());
        this.featureBuilders.append(AvgDaysBetweenTrips_FeatureBuilder());
        this.featureBuilders.append(ExpectedGapEwma_FeatureBuilder());
        this.featureBuilders.append(ItemTotalPurchaseCount_FeatureBuilder());
        this.featureBuilders.append(ItemSupplyLevel_FeatureBuilder());
    #======================================================================#
    def _apply_feature_pipeline(this, df):
        this.logger.info("_apply_feature_pipeline() start");
        for builder in this.featureBuilders:
            builderName = builder.__class__.__name__;
            this.logger.info("Applying feature builder: %s", builderName);
            df = builder.build_feature(df);
        this.logger.info("_apply_feature_pipeline() done");
        return df;
    #======================================================================#
