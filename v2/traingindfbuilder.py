import logging
import pandas as pd

from feature_builders.payday_prox_feature_builder import PaydayProximity_FeatureBuilder
from feature_builders.weather_history_feat_builder import WeatherHistory_FeatureBuilder
from feature_builders.item_supply_level_feature_builder import ItemSupplyLevel_FeatureBuilder
from feature_builders.school_schedule_feat_builder import SchoolSchedule_FeatureBuilder
from feature_builders.days_since_last_purchase_feat_builder import DaysSinceLastPurchase_FeatBuilder
from feature_builders.item_id_feature_builder import ItemIdFeatureBuilder
from feature_builders.avg_days_between_trips_feature_builder import AvgDaysBetweenTrips_FeatureBuilder
from feature_builders.expected_gap_ewma_feature_builder import ExpectedGapEwma_FeatureBuilder
from feature_builders.item_total_purchase_count_feature_builder import ItemTotalPurchaseCount_FeatureBuilder
from feature_builders.is_dst_feature_builder import IsDst_FeatureBuilder
from purchase_event_builders.winn_dixie_events_df_builder import WinnDixieEventsDfBuilder

class TrainingDataBuilder:

    def __init__(this, sources):
        this.logger = logging.getLogger(this.__class__.__name__);
        this.sources = sources;
        this.winndixieDfBuilder = WinnDixieEventsDfBuilder(this.sources);
        this.featureBuilders = [];
        this._register_feature_builders();
    #======================================================================#
    def build_training_df(this):
        this.logger.info("build_training_df() start");
        df = this._build_event_dfs();
        df = this._build_target_col(df);
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
        return df;
    #======================================================================#
    def _register_feature_builders(this):
        this.logger.info("_register_feature_builders()");
        this.featureBuilders.append(WeatherHistory_FeatureBuilder());
        this.featureBuilders.append(SchoolSchedule_FeatureBuilder());
        this.featureBuilders.append(IsDst_FeatureBuilder());
        this.featureBuilders.append(PaydayProximity_FeatureBuilder("ang", pd.Timestamp("2026-01-23", tz="US/Central")));
        this.featureBuilders.append(PaydayProximity_FeatureBuilder("sjm", pd.Timestamp("2026-01-30", tz="US/Central")));
        this.featureBuilders.append(ItemIdFeatureBuilder());
        this.featureBuilders.append(DaysSinceLastPurchase_FeatBuilder());
        this.featureBuilders.append(ItemTotalPurchaseCount_FeatureBuilder());
        this.featureBuilders.append(ExpectedGapEwma_FeatureBuilder());
        this.featureBuilders.append(AvgDaysBetweenTrips_FeatureBuilder("date"));
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
